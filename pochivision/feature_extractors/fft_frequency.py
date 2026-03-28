"""FFT（高速フーリエ変換）周波数領域特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from pochivision.capturelib.log_manager import LogManager

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("fft")
class FFTFrequencyExtractor(BaseFeatureExtractor):
    """
    画像のFFT（高速フーリエ変換）周波数領域特徴量を抽出するクラス.

    FFTは画像の周波数成分を解析し, テクスチャや周期性の特徴を抽出する.
    空間領域の画像を周波数領域に変換することで, 画像の周波数特性を定量化できる.

    前処理:
    - グレースケール変換後, float64 のまま処理 (コントラスト情報を保持)
    - Hanning 窓関数を適用 (画像境界のスペクトルリークを抑制)
    - DC 成分 (平均輝度) を除外し, AC 成分のみで特徴量を計算

    抽出する特徴量:
    - high_low_ratio: 高周波/低周波エネルギー比 [ratio]
    - spectral_std: スペクトルの標準偏差 [cycle/mm or cycle/pixel]
    - horizontal_energy: 水平方向エネルギー [ratio]
    - vertical_energy: 垂直方向エネルギー [ratio]
    - num_peaks: スペクトルピーク数 [count]
    - max_peak_amp: 最大ピーク振幅 [amplitude]
    - band_energy: 周波数帯域別エネルギー [ratio]
    - spectral_centroid: スペクトル重心 [cycle/mm or cycle/pixel]
    - spectral_entropy: スペクトラム全体の正規化エントロピー [normalized, 0-1]
    - horizontal_entropy: 水平方向の正規化エントロピー [normalized, 0-1]
    - vertical_entropy: 垂直方向の正規化エントロピー [normalized, 0-1]
    - band_entropy: 周波数帯域別の正規化エントロピー [normalized, 0-1]

    設計上の制約:
    - 最小画像サイズ: 4x4. それ未満は ValueError を送出.
    - Hanning 窓により画像端のピクセルがゼロに減衰するため,
      max_peak_amp 等の絶対値はピクセル位置に依存する.
    - 非正方形画像では freq_norm を max(cx, cy) で正規化するため,
      短辺方向の Nyquist 周波数は 0.5 に達しない.
      最終帯域は上限なしで対角線分を含め, 帯域合計 ~1.0 を保証する.

    詳細は docs/fft_features.md を参照.
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "high_low_ratio": "ratio",
        "spectral_std": "cycle/mm_or_cycle/pixel",
        "horizontal_energy": "ratio",
        "vertical_energy": "ratio",
        "num_peaks": "count",
        "max_peak_amp": "amplitude",
        "spectral_centroid": "cycle/mm_or_cycle/pixel",
        "spectral_entropy": "normalized",
        "horizontal_entropy": "normalized",
        "vertical_entropy": "normalized",
        "band_1_0.00_0.10": "ratio",
        "band_2_0.10_0.30": "ratio",
        "band_3_0.30_0.50": "ratio",
        "band_1_0.00_0.10_entropy": "normalized",
        "band_2_0.10_0.30_entropy": "normalized",
        "band_3_0.30_0.50_entropy": "normalized",
    }

    def __init__(
        self,
        name: str = "fft_frequency",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        FFTFrequencyExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "fft_frequency".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得
        self.frequency_bands = self.config["frequency_bands"]
        self.high_low_threshold = self.config["high_low_threshold"]
        self.directional_tolerance = self.config["directional_tolerance"]
        self.peak_threshold_ratio = self.config["peak_threshold_ratio"]
        self.mm_per_pixel = self.config.get("mm_per_pixel")
        if self.mm_per_pixel is not None and self.mm_per_pixel <= 0:
            raise ValueError(
                f"mm_per_pixel must be a positive number, got {self.mm_per_pixel}"
            )

    def _compute_fft_data(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        FFT を1回計算し, 各ヘルパーで共有するデータを返す.

        Args:
            image (np.ndarray): グレースケール画像

        Returns:
            Tuple: (magnitude, power_spectrum, freq_norm, angle_map)
                - magnitude: 振幅スペクトル |F|
                - power_spectrum: パワースペクトル |F|^2
                - freq_norm: 正規化周波数マップ (0〜0.5)
                - angle_map: 角度マップ (0〜180度)
        """
        h, w = image.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        f = np.fft.fft2(image * window)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = h // 2, w // 2

        # DC 成分 (平均輝度) を除外し, AC 成分のみで特徴量を計算する
        magnitude[cy, cx] = 0

        power_spectrum = magnitude**2
        max_dim = max(cx, cy)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        freq_norm = dist / max_dim / 2

        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle_map = (angle + 360) % 180

        return magnitude, power_spectrum, freq_norm, angle_map

    def _compute_band_energy(
        self,
        power_spectrum: np.ndarray,
        freq_norm: np.ndarray,
        bands: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        周波数帯域ごとのエネルギーを計算する.

        Args:
            power_spectrum (np.ndarray): パワースペクトル
            freq_norm (np.ndarray): 正規化周波数マップ
            bands (List[Tuple[float, float]]): 周波数帯域のリスト

        Returns:
            Dict[str, float]: 帯域別エネルギーの辞書
        """
        total_energy = np.sum(power_spectrum)
        energies = {}

        for i, (fmin, fmax) in enumerate(bands):
            # 最終帯域は上限なし (非正方形画像の対角線で freq_norm > 0.5 になる分を含める)
            if i == len(bands) - 1:
                mask = freq_norm >= fmin
            else:
                mask = (freq_norm >= fmin) & (freq_norm < fmax)
            energy = (
                np.sum(power_spectrum[mask]) / total_energy if total_energy > 0 else 0.0
            )
            band_key = f"band_{i+1}_{fmin:.2f}_{fmax:.2f}"
            energies[band_key] = energy

        return energies

    def _compute_spectral_centroid(
        self, magnitude: np.ndarray, freq_norm: np.ndarray
    ) -> float:
        """
        2D画像のスペクトル重心を計算する.

        Args:
            magnitude (np.ndarray): 振幅スペクトル
            freq_norm (np.ndarray): 正規化周波数マップ

        Returns:
            float: スペクトル重心（正規化された空間周波数）
        """
        denominator = np.sum(magnitude)
        if denominator == 0:
            return 0.0
        return float(np.sum(freq_norm * magnitude) / denominator)

    def _compute_high_low_freq_ratio(
        self, power_spectrum: np.ndarray, freq_norm: np.ndarray, threshold: float
    ) -> float:
        """
        高周波/低周波エネルギー比を計算する.

        Args:
            power_spectrum (np.ndarray): パワースペクトル
            freq_norm (np.ndarray): 正規化周波数マップ
            threshold (float): 高周波/低周波の境界閾値

        Returns:
            float: 高周波/低周波エネルギー比. 低周波が 0 の場合は 0.0.
        """
        low_energy = np.sum(power_spectrum[freq_norm < threshold])
        high_energy = np.sum(power_spectrum[freq_norm >= threshold])
        return float(high_energy / low_energy) if low_energy > 0 else 0.0

    def _compute_spectral_std(
        self, magnitude: np.ndarray, freq_norm: np.ndarray
    ) -> float:
        """
        スペクトルの標準偏差を計算する.

        Args:
            magnitude (np.ndarray): 振幅スペクトル
            freq_norm (np.ndarray): 正規化周波数マップ

        Returns:
            float: スペクトルの標準偏差
        """
        total_weight = np.sum(magnitude)
        if total_weight == 0:
            return 0.0

        mean = np.sum(freq_norm * magnitude) / total_weight
        var = np.sum(((freq_norm - mean) ** 2) * magnitude) / total_weight
        return float(np.sqrt(var))

    def _compute_directional_energy(
        self,
        power_spectrum: np.ndarray,
        angle_map: np.ndarray,
        angle_deg: float,
        tolerance_deg: float,
    ) -> float:
        """
        指定方向のエネルギーを計算する.

        Args:
            power_spectrum (np.ndarray): パワースペクトル
            angle_map (np.ndarray): 角度マップ (0〜180度)
            angle_deg (float): 角度（度）
            tolerance_deg (float): 許容角度（度）

        Returns:
            float: 方向性エネルギー比
        """
        diff = np.abs(angle_map - angle_deg)
        diff = np.minimum(
            diff, 180 - diff
        )  # 0度と180度は同一方向のため短い方の角度差を採用
        mask = diff <= tolerance_deg

        total_energy = np.sum(power_spectrum)
        return (
            float(np.sum(power_spectrum[mask]) / total_energy)
            if total_energy > 0
            else 0.0
        )

    def _compute_spectral_peaks(
        self, magnitude: np.ndarray, threshold_ratio: float
    ) -> Tuple[int, float]:
        """
        スペクトルピーク数と最大値を計算する.

        Args:
            magnitude (np.ndarray): 振幅スペクトル
            threshold_ratio (float): ピーク検出の閾値比

        Returns:
            Tuple[int, float]: (ピーク数, 最大ピーク振幅)
        """
        max_local = maximum_filter(magnitude, size=3)
        peaks = (magnitude == max_local) & (
            magnitude > magnitude.max() * threshold_ratio
        )
        num_peaks = int(np.sum(peaks))
        max_peak_amp = float(magnitude[peaks].max()) if num_peaks > 0 else 0.0
        return num_peaks, max_peak_amp

    def _compute_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """
        正規化スペクトラムエントロピーを計算する.

        Shannon エントロピーを log2(N) で正規化し, [0, 1] の範囲に変換する.
        これにより異なるピクセル数の領域間でエントロピーを直接比較できる.

        Args:
            magnitude (np.ndarray): スペクトラムの振幅

        Returns:
            float: 正規化スペクトラルエントロピー (0.0〜1.0)
        """
        total = np.sum(magnitude)
        n = magnitude.size
        if total == 0 or n <= 1:
            return 0.0

        prob = magnitude / total
        nonzero = prob > 0
        entropy = -np.sum(prob[nonzero] * np.log2(prob[nonzero]))
        max_entropy = np.log2(n)
        return float(entropy / max_entropy)

    def _compute_directional_entropy(
        self,
        magnitude: np.ndarray,
        angle_map: np.ndarray,
        angle_deg: float,
        tolerance_deg: float,
    ) -> float:
        """
        指定方向のスペクトラムエントロピーを計算する.

        Args:
            magnitude (np.ndarray): 振幅スペクトル
            angle_map (np.ndarray): 角度マップ (0〜180度)
            angle_deg (float): 角度（度）
            tolerance_deg (float): 許容角度（度）

        Returns:
            float: 方向性スペクトラルエントロピー
        """
        diff = np.abs(angle_map - angle_deg)
        diff = np.minimum(
            diff, 180 - diff
        )  # 0度と180度は同一方向のため短い方の角度差を採用
        mask = diff <= tolerance_deg

        directional_magnitude = magnitude[mask]
        if directional_magnitude.size == 0:
            return 0.0

        return self._compute_spectral_entropy(directional_magnitude)

    def _compute_band_entropy(
        self,
        magnitude: np.ndarray,
        freq_norm: np.ndarray,
        bands: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        周波数帯域ごとのエントロピーを計算する.

        Args:
            magnitude (np.ndarray): 振幅スペクトル
            freq_norm (np.ndarray): 正規化周波数マップ
            bands (List[Tuple[float, float]]): 周波数帯域のリスト

        Returns:
            Dict[str, float]: 帯域別エントロピーの辞書
        """
        entropies = {}

        for i, (fmin, fmax) in enumerate(bands):
            # 最終帯域は上限なし (非正方形画像の対角線で freq_norm > 0.5 になる分を含める)
            if i == len(bands) - 1:
                mask = freq_norm >= fmin
            else:
                mask = (freq_norm >= fmin) & (freq_norm < fmax)
            band_magnitude = magnitude[mask]

            if band_magnitude.size == 0:
                entropy = 0.0
            else:
                entropy = self._compute_spectral_entropy(band_magnitude)

            band_key = f"band_{i+1}_{fmin:.2f}_{fmax:.2f}_entropy"
            entropies[band_key] = entropy

        return entropies

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からFFT周波数領域特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image.copy()
        else:
            raise ValueError(f"Input image must be 2D or 3D, got shape: {image.shape}")

        _MIN_FFT_SIZE = 4
        if gray_image.shape[0] < _MIN_FFT_SIZE or gray_image.shape[1] < _MIN_FFT_SIZE:
            raise ValueError(
                f"Image too small for FFT: {gray_image.shape}. "
                f"Minimum size is {_MIN_FFT_SIZE}x{_MIN_FFT_SIZE}."
            )

        # コントラスト情報を保持するため float64 のまま FFT に渡す
        gray_image = gray_image.astype(np.float64)

        results = {}

        try:
            # 周波数単位の変換係数
            scale = 1.0
            if self.mm_per_pixel is not None:
                scale = 1.0 / self.mm_per_pixel

            # FFT を1回だけ計算し, 全ヘルパーで共有
            magnitude, power_spectrum, freq_norm, angle_map = self._compute_fft_data(
                gray_image
            )

            # 1. 高周波/低周波エネルギー比
            results["high_low_ratio"] = self._compute_high_low_freq_ratio(
                power_spectrum, freq_norm, self.high_low_threshold
            )

            # 2. スペクトルの標準偏差
            results["spectral_std"] = (
                self._compute_spectral_std(magnitude, freq_norm) * scale
            )

            # 3. 方向性エネルギー（水平／垂直）
            results["horizontal_energy"] = self._compute_directional_energy(
                power_spectrum, angle_map, 0.0, self.directional_tolerance
            )
            results["vertical_energy"] = self._compute_directional_energy(
                power_spectrum, angle_map, 90.0, self.directional_tolerance
            )

            # 4. スペクトルピーク数と最大値
            num_peaks, max_peak_amp = self._compute_spectral_peaks(
                magnitude, self.peak_threshold_ratio
            )
            results["num_peaks"] = int(num_peaks)
            results["max_peak_amp"] = float(max_peak_amp)

            # 5. 周波数帯エネルギー
            results.update(
                self._compute_band_energy(
                    power_spectrum, freq_norm, self.frequency_bands
                )
            )

            # 6. スペクトル重心
            results["spectral_centroid"] = (
                self._compute_spectral_centroid(magnitude, freq_norm) * scale
            )

            # 7. スペクトラル全体エントロピー
            results["spectral_entropy"] = self._compute_spectral_entropy(magnitude)

            # 8. 方向性エントロピー（水平／垂直）
            results["horizontal_entropy"] = self._compute_directional_entropy(
                magnitude, angle_map, 0.0, self.directional_tolerance
            )
            results["vertical_entropy"] = self._compute_directional_entropy(
                magnitude, angle_map, 90.0, self.directional_tolerance
            )

            # 9. 周波数帯域エントロピー
            results.update(
                self._compute_band_entropy(magnitude, freq_norm, self.frequency_bands)
            )

            # NaNや無限大の値をチェックして修正
            for key, value in results.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    results[key] = 0.0

        except Exception:
            LogManager().get_logger().exception("FFT feature extraction failed")
            raise

        return results

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        FFTFrequencyExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - frequency_bands: 周波数帯域のリスト
                - high_low_threshold: 高周波/低周波の境界閾値
                - directional_tolerance: 方向性エネルギー計算の許容角度
                - peak_threshold_ratio: ピーク検出の閾値比
                - mm_per_pixel: ピクセルあたりのmm（Noneの場合はピクセル単位）
        """
        return {
            "frequency_bands": [
                [0.0, 0.1],  # 低周波帯域
                [0.1, 0.3],  # 中周波帯域
                [0.3, 0.5],  # 高周波帯域
            ],
            "high_low_threshold": 0.2,  # 高周波/低周波の境界
            "directional_tolerance": 10.0,  # 方向性エネルギーの許容角度（度）
            "peak_threshold_ratio": 0.1,  # ピーク検出の閾値比
            "mm_per_pixel": None,  # ピクセル解像度（Noneの場合はピクセル単位）
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = FFTFrequencyExtractor.get_base_feature_names()
        return [
            f"{name}[{FFTFrequencyExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        return [
            "high_low_ratio",
            "spectral_std",
            "horizontal_energy",
            "vertical_energy",
            "num_peaks",
            "max_peak_amp",
            "spectral_centroid",
            "spectral_entropy",
            "horizontal_entropy",
            "vertical_entropy",
            "band_1_0.00_0.10",
            "band_2_0.10_0.30",
            "band_3_0.30_0.50",
            "band_1_0.00_0.10_entropy",
            "band_2_0.10_0.30_entropy",
            "band_3_0.30_0.50_entropy",
        ]

    @staticmethod
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
        """
        return FFTFrequencyExtractor._FEATURE_UNITS.copy()

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名

        Returns:
            str: 対応する単位
        """
        return FFTFrequencyExtractor._FEATURE_UNITS.get(feature_name, "unknown")
