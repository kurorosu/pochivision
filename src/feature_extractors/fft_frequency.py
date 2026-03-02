"""FFT（高速フーリエ変換）周波数領域特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("fft")
class FFTFrequencyExtractor(BaseFeatureExtractor):
    """
    画像のFFT（高速フーリエ変換）周波数領域特徴量を抽出するクラス.

    FFTは画像の周波数成分を解析し、テクスチャや周期性の特徴を抽出します。
    空間領域の画像を周波数領域に変換することで、画像の周波数特性を定量化できます。

    抽出する特徴量:
    - high_low_ratio: 高周波/低周波エネルギー比 [ratio]
    - spectral_std: スペクトルの標準偏差 [cycle/mm or cycle/pixel]
    - horizontal_energy: 水平方向エネルギー [ratio]
    - vertical_energy: 垂直方向エネルギー [ratio]
    - num_peaks: スペクトルピーク数 [count]
    - max_peak_amp: 最大ピーク振幅 [amplitude]
    - band_energy: 周波数帯域別エネルギー [ratio]
    - spectral_centroid: スペクトル重心 [cycle/mm or cycle/pixel]
    - spectral_entropy: スペクトラム全体のエントロピー [bits]
    - horizontal_entropy: 水平方向スペクトラムエントロピー [bits]
    - vertical_entropy: 垂直方向スペクトラムエントロピー [bits]
    - band_entropy: 周波数帯域別エントロピー [bits]

    設定により、周波数帯域、閾値、ピクセル解像度などを調整できます。
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
        "spectral_entropy": "bits",
        "horizontal_entropy": "bits",
        "vertical_entropy": "bits",
        "band_1_0.00_0.10": "ratio",
        "band_2_0.10_0.30": "ratio",
        "band_3_0.30_0.50": "ratio",
        "band_1_0.00_0.10_entropy": "bits",
        "band_2_0.10_0.30_entropy": "bits",
        "band_3_0.30_0.50_entropy": "bits",
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

    def _compute_band_energy(
        self, image: np.ndarray, bands: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        周波数帯域ごとのエネルギーを計算する.

        Args:
            image (np.ndarray): グレースケール画像
            bands (List[Tuple[float, float]]): 周波数帯域のリスト

        Returns:
            Dict[str, float]: 帯域別エネルギーの辞書
        """
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) ** 2

        h, w = image.shape
        cy, cx = h // 2, w // 2
        max_radius = np.sqrt(cx**2 + cy**2)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        freq_norm = dist / max_radius / 2

        total_energy = np.sum(magnitude)
        energies = {}

        for i, (fmin, fmax) in enumerate(bands):
            mask = (freq_norm >= fmin) & (freq_norm < fmax)
            energy = np.sum(magnitude[mask]) / total_energy if total_energy > 0 else 0.0
            band_key = f"band_{i+1}_{fmin:.2f}_{fmax:.2f}"
            energies[band_key] = energy

        return energies

    def _compute_spectral_centroid(self, image: np.ndarray) -> float:
        """
        2D画像のスペクトル重心を計算する.

        Args:
            image (np.ndarray): グレースケール画像

        Returns:
            float: スペクトル重心（正規化された空間周波数）
        """
        h, w = image.shape
        cy, cx = h // 2, w // 2

        # 2D FFTとスペクトル
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # 距離ベースの周波数マップ（0〜0.5）
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = np.sqrt(cx**2 + cy**2)
        norm_freq = dist / max_radius / 2  # 0〜0.5に正規化

        # スペクトル重心の計算
        numerator = np.sum(norm_freq * magnitude)
        denominator = np.sum(magnitude)

        return numerator / denominator if denominator != 0 else 0.0

    def _compute_high_low_freq_ratio(
        self, image: np.ndarray, threshold: float
    ) -> float:
        """
        高周波/低周波エネルギー比を計算する.

        Args:
            image (np.ndarray): グレースケール画像
            threshold (float): 高周波/低周波の境界閾値

        Returns:
            float: 高周波/低周波エネルギー比
        """
        h, w = image.shape
        cy, cx = h // 2, w // 2
        max_radius = np.sqrt(cx**2 + cy**2)

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) ** 2  # パワースペクトル

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        norm_freq = dist / max_radius / 2  # 0〜0.5

        low_mask = norm_freq < threshold
        high_mask = norm_freq >= threshold

        low_energy = np.sum(magnitude[low_mask])
        high_energy = np.sum(magnitude[high_mask])

        return float(high_energy / low_energy) if low_energy > 0 else np.inf

    def _compute_spectral_std(self, image: np.ndarray) -> float:
        """
        スペクトルの標準偏差を計算する.

        Args:
            image (np.ndarray): グレースケール画像

        Returns:
            float: スペクトルの標準偏差
        """
        h, w = image.shape
        cy, cx = h // 2, w // 2
        max_radius = np.sqrt(cx**2 + cy**2)

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        norm_freq = dist / max_radius / 2

        weights = magnitude
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0

        mean = np.sum(norm_freq * weights) / total_weight
        var = np.sum(((norm_freq - mean) ** 2) * weights) / total_weight
        return float(np.sqrt(var))

    def _compute_directional_energy(
        self, image: np.ndarray, angle_deg: float, tolerance_deg: float
    ) -> float:
        """
        指定方向のエネルギーを計算する.

        Args:
            image (np.ndarray): グレースケール画像
            angle_deg (float): 角度（度）
            tolerance_deg (float): 許容角度（度）

        Returns:
            float: 方向性エネルギー比
        """
        h, w = image.shape
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) ** 2

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx) * 180 / np.pi  # -180〜180度

        angle = (angle + 360) % 180  # 0〜180度に正規化
        diff = np.abs(angle - angle_deg)
        mask = diff <= tolerance_deg

        directional_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)
        return directional_energy / total_energy if total_energy > 0 else 0.0

    def _compute_spectral_peaks(
        self, image: np.ndarray, threshold_ratio: float
    ) -> Tuple[int, float]:
        """
        スペクトルピーク数と最大値を計算する.

        Args:
            image (np.ndarray): グレースケール画像
            threshold_ratio (float): ピーク検出の閾値比

        Returns:
            Tuple[int, float]: (ピーク数, 最大ピーク振幅)
        """
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # 局所ピークを検出
        max_local = maximum_filter(magnitude, size=3)
        peaks = (magnitude == max_local) & (
            magnitude > magnitude.max() * threshold_ratio
        )

        num_peaks = int(np.sum(peaks))
        max_peak = float(magnitude.max())

        return num_peaks, max_peak

    def _compute_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """
        スペクトラムのエントロピーを計算する.

        Args:
            magnitude (np.ndarray): スペクトラムの振幅

        Returns:
            float: スペクトラルエントロピー
        """
        # 正規化してプロバビリティ分布に変換
        total = np.sum(magnitude)
        if total == 0:
            return 0.0

        prob = magnitude / total
        # エントロピー計算（0 log 0 = 0として処理）
        prob_safe = prob + 1e-12  # 数値安定性のための小さな値
        entropy = -np.sum(prob * np.log2(prob_safe))
        return float(entropy)

    def _compute_directional_entropy(
        self, image: np.ndarray, angle_deg: float, tolerance_deg: float
    ) -> float:
        """
        指定方向のスペクトラムエントロピーを計算する.

        Args:
            image (np.ndarray): グレースケール画像
            angle_deg (float): 角度（度）
            tolerance_deg (float): 許容角度（度）

        Returns:
            float: 方向性スペクトラルエントロピー
        """
        h, w = image.shape
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dy = y - cy
        dx = x - cx
        angle = np.arctan2(dy, dx) * 180 / np.pi  # -180〜180度

        angle = (angle + 360) % 180  # 0〜180度に正規化
        diff = np.abs(angle - angle_deg)
        mask = diff <= tolerance_deg

        # 指定方向のスペクトラムを抽出
        directional_magnitude = magnitude[mask]
        if directional_magnitude.size == 0:
            return 0.0

        return self._compute_spectral_entropy(directional_magnitude)

    def _compute_band_entropy(
        self, image: np.ndarray, bands: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        周波数帯域ごとのエントロピーを計算する.

        Args:
            image (np.ndarray): グレースケール画像
            bands (List[Tuple[float, float]]): 周波数帯域のリスト

        Returns:
            Dict[str, float]: 帯域別エントロピーの辞書
        """
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = image.shape
        cy, cx = h // 2, w // 2
        max_radius = np.sqrt(cx**2 + cy**2)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        freq_norm = dist / max_radius / 2

        entropies = {}

        for i, (fmin, fmax) in enumerate(bands):
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

        # 画像の型を適切に変換
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            if image.dtype in [np.int32, np.int64]:
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.float32)
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

        # グレースケール変換
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image.copy()
        else:
            raise ValueError(f"Input image must be 2D or 3D, got shape: {image.shape}")

        # 画像を0-255の範囲に正規化
        gray_image = cv2.normalize(
            gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        results = {}

        try:
            # 周波数単位の変換係数
            scale = 1.0
            if self.mm_per_pixel is not None:
                scale = 1.0 / self.mm_per_pixel

            # 1. 高周波/低周波エネルギー比
            high_low_ratio = self._compute_high_low_freq_ratio(
                gray_image, self.high_low_threshold
            )
            # 無限大の値をチェック
            if np.isinf(high_low_ratio):
                high_low_ratio = 0.0
            results["high_low_ratio"] = float(high_low_ratio)

            # 2. スペクトルの標準偏差
            spectral_std = self._compute_spectral_std(gray_image) * scale
            results["spectral_std"] = float(spectral_std)

            # 3. 方向性エネルギー（水平／垂直）
            horizontal_energy = self._compute_directional_energy(
                gray_image, 0.0, self.directional_tolerance
            )
            results["horizontal_energy"] = float(horizontal_energy)

            vertical_energy = self._compute_directional_energy(
                gray_image, 90.0, self.directional_tolerance
            )
            results["vertical_energy"] = float(vertical_energy)

            # 4. スペクトルピーク数と最大値
            num_peaks, max_peak_amp = self._compute_spectral_peaks(
                gray_image, self.peak_threshold_ratio
            )
            results["num_peaks"] = int(num_peaks)
            results["max_peak_amp"] = float(max_peak_amp)

            # 5. 周波数帯エネルギー
            band_energies = self._compute_band_energy(gray_image, self.frequency_bands)
            results.update(band_energies)

            # 6. スペクトル重心
            spectral_centroid = self._compute_spectral_centroid(gray_image) * scale
            results["spectral_centroid"] = float(spectral_centroid)

            # 7. スペクトラル全体エントロピー
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            spectral_entropy = self._compute_spectral_entropy(magnitude)
            results["spectral_entropy"] = float(spectral_entropy)

            # 8. 方向性エントロピー（水平／垂直）
            horizontal_entropy = self._compute_directional_entropy(
                gray_image, 0.0, self.directional_tolerance
            )
            results["horizontal_entropy"] = float(horizontal_entropy)

            vertical_entropy = self._compute_directional_entropy(
                gray_image, 90.0, self.directional_tolerance
            )
            results["vertical_entropy"] = float(vertical_entropy)

            # 9. 周波数帯域エントロピー
            band_entropies = self._compute_band_entropy(
                gray_image, self.frequency_bands
            )
            results.update(band_entropies)

            # NaNや無限大の値をチェックして修正
            for key, value in results.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    results[key] = 0.0

        except Exception:
            # エラーが発生した場合、すべて0で埋める
            feature_names = self.get_feature_names()
            results = {name: 0.0 for name in feature_names}
            return results

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
