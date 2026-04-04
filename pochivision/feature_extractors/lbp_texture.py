"""LBP（Local Binary Pattern）テクスチャ特徴量抽出を行うモジュール."""

from typing import Any

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from pochivision.capturelib.log_manager import LogManager
from pochivision.exceptions.extractor import ExtractorValidationError
from pochivision.processors.resize import ResizeProcessor

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("lbp")
class LBPTextureExtractor(BaseFeatureExtractor):
    """
    画像のLBP（Local Binary Pattern）テクスチャ特徴量を抽出するクラス.

    LBPは画像のテクスチャ解析に使用される重要な特徴量で、
    各ピクセルの近傍パターンを二進数で表現することで、
    回転不変性やスケール不変性を持つテクスチャ記述子を生成します.
    局所的なテクスチャパターンの分布を定量化できます.

    抽出する特徴量:
    - ヒストグラム統計量:
       - 平均 [pattern index] 標準偏差 [pattern index]
       - 歪度 [dimensionless] 尖度 [dimensionless]
    - エントロピー: パターンの複雑さを表す [bits]
    - energy: ヒストグラムのエネルギー (sum(p^2), GLCM の ASM と同一計算) [ratio]
    - 各LBPビンの正規化頻度（オプション） [ratio]

    設定により、近傍点数、半径、手法、画像リサイズなどを調整できます.
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "lbp_mean": "pattern_index",
        "lbp_std": "pattern_index",
        "lbp_skewness": "dimensionless",
        "lbp_kurtosis": "dimensionless",
        "lbp_entropy": "normalized",
        "lbp_energy": "ratio",
    }

    def __init__(
        self,
        name: str = "lbp_texture",
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        LBPTextureExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "lbp_texture".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.P = self.config["P"]  # 近傍点数
        self.R = self.config["R"]  # 半径
        self.method = self.config["method"]  # LBP手法
        # リサイズ形状（Noneの場合はリサイズしない）
        resize_shape_config = self.config["resize_shape"]
        self.resize_shape = (
            tuple(resize_shape_config) if resize_shape_config is not None else None
        )
        self.include_histogram = self.config[
            "include_histogram"
        ]  # ヒストグラム含有フラグ

        # ResizeProcessorの準備（リサイズが必要な場合のみ）
        self.resize_processor = None
        if self.resize_shape is not None:
            resize_config = ResizeProcessor.get_default_config()
            resize_config["width"] = self.resize_shape[1]
            resize_config["height"] = self.resize_shape[0]
            resize_config["preserve_aspect_ratio"] = self.config[
                "preserve_aspect_ratio"
            ]
            resize_config["aspect_ratio_mode"] = self.config["aspect_ratio_mode"]
            self.resize_processor = ResizeProcessor(
                name="resize_for_lbp", config=resize_config
            )

    def extract(self, image: np.ndarray) -> dict[str, float | int]:
        """
        画像からLBPテクスチャ特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            dict[str, float | int]: 抽出された特徴量の辞書.
                - lbp_mean: LBPヒストグラムの平均
                - lbp_std: LBPヒストグラムの標準偏差
                - lbp_skewness: LBPヒストグラムの歪度
                - lbp_kurtosis: LBPヒストグラムの尖度
                - lbp_entropy: LBPヒストグラムのエントロピー
                - lbp_energy: LBPヒストグラムのエネルギー (sum(p^2))
                - lbp_bin_{i}: 各LBPビンの正規化頻度（include_histogramがTrueの場合）

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ExtractorValidationError("Input image is empty or None")

        try:
            # 画像の型を適切に変換
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                # サポートされていない型の場合、uint8に変換
                if image.dtype in [np.int32, np.int64]:
                    # 整数型の場合、0-255の範囲にクリップしてuint8に変換
                    image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    # その他の型の場合、float32経由でuint8に変換
                    image = image.astype(np.float32)
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)

            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray_image = image.copy()
            else:
                raise ExtractorValidationError(
                    f"Input image must be 2D or 3D, got shape: {image.shape}"
                )

            # 画像を指定サイズにリサイズ（スケール統一のため）
            if self.resize_processor is not None:
                # リサイズプロセッサーを使用してリサイズ
                gray_image = self.resize_processor.process(gray_image)

            lbp = local_binary_pattern(gray_image, self.P, self.R, method=self.method)

            # メソッド別のビン数と範囲を設定
            if self.method == "uniform":
                n_bins = self.P + 2
                hist_range = (0.0, float(n_bins))
            elif self.method == "nri_uniform":
                n_bins = self.P * (self.P - 1) + 3
                hist_range = (0.0, float(n_bins))
            elif self.method == "var":
                # var メソッドは連続値を返すため, 実際の値域を使用
                n_bins = 256
                hist_range = (float(lbp.min()), float(lbp.max()) + 1e-10)
            else:  # default, ror
                n_bins = 2**self.P
                hist_range = (0.0, float(n_bins))

            # density=False + 手動正規化で確率分布 (和=1) を計算
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_bins, range=hist_range, density=False
            )
            hist = hist.astype(np.float64)
            total = hist.sum()
            if total > 0:
                hist = hist / total

            results = self._calculate_statistics(hist, lbp)

            # ヒストグラムの各ビンを特徴量として追加（オプション）
            if self.include_histogram:
                for i, bin_value in enumerate(hist):
                    results[f"lbp_bin_{i}"] = float(bin_value)

            return results

        except Exception:
            LogManager().get_logger().exception("LBP feature extraction failed")
            raise

    def _calculate_statistics(
        self, hist: np.ndarray, lbp: np.ndarray
    ) -> dict[str, float]:
        """
        LBP 画像とヒストグラムから統計量を計算する.

        mean/std/skewness/kurtosis は LBP 画像の値から直接計算する.
        entropy/energy はヒストグラムの分布から計算する.

        Args:
            hist (np.ndarray): 正規化されたLBPヒストグラム
            lbp (np.ndarray): LBP画像

        Returns:
            dict[str, float]: 統計量の辞書
        """
        results = {}

        try:
            # LBP 画像から直接計算 (パターン番号ではなくテクスチャ特性を反映)
            lbp_flat = lbp.ravel().astype(np.float64)
            mean = float(np.mean(lbp_flat))
            std = float(np.std(lbp_flat))
            results["lbp_mean"] = mean
            results["lbp_std"] = std

            # 歪度
            if std > 0:
                skewness = float(np.mean(((lbp_flat - mean) / std) ** 3))
            else:
                skewness = 0.0
            results["lbp_skewness"] = skewness

            # 尖度 (excess kurtosis)
            if std > 0:
                kurtosis = float(np.mean(((lbp_flat - mean) / std) ** 4) - 3.0)
            else:
                kurtosis = 0.0
            results["lbp_kurtosis"] = kurtosis

            # 正規化エントロピー [0, 1] (FFT/SWT と統一)
            hist_nonzero = hist[hist > 0]
            if len(hist_nonzero) > 1:
                raw_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                max_entropy = np.log2(len(hist))
                entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                entropy = 0.0
            results["lbp_entropy"] = float(entropy)

            # エネルギー (Angular Second Moment): sum(p^2)
            energy = np.sum(hist**2)
            results["lbp_energy"] = float(energy)

        except Exception:
            LogManager().get_logger().exception("LBP statistics calculation failed")
            raise

        return results

    def _get_default_results(self) -> dict[str, float]:
        """
        エラー時のデフォルト結果を返す.

        Returns:
            dict[str, float]: デフォルト値の辞書
        """
        results = {
            "lbp_mean": 0.0,
            "lbp_std": 0.0,
            "lbp_skewness": 0.0,
            "lbp_kurtosis": 0.0,
            "lbp_entropy": 0.0,
            "lbp_energy": 0.0,
        }

        # ヒストグラムを含む場合のデフォルト値
        if self.include_histogram:
            # uniform LBPの場合の最大ビン数を推定
            if self.method == "uniform":
                max_bins = self.P + 2  # uniform LBPの場合
            else:
                max_bins = 2**self.P  # 通常のLBPの場合

            for i in range(max_bins):
                results[f"lbp_bin_{i}"] = 0.0

        return results

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """
        LBPTextureExtractorのデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
                - P: 近傍点数
                - R: 半径
                - method: LBP手法
                - resize_shape: リサイズ形状（None の場合はリサイズしない）
                - include_histogram: ヒストグラムの各ビンを特徴量として含むかどうか
        """
        return {
            "P": 8,  # 8近傍点
            "R": 1,  # 半径1
            "method": "uniform",  # 回転不変uniform LBP
            "resize_shape": [128, 128],  # 128x128にリサイズ
            "include_histogram": False,  # ヒストグラムは含めない（統計量のみ）
            "preserve_aspect_ratio": True,
            "aspect_ratio_mode": "width",
        }

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            list[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = LBPTextureExtractor.get_base_feature_names()
        return [
            f"{name}[{LBPTextureExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            list[str]: 基本特徴量名のリスト.
        """
        # 基本統計量
        feature_names = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_energy",
        ]

        # デフォルト設定でヒストグラムを含む場合
        default_config = LBPTextureExtractor.get_default_config()
        if default_config["include_histogram"]:
            P = default_config["P"]
            method = default_config["method"]

            if method == "uniform":
                max_bins = P + 2
            elif method == "nri_uniform":
                max_bins = P * (P - 1) + 3
            elif method == "var":
                max_bins = 256
            else:
                max_bins = 2**P

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}")

        return feature_names

    @staticmethod
    def get_feature_units() -> dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            dict[str, str]: 特徴量名と単位の対応辞書.
        """
        # 基本特徴量名を取得
        base_names = LBPTextureExtractor.get_base_feature_names()

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = LBPTextureExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "lbp_mean"）

        Returns:
            str: 対応する単位
        """
        # ヒストグラムビンの場合
        if feature_name.startswith("lbp_bin_"):
            return "ratio"

        # 基本統計量の場合
        return LBPTextureExtractor._FEATURE_UNITS.get(feature_name, "unknown")

    # インスタンスメソッド版（後方互換性のため残す）
    def get_feature_names_instance(self) -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き、インスタンス設定反映）.

        Returns:
            list[str]: 特徴量名のリスト（単位付き）.
        """
        # 基本統計量（単位付き）
        feature_names = []
        base_features = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_energy",
        ]

        for feature in base_features:
            unit = self._FEATURE_UNITS.get(feature, "unknown")
            feature_names.append(f"{feature}[{unit}]")

        # インスタンスの設定でヒストグラムを含む場合
        if self.include_histogram:
            # uniform LBPの場合のビン数を計算
            P = self.P
            method = self.method

            if method == "uniform":
                max_bins = P + 2
            elif method == "nri_uniform":
                max_bins = P * (P - 1) + 3
            elif method == "var":
                max_bins = 256
            else:
                max_bins = 2**P

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}[ratio]")

        return feature_names

    def get_base_feature_names_instance(self) -> list[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし、インスタンス設定反映）.

        Returns:
            list[str]: 基本特徴量名のリスト.
        """
        # 基本統計量
        feature_names = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_energy",
        ]

        # インスタンスの設定でヒストグラムを含む場合
        if self.include_histogram:
            P = self.P
            method = self.method

            if method == "uniform":
                max_bins = P + 2
            elif method == "nri_uniform":
                max_bins = P * (P - 1) + 3
            elif method == "var":
                max_bins = 256
            else:
                max_bins = 2**P

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}")

        return feature_names

    def get_feature_units_instance(self) -> dict[str, str]:
        """
        この特徴量抽出器が出力する特徴量の単位を返す（インスタンス設定反映）.

        Returns:
            dict[str, str]: 特徴量名と単位の辞書.
        """
        units = self._FEATURE_UNITS.copy()

        # インスタンスの設定でヒストグラムを含む場合
        if self.include_histogram:
            P = self.P
            method = self.method

            if method == "uniform":
                max_bins = P + 2
            elif method == "nri_uniform":
                max_bins = P * (P - 1) + 3
            elif method == "var":
                max_bins = 256
            else:
                max_bins = 2**P

            for i in range(max_bins):
                units[f"lbp_bin_{i}"] = "ratio"

        return units

    def get_feature_unit_instance(self, feature_name: str) -> str:
        """
        指定された特徴量の単位を返す（インスタンス設定反映）.

        Args:
            feature_name (str): 特徴量名.

        Returns:
            str: 特徴量の単位.
        """
        units = self.get_feature_units_instance()
        return units.get(feature_name, "unknown")
