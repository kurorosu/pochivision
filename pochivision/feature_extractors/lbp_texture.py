"""LBP（Local Binary Pattern）テクスチャ特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

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
    - 均一性: パターンの均一性を表す [ratio]
    - 各LBPビンの正規化頻度（オプション） [ratio]

    設定により、近傍点数、半径、手法、画像リサイズなどを調整できます.
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "lbp_mean": "pattern_index",
        "lbp_std": "pattern_index",
        "lbp_skewness": "dimensionless",
        "lbp_kurtosis": "dimensionless",
        "lbp_entropy": "bits",
        "lbp_uniformity": "ratio",
    }

    def __init__(
        self,
        name: str = "lbp_texture",
        config: Optional[Dict[str, Any]] = None,
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
            # 特徴量抽出では正確なサイズ合わせが必要なため、アスペクト比保持を無効化
            resize_config["preserve_aspect_ratio"] = False

            self.resize_processor = ResizeProcessor(
                name="resize_for_lbp", config=resize_config
            )

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からLBPテクスチャ特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                - lbp_mean: LBPヒストグラムの平均
                - lbp_std: LBPヒストグラムの標準偏差
                - lbp_skewness: LBPヒストグラムの歪度
                - lbp_kurtosis: LBPヒストグラムの尖度
                - lbp_entropy: LBPヒストグラムのエントロピー
                - lbp_uniformity: LBPヒストグラムの均一性
                - lbp_bin_{i}: 各LBPビンの正規化頻度（include_histogramがTrueの場合）

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

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

            # グレースケール変換
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray_image = image.copy()
            else:
                raise ValueError(
                    f"Input image must be 2D or 3D, got shape: {image.shape}"
                )

            # 画像を指定サイズにリサイズ（スケール統一のため）
            if self.resize_processor is not None:
                # リサイズプロセッサーを使用してリサイズ
                gray_image = self.resize_processor.process(gray_image)

            # LBP計算
            lbp = local_binary_pattern(gray_image, self.P, self.R, method=self.method)

            # ヒストグラム計算
            n_bins = int(lbp.max() + 1)  # 実際のラベル数で固定次元化
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
            )

            # 統計量計算
            results = self._calculate_statistics(hist, lbp)

            # ヒストグラムの各ビンを特徴量として追加（オプション）
            if self.include_histogram:
                for i, bin_value in enumerate(hist):
                    results[f"lbp_bin_{i}"] = float(bin_value)

            return results

        except Exception:
            # エラーが発生した場合、デフォルト値で埋める
            return self._get_default_results()

    def _calculate_statistics(
        self, hist: np.ndarray, lbp: np.ndarray
    ) -> Dict[str, float]:
        """
        LBPヒストグラムから統計量を計算する.

        Args:
            hist (np.ndarray): 正規化されたLBPヒストグラム
            lbp (np.ndarray): LBP画像

        Returns:
            Dict[str, float]: 統計量の辞書
        """
        results = {}

        try:
            # 基本統計量
            # ヒストグラムの重心（平均）
            bin_centers = np.arange(len(hist))
            mean = np.sum(bin_centers * hist) if np.sum(hist) > 0 else 0.0
            results["lbp_mean"] = float(mean)

            # 標準偏差
            variance = (
                np.sum(((bin_centers - mean) ** 2) * hist) if np.sum(hist) > 0 else 0.0
            )
            std = np.sqrt(variance)
            results["lbp_std"] = float(std)

            # 歪度（skewness）
            if std > 0:
                skewness = np.sum(((bin_centers - mean) ** 3) * hist) / (std**3)
            else:
                skewness = 0.0
            results["lbp_skewness"] = float(skewness)

            # 尖度（kurtosis）
            if std > 0:
                kurtosis = np.sum(((bin_centers - mean) ** 4) * hist) / (std**4) - 3.0
            else:
                kurtosis = 0.0
            results["lbp_kurtosis"] = float(kurtosis)

            # エントロピー
            # 0の値を除外してエントロピーを計算
            hist_nonzero = hist[hist > 0]
            if len(hist_nonzero) > 0:
                entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            else:
                entropy = 0.0
            results["lbp_entropy"] = float(entropy)

            # 均一性（Uniformity）- エネルギーとも呼ばれる
            uniformity = np.sum(hist**2)
            results["lbp_uniformity"] = float(uniformity)

        except Exception:
            # 統計量計算でエラーが発生した場合、0で埋める
            results.update(
                {
                    "lbp_mean": 0.0,
                    "lbp_std": 0.0,
                    "lbp_skewness": 0.0,
                    "lbp_kurtosis": 0.0,
                    "lbp_entropy": 0.0,
                    "lbp_uniformity": 0.0,
                }
            )

        return results

    def _get_default_results(self) -> Dict[str, float]:
        """
        エラー時のデフォルト結果を返す.

        Returns:
            Dict[str, float]: デフォルト値の辞書
        """
        results = {
            "lbp_mean": 0.0,
            "lbp_std": 0.0,
            "lbp_skewness": 0.0,
            "lbp_kurtosis": 0.0,
            "lbp_entropy": 0.0,
            "lbp_uniformity": 0.0,
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
    def get_default_config() -> Dict[str, Any]:
        """
        LBPTextureExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
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
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = LBPTextureExtractor.get_base_feature_names()
        return [
            f"{name}[{LBPTextureExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        # 基本統計量
        feature_names = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_uniformity",
        ]

        # デフォルト設定でヒストグラムを含む場合
        default_config = LBPTextureExtractor.get_default_config()
        if default_config["include_histogram"]:
            # uniform LBPの場合のビン数を計算
            P = default_config["P"]
            method = default_config["method"]

            if method == "uniform":
                max_bins = P + 2  # uniform LBPの場合
            else:
                max_bins = 2**P  # 通常のLBPの場合

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}")

        return feature_names

    @staticmethod
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
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
    def get_feature_names_instance(self) -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き、インスタンス設定反映）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        # 基本統計量（単位付き）
        feature_names = []
        base_features = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_uniformity",
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
                max_bins = P + 2  # uniform LBPの場合
            else:
                max_bins = 2**P  # 通常のLBPの場合

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}[ratio]")

        return feature_names

    def get_base_feature_names_instance(self) -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし、インスタンス設定反映）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        # 基本統計量
        feature_names = [
            "lbp_mean",
            "lbp_std",
            "lbp_skewness",
            "lbp_kurtosis",
            "lbp_entropy",
            "lbp_uniformity",
        ]

        # インスタンスの設定でヒストグラムを含む場合
        if self.include_histogram:
            # uniform LBPの場合のビン数を計算
            P = self.P
            method = self.method

            if method == "uniform":
                max_bins = P + 2  # uniform LBPの場合
            else:
                max_bins = 2**P  # 通常のLBPの場合

            for i in range(max_bins):
                feature_names.append(f"lbp_bin_{i}")

        return feature_names

    def get_feature_units_instance(self) -> Dict[str, str]:
        """
        この特徴量抽出器が出力する特徴量の単位を返す（インスタンス設定反映）.

        Returns:
            Dict[str, str]: 特徴量名と単位の辞書.
        """
        units = self._FEATURE_UNITS.copy()

        # インスタンスの設定でヒストグラムを含む場合
        if self.include_histogram:
            # uniform LBPの場合のビン数を計算
            P = self.P
            method = self.method

            if method == "uniform":
                max_bins = P + 2  # uniform LBPの場合
            else:
                max_bins = 2**P  # 通常のLBPの場合

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
