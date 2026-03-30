"""RGB統計特徴量抽出を行うモジュール."""

from typing import Any, Dict, Optional, Union

import numpy as np

from pochivision.capturelib.log_manager import LogManager
from pochivision.utils.image import to_rgb

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("rgb")
class RGBStatisticsExtractor(BaseFeatureExtractor):
    """
    画像のRGB統計特徴量を抽出するクラス.

    RGB色空間における各チャンネル（赤、緑、青）の統計的特性を定量化します。
    色の分布や変動性を把握することで、画像の色彩特性を分析できます。

    抽出する特徴量（R、G、Bチャンネルそれぞれ）:
    - mean: RGB平均値 [0-255]
    - median: RGB中央値 [0-255]
    - variance: RGB分散 [0-255_squared]
    - std_dev: RGB標準偏差 [0-255]
    - cv: 変動係数（標準偏差/平均値） [ratio]

    exclude_black_pixels の動作:
    - True: R=0, G=0, B=0 のピクセル (完全な黒) のみを統計から除外する.
      R=200, G=0, B=0 のようなピクセルは除外されない (いずれかのチャンネルが非ゼロ).
    - False: 全ピクセルを統計に含む.
    - 用途: 背景が真っ黒の画像で, 背景領域を統計から除外したい場合に使用.
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "mean": "0-255",
        "median": "0-255",
        "variance": "0-255_squared",
        "std_dev": "0-255",
        "cv": "ratio",
    }

    def __init__(
        self,
        name: str = "rgb_statistics",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        RGBStatisticsExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "rgb_statistics".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.exclude_black_pixels = self.config["exclude_black_pixels"]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からRGB統計特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                R、G、Bチャンネルそれぞれについて:
                - {channel}_mean: 平均値
                - {channel}_median: 中央値
                - {channel}_variance: 分散
                - {channel}_std_dev: 標準偏差
                - {channel}_cv: 変動係数

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        try:
            # float (0-1) 入力を uint8 スケールに変換
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            rgb_image = to_rgb(image)

            results = {}

            if self.exclude_black_pixels:
                # R=0, G=0, B=0 の完全黒ピクセルを除外 (いずれかが非ゼロなら残す)
                non_black_mask = np.any(rgb_image > 0, axis=2)
            else:
                non_black_mask = np.ones(rgb_image.shape[:2], dtype=bool)

            channel_names = ["red", "green", "blue"]

            for i, channel_name in enumerate(channel_names):
                channel_data = rgb_image[:, :, i]

                if self.exclude_black_pixels:
                    pixels = channel_data[non_black_mask].astype(np.float64)
                else:
                    pixels = channel_data.flatten().astype(np.float64)

                if len(pixels) == 0:
                    mean_val = 0.0
                    median_val = 0.0
                    variance_val = 0.0
                    std_dev_val = 0.0
                    cv_val = float("inf")
                else:
                    mean_val = float(np.mean(pixels))
                    median_val = float(np.median(pixels))
                    variance_val = float(np.var(pixels))
                    std_dev_val = float(np.std(pixels))
                    cv_val = (
                        float(std_dev_val / mean_val) if mean_val != 0 else float("inf")
                    )

                results[f"{channel_name}_mean"] = mean_val
                results[f"{channel_name}_median"] = median_val
                results[f"{channel_name}_variance"] = variance_val
                results[f"{channel_name}_std_dev"] = std_dev_val
                results[f"{channel_name}_cv"] = cv_val

            return results
        except Exception:
            LogManager().get_logger().exception("RGB feature extraction failed")
            raise

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        RGBStatisticsExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - exclude_black_pixels: RGB値がすべて0のピクセルを除外するかどうか
        """
        return {"exclude_black_pixels": True}

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            list[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = RGBStatisticsExtractor.get_base_feature_names()
        return [
            f"{name}[{RGBStatisticsExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            list[str]: 基本特徴量名のリスト.
        """
        feature_names = []
        channels = ["red", "green", "blue"]
        stats = ["mean", "median", "variance", "std_dev", "cv"]

        for channel in channels:
            for stat in stats:
                feature_names.append(f"{channel}_{stat}")

        return feature_names

    @staticmethod
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
        """
        # 基本特徴量名を取得
        base_names = RGBStatisticsExtractor.get_base_feature_names()

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = RGBStatisticsExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "red_mean"）

        Returns:
            str: 対応する単位
        """
        # 特徴量名から統計量部分を抽出
        parts = feature_name.split("_")
        if len(parts) >= 2:
            # "red_std_dev" -> "std_dev", "red_mean" -> "mean"
            stat = "_".join(parts[1:])  # チャンネル名以外の部分を結合
            return RGBStatisticsExtractor._FEATURE_UNITS.get(stat, "unknown")
        return "unknown"
