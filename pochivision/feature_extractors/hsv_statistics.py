"""HSV統計特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from scipy.stats import circmean, circstd

from pochivision.capturelib.log_manager import LogManager
from pochivision.utils.image import to_bgr

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("hsv")
class HSVStatisticsExtractor(BaseFeatureExtractor):
    """
    画像のHSV統計特徴量を抽出するクラス.

    HSV色空間における各チャンネル（色相、彩度、明度）の統計的特性を定量化します。
    色の性質をより直感的に把握することで、画像の色彩特性を分析できます。

    抽出する特徴量（H、S、Vチャンネルそれぞれ）:
    - mean: HSV平均値 [H: degree, S: intensity, V: intensity]
    - median: HSV中央値 [H: degree, S: intensity, V: intensity]
    - variance: HSV分散 [H: squared_degree, S: squared_intensity, V: squared_intensity]
    - std_dev: HSV標準偏差 [H: degree, S: intensity, V: intensity]
    - cv: 変動係数（標準偏差/平均値） [ratio]

    exclude_black_pixels の動作:
    - True: 元の BGR 画像で B=0, G=0, R=0 のピクセル (完全な黒) のみを統計から除外する.
      HSV 変換前の BGR 値で判定するため, 色味のあるピクセルは除外されない.
    - False: 全ピクセルを統計に含む.
    - 用途: 背景が真っ黒の画像で, 背景領域を統計から除外したい場合に使用.
    """

    # OpenCV の HSV Hue 範囲 (0-179)
    _HUE_HIGH = 180

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "hue_mean": "hue_0_179",
        "hue_median": "hue_0_179",
        "hue_variance": "hue_0_179_squared",
        "hue_std_dev": "hue_0_179",
        "hue_cv": "ratio",
        "saturation_mean": "intensity",
        "saturation_median": "intensity",
        "saturation_variance": "squared_intensity",
        "saturation_std_dev": "intensity",
        "saturation_cv": "ratio",
        "value_mean": "intensity",
        "value_median": "intensity",
        "value_variance": "squared_intensity",
        "value_std_dev": "intensity",
        "value_cv": "ratio",
    }

    def __init__(
        self,
        name: str = "hsv_statistics",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        HSVStatisticsExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "hsv_statistics".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.exclude_black_pixels = self.config["exclude_black_pixels"]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からHSV統計特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                H、S、Vチャンネルそれぞれについて:
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

            bgr_image = to_bgr(image)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

            results = {}

            if self.exclude_black_pixels:
                # BGR 全チャンネル 0 の完全黒ピクセルを除外 (いずれかが非ゼロなら残す)
                non_black_mask = np.any(bgr_image > 0, axis=2)
            else:
                non_black_mask = np.ones(bgr_image.shape[:2], dtype=bool)

            channel_names = ["hue", "saturation", "value"]

            for i, channel_name in enumerate(channel_names):
                channel_data = hsv_image[:, :, i]

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
                elif channel_name == "hue":
                    mean_val, median_val, variance_val, std_dev_val, cv_val = (
                        self._compute_circular_stats(pixels)
                    )
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
            LogManager().get_logger().exception("HSV feature extraction failed")
            raise

    def _compute_circular_stats(
        self, pixels: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """
        Hue チャンネル用の循環統計を計算する.

        OpenCV の Hue は [0, 180) の循環量であり, 線形統計では
        境界付近 (赤付近: H~0 と H~179) で不正確になるため,
        循環統計 (circular statistics) を使用する.

        Args:
            pixels: Hue ピクセル値の配列 (float64, 範囲 [0, 180)).

        Returns:
            (mean, median, variance, std_dev, cv) のタプル.
        """
        high = self._HUE_HIGH

        # ラジアンに変換して循環統計を計算
        radians = pixels * (2 * np.pi / high)
        mean_rad = circmean(radians, high=2 * np.pi, low=0)
        std_rad = circstd(radians, high=2 * np.pi, low=0)

        # Hue スケール [0, 180) に戻す
        mean_val = float(mean_rad * high / (2 * np.pi))
        std_dev_val = float(std_rad * high / (2 * np.pi))
        variance_val = std_dev_val**2

        # 循環中央値: 循環平均を中心にシフトして線形中央値を取り, 戻す
        shift = high / 2 - mean_val
        shifted = (pixels + shift) % high
        median_shifted = float(np.median(shifted))
        median_val = (median_shifted - shift) % high

        cv_val = float(std_dev_val / mean_val) if mean_val != 0 else float("inf")

        return mean_val, median_val, variance_val, std_dev_val, cv_val

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        HSVStatisticsExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - exclude_black_pixels: RGB値がすべて0のピクセルを除外するかどうか
        """
        return {"exclude_black_pixels": True}

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = HSVStatisticsExtractor.get_base_feature_names()
        return [
            f"{name}[{HSVStatisticsExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        feature_names = []
        channels = ["hue", "saturation", "value"]
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
        base_names = HSVStatisticsExtractor.get_base_feature_names()

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = HSVStatisticsExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "hue_mean"）

        Returns:
            str: 対応する単位
        """
        return HSVStatisticsExtractor._FEATURE_UNITS.get(feature_name, "unknown")
