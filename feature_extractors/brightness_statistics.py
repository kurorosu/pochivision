"""輝度統計特徴量抽出を行うモジュール."""

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("brightness")
class BrightnessStatisticsExtractor(BaseFeatureExtractor):
    """
    画像の輝度統計特徴量を抽出するクラス.

    抽出する特徴量:
    - mean: 輝度平均値
    - median: 輝度中央値
    - variance: 輝度分散
    - std_dev: 輝度標準偏差
    - cv: 変動係数（標準偏差/平均値）

    設定により、輝度値が0のピクセルを計算から除外することができます。
    """

    def __init__(
        self,
        name: str = "brightness_statistics",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        BrightnessStatisticsExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "brightness_statistics".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.color_mode = self.config["color_mode"]
        self.exclude_zero_pixels = self.config["exclude_zero_pixels"]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像から輝度統計特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                - mean: 輝度平均値
                - median: 輝度中央値
                - variance: 輝度分散
                - std_dev: 輝度標準偏差
                - cv: 変動係数

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        # 輝度画像の取得
        brightness_image = self._get_brightness_image(image)

        # 統計値の計算
        pixels = brightness_image.flatten().astype(np.float64)

        # ゼロピクセル除外の処理
        if self.exclude_zero_pixels:
            # 輝度値0のピクセルを除外
            non_zero_pixels = pixels[pixels > 0]
            calculation_pixels = non_zero_pixels
        else:
            # すべてのピクセルを使用
            calculation_pixels = pixels

        if len(calculation_pixels) == 0:
            # 有効なピクセルがない場合
            mean_val = 0.0
            median_val = 0.0
            variance_val = 0.0
            std_dev_val = 0.0
            cv_val = float("inf")
        else:
            mean_val = float(np.mean(calculation_pixels))
            median_val = float(np.median(calculation_pixels))
            variance_val = float(np.var(calculation_pixels))
            std_dev_val = float(np.std(calculation_pixels))

            # 変動係数の計算（平均値が0の場合は無限大になるため特別処理）
            cv_val = float(std_dev_val / mean_val) if mean_val != 0 else float("inf")

        return {
            "mean": mean_val,
            "median": median_val,
            "variance": variance_val,
            "std_dev": std_dev_val,
            "cv": cv_val,
        }

    def _get_brightness_image(self, image: np.ndarray) -> np.ndarray:
        """
        入力画像から輝度画像を取得する.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 輝度画像（グレースケール）.

        Raises:
            ValueError: サポートされていないcolor_modeの場合.
        """
        if len(image.shape) == 2:
            # すでにグレースケール画像
            return image
        elif len(image.shape) == 3:
            if self.color_mode == "gray":
                # BGR to Grayscale
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif self.color_mode == "lab_l":
                # LAB色空間のL成分（明度）を使用
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                return lab[:, :, 0]  # L成分
            elif self.color_mode == "hsv_v":
                # HSV色空間のV成分（明度）を使用
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return hsv[:, :, 2]  # V成分
            else:
                raise ValueError(f"Unsupported color_mode: {self.color_mode}")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        BrightnessStatisticsExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - color_mode: 輝度計算モード ("gray", "lab_l", "hsv_v")
                - exclude_zero_pixels: 輝度値が0のピクセルを除外するかどうか
        """
        return {
            "color_mode": "gray",
            "exclude_zero_pixels": True,
        }

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す.

        Returns:
            list[str]: 特徴量名のリスト.
        """
        return ["mean", "median", "variance", "std_dev", "cv"]
