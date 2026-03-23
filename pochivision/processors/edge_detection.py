"""Cannyエッジ検出プロセッサーを定義します."""

from typing import Any, Dict

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.utils.image import to_grayscale

from .base import BaseProcessor
from .registry import register_processor
from .validators.edge_detection.canny import CannyEdgeValidator


@register_processor("canny_edge")
class CannyEdgeProcessor(BaseProcessor):
    """画像にCannyエッジ検出を適用します."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        CannyEdgeProcessorを初期化.

        Args:
            name (str): プロセッサの名前.
            config (Dict[str, Any]): Cannyエッジ検出の設定辞書.
        """
        super().__init__(name, config)
        self.validator = CannyEdgeValidator(
            self.config
        )  # バリデータのインスタンス化とconfigの引き渡し
        self.validator.validate_config()  # 設定のバリデーションを実行

        default_vals = self.get_default_config()
        self._threshold1 = self.config.get("threshold1", default_vals["threshold1"])
        self._threshold2 = self.config.get("threshold2", default_vals["threshold2"])
        self._aperture_size = self.config.get(
            "aperture_size", default_vals["aperture_size"]
        )
        self._l2_gradient = self.config.get("l2_gradient", default_vals["l2_gradient"])

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        入力画像にCannyエッジ検出を適用します.

        Args:
            image (np.ndarray): 入力画像 (BGRまたはグレースケール).

        Returns:
            np.ndarray: 結果のエッジ検出画像 (グレースケール).

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
            ProcessorRuntimeError: 画像処理中にエラーが発生した場合.
        """
        self.validator.validate_image(image)  # 入力画像のバリデーションを実行

        gray_image = to_grayscale(image)

        if gray_image.dtype != np.uint8:
            if np.max(gray_image) <= 1.0 and gray_image.dtype == np.float32:
                gray_image = (gray_image * 255).astype(np.uint8)
            elif np.max(gray_image) <= 255:
                gray_image = gray_image.astype(np.uint8)
            else:
                try:
                    gray_image = cv2.normalize(
                        gray_image, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                except cv2.error as e:
                    raise ProcessorRuntimeError(
                        f"Failed to convert image to uint8 for Canny: {e}"
                    )

        try:
            edges = cv2.Canny(
                gray_image,
                int(self._threshold1),
                int(self._threshold2),
                apertureSize=self._aperture_size,
                L2gradient=self._l2_gradient,
            )
        except cv2.error as e:
            # cv2.Canny で予期せぬエラーが発生した場合のフォールバック
            raise ProcessorRuntimeError(f"Error during Canny edge detection: {e}")

        return edges

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        CannyEdgeProcessorのデフォルト設定を返します.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {
            "threshold1": 100.0,
            "threshold2": 200.0,
            "aperture_size": 3,
            "l2_gradient": False,
        }
