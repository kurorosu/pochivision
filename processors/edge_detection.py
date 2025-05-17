"""Cannyエッジ検出プロセッサーを定義します."""

from typing import Any, Dict

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from utils.image import to_grayscale

from .base import BaseProcessor
from .registry import register_processor
from .validators.edge_detection.canny import CannyConfigValidator


@register_processor("canny_edge")
class CannyEdgeProcessor(BaseProcessor):
    """画像にCannyエッジ検出を適用します."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        CannyEdgeProcessorを初期化.

        Args:
            name (str): プロセッサの名前.
            config (Dict[str, Any]): Cannyエッジ検出の設定辞書.
                configにキーが存在しない場合はデフォルト値が使用されます。
                期待されるキー:
                - threshold1 (float): ヒステリシス手順の最初のしきい値.
                - threshold2 (float): ヒステリシス手順の2番目のしきい値.
                - aperture_size (int, optional): Sobelオペレータのアパーチャサイズ.
                  デフォルトは3. 3から7の奇数である必要があります.
                - l2_gradient (bool, optional): グラデーションの大きさをより正確に計算するために
                  L2ノルムを使用するかどうかを示すフラグ. デフォルトはFalse.
        """
        super().__init__(name, config)
        # バリデーターはconfigに存在するキーの値の妥当性のみをチェックする
        self.validator = CannyConfigValidator(self.config)  # self.config を渡す
        self.validator.validate()

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

        入力画像がカラーの場合、最初にグレースケールに変換されます.

        Args:
            image (np.ndarray): 入力画像 (BGRまたはグレースケール).

        Returns:
            np.ndarray: 結果のエッジ検出画像 (グレースケール).

        Raises:
            ProcessorRuntimeError: 入力画像が処理不可能な形式の場合.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
            # Cannyはグレースケール画像を期待するが、入力として一般的なカラー(3ch)も許容する
            # to_grayscale で処理できない、または意図しない形式の場合はここでエラー
            raise ProcessorRuntimeError(
                "Input image for CannyEdgeProcessor must be 2D grayscale "
                "or 3-channel color image."
            )

        gray_image = to_grayscale(image)

        # Ensure the image is 8-bit, as Canny requires it.
        if gray_image.dtype != np.uint8:
            if (
                np.max(gray_image) <= 1.0 and gray_image.dtype == np.float32
            ):  # Assuming 0-1 float
                gray_image = (gray_image * 255).astype(np.uint8)
            elif np.max(gray_image) <= 255:  # Assuming 0-255 but not uint8
                gray_image = gray_image.astype(np.uint8)
            else:
                # Attempt to normalize if it's a larger dtype like uint16 or int32
                gray_image = cv2.normalize(
                    gray_image, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

        edges = cv2.Canny(
            gray_image,
            int(self._threshold1),  # cv2.Canny expects int
            int(self._threshold2),  # cv2.Canny expects int
            apertureSize=self._aperture_size,
            L2gradient=self._l2_gradient,
        )
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
