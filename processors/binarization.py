"""2値化処理プロセッサの実装を提供するモジュール."""

import logging
from typing import Dict

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from processors.validators.binarization import StandardBinarizationValidator


@register_processor("standard_binarization")
class StandardBinarizationProcessor(BaseProcessor):
    """
    スタンダードな2値化（しきい値による通常の2値化）を行う画像処理プロセッサ.

    入力画像がグレースケールまたはカラーかを自動判別し、
    適切に2値化処理（cv2.threshold）を行います.

    登録名:
        "standard_binarization"

    設定例:
        {
            "standard_binarization": {
                "threshold": 128
            }
        }
    Attributes:
        threshold (int): 2値化の閾値（0-255, デフォルト128）
    """

    def __init__(self, name: str, config: Dict[str, int]) -> None:
        """
        StandardBinarizationProcessorのコンストラクタ.

        Args:
            name (str): プロセッサ名.
            config (dict, optional): 設定パラメータ. デフォルトはNone.
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.threshold: int = self.config.get("threshold", 128)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        通常の2値化処理（cv2.threshold）を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 2値化後の画像.

        Raises:
            ProcessorRuntimeError: サポート外の画像形式の場合やバリデーション失敗時.
        """
        try:
            StandardBinarizationValidator(self.config, image).validate()
        except Exception as e:
            raise ProcessorRuntimeError(f"StandardBinarization validation failed: {e}")

        if image.ndim == 2:
            gray = image
            self.logger.debug("Input image is grayscale.")
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.logger.debug("Input image is color. Converted to grayscale.")
        else:
            self.logger.error(f"Unsupported image format: shape={image.shape}")
            raise ProcessorRuntimeError(
                "Unsupported image format. "
                "Only 2D or 3D (BGR/BGRA) images are supported."
            )

        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        self.logger.info(f"Applied binarization with threshold {self.threshold}.")
        return binary
