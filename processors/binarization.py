"""2値化処理プロセッサの実装を提供するモジュール."""

import logging
from typing import Dict

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from processors.validators.binarization import StandardBinarizationValidator
from processors.validators.binarization.adaptive import (
    GaussianAdaptiveBinarizationValidator,
    MeanAdaptiveBinarizationValidator,
)
from processors.validators.binarization.otsu import OtsuBinarizationValidator
from utils.image import to_grayscale


@register_processor("std_bin")
class StandardBinarizationProcessor(BaseProcessor):
    """
    スタンダードな2値化（しきい値による通常の2値化）を行う画像処理プロセッサ.

    入力画像がグレースケールまたはカラーかを自動判別し、
    適切に2値化処理（cv2.threshold）を行います.

    登録名:
        "std_bin"

    設定例:
        {
            "std_bin": {
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
        self.validator = StandardBinarizationValidator(self.config)
        self.validator.validate()
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
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorRuntimeError(
                "Input image for StandardBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )

        try:
            gray = to_grayscale(image)
            self.logger.debug(
                "Processing input image: original shape=%s, "
                "after grayscale conversion=%s",
                image.shape,
                gray.shape,
            )
        except ValueError as e:
            self.logger.error("Image conversion failed: %s", str(e))
            raise ProcessorRuntimeError(f"Image conversion failed: {e}")

        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        self.logger.info(f"Applied binarization with threshold {self.threshold}")
        return binary


@register_processor("otsu_bin")
class OtsuBinarizationProcessor(BaseProcessor):
    """
    大津の2値化（Otsu's Binarization）を行う画像処理プロセッサ.

    画像の2値化において、最適なしきい値を自動的に決定します.
    画像のヒストグラムの分散を最小化する閾値を見つけます.

    登録名:
        "otsu_bin"

    設定例:
        {
            "otsu_bin": {}  # パラメータは不要
        }
    """

    def __init__(self, name: str, config: Dict[str, int]) -> None:
        """
        OtsuBinarizationProcessorのコンストラクタ.

        Args:
            name (str): プロセッサ名.
            config (dict, optional): 設定パラメータ.
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.validator = OtsuBinarizationValidator(self.config)
        self.validator.validate()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        大津の2値化処理を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 2値化後の画像.

        Raises:
            ProcessorRuntimeError: 入力画像の検証に失敗した場合.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorRuntimeError(
                "Input image for OtsuBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )

        try:
            gray = to_grayscale(image)
            self.logger.debug(
                "Processing input image: original shape=%s, "
                "after grayscale conversion=%s",
                image.shape,
                gray.shape,
            )
        except ValueError as e:
            self.logger.error("Image conversion failed: %s", str(e))
            raise ProcessorRuntimeError(f"Image conversion failed: {e}")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.logger.info("Applied Otsu's binarization")
        return binary


@register_processor("gauss_adapt_bin")
class GaussianAdaptiveBinarizationProcessor(BaseProcessor):
    """
    ガウシアン適応的2値化を行う画像処理プロセッサ.

    画像の局所的な領域に対してガウシアン重み付けを用いて
    動的にしきい値を決定し、2値化を行います.

    登録名:
        "gauss_adapt_bin"

    設定例:
        {
            "block_size": 11,
            "c": 2
        }
    """

    def __init__(self, name: str, config: Dict[str, int]) -> None:
        """
        GaussianAdaptiveBinarizationProcessorのコンストラクタ.

        Args:
            name (str): プロセッサ名.
            config (dict, optional): 設定パラメータ.

        Raises:
            ProcessorRuntimeError: 不正な設定値が検出された場合.
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.validator = GaussianAdaptiveBinarizationValidator(self.config)
        self.validator.validate()
        self.block_size: int = self.config.get("block_size", 11)
        self.c_value: int | float = self.config.get("c", 2)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ガウシアン適応的2値化処理を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 2値化後の画像.

        Raises:
            ProcessorRuntimeError: 入力画像の検証に失敗した場合.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorRuntimeError(
                "Input image for GaussianAdaptiveBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )

        try:
            gray = to_grayscale(image)
            self.logger.debug(
                "Processing input image: original shape=%s, "
                "after grayscale conversion=%s",
                image.shape,
                gray.shape,
            )
        except ValueError as e:
            self.logger.error("Image conversion failed: %s", str(e))
            raise ProcessorRuntimeError(f"Image conversion failed: {e}")

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c_value,
        )
        self.logger.info(
            f"Applied Gaussian adaptive binarization with block_size={self.block_size},"
            f"c={self.c_value}"
        )
        return binary


@register_processor("mean_adapt_bin")
class MeanAdaptiveBinarizationProcessor(BaseProcessor):
    """
    平均適応的2値化を行う画像処理プロセッサ.

    画像の局所的な領域の平均値を用いて
    動的にしきい値を決定し、2値化を行います.

    登録名:
        "mean_adapt_bin"

    設定例:
        {
            "block_size": 11,
            "c": 2
        }
    """

    def __init__(self, name: str, config: Dict[str, int]) -> None:
        """
        MeanAdaptiveBinarizationProcessorのコンストラクタ.

        Args:
            name (str): プロセッサ名.
            config (dict, optional): 設定パラメータ.

        Raises:
            ProcessorRuntimeError: 不正な設定値が検出された場合.
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.validator = MeanAdaptiveBinarizationValidator(self.config)
        self.validator.validate()
        self.block_size: int = self.config.get("block_size", 11)
        self.c_value: int | float = self.config.get("c", 2)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        平均適応的2値化処理を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 2値化後の画像.

        Raises:
            ProcessorRuntimeError: 入力画像の検証に失敗した場合.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorRuntimeError(
                "Input image for MeanAdaptiveBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )

        try:
            gray = to_grayscale(image)
            self.logger.debug(
                "Processing input image: original shape=%s, "
                "after grayscale conversion=%s",
                image.shape,
                gray.shape,
            )
        except ValueError as e:
            self.logger.error("Image conversion failed: %s", str(e))
            raise ProcessorRuntimeError(f"Image conversion failed: {e}")

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c_value,
        )
        self.logger.info(
            f"Applied Mean adaptive binarization with block_size={self.block_size},"
            f"c={self.c_value}"
        )
        return binary
