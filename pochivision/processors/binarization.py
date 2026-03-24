"""2値化処理プロセッサの実装を提供するモジュール."""

import logging
from typing import Any, Dict

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.validators.binarization import StandardBinarizationValidator
from pochivision.processors.validators.binarization.adaptive import (
    GaussianAdaptiveBinarizationValidator,
    MeanAdaptiveBinarizationValidator,
)
from pochivision.processors.validators.binarization.otsu import (
    OtsuBinarizationValidator,
)
from pochivision.utils.image import to_grayscale


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
        self.validator.validate_config()
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
            ProcessorValidationError: 入力画像が無効な場合 (バリデーションによる).
        """
        self.validator.validate_image(image)

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

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        標準2値化プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"threshold": 128}


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        OtsuBinarizationProcessorのコンストラクタ.

        Args:
            name (str): プロセッサ名.
            config (dict, optional): 設定パラメータ.
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)
        self.validator = OtsuBinarizationValidator(self.config)
        self.validator.validate_config()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        大津の2値化処理を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: 2値化後の画像.

        Raises:
            ProcessorRuntimeError: 画像変換処理に失敗した場合.
            ProcessorValidationError: 入力画像が無効な場合 (バリデーションによる).
        """
        self.validator.validate_image(image)

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

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        大津の2値化プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定（空の辞書）.
        """
        return {}


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
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
        self.validator.validate_config()
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
            ProcessorRuntimeError: 画像変換処理に失敗した場合.
            ProcessorValidationError: 入力画像が無効な場合 (バリデーションによる).
        """
        self.validator.validate_image(image)

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
            f"Applied Gaussian adaptive binarization "
            f"(block_size={self.block_size}, C={self.c_value})"
        )
        return binary

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        ガウシアン適応的2値化プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"block_size": 11, "c": 2}


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
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
        self.validator.validate_config()
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
            ProcessorRuntimeError: 画像変換処理に失敗した場合.
            ProcessorValidationError: 入力画像が無効な場合 (バリデーションによる).
        """
        self.validator.validate_image(image)

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
            f"Applied Mean adaptive binarization "
            f"(block_size={self.block_size}, C={self.c_value})"
        )
        return binary

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        平均値による適応的2値化プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"block_size": 11, "c": 2}
