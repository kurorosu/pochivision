"""適応的2値化バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


def _validate_adaptive_block_size(config: dict[str, Any], processor_label: str) -> None:
    """
    Adaptive 2値化系プロセッサの ``block_size`` を起動時に検証する共通関数.

    ``block_size`` は ``int`` 型かつ 3 以上の奇数でなければならない.

    Args:
        config (dict[str, Any]): バリデーション対象の設定辞書.
        processor_label (str): エラーメッセージに含めるプロセッサ名.

    Raises:
        ProcessorValidationError: ``block_size`` が不正な場合.
    """
    if "block_size" not in config:
        return
    block_size = config["block_size"]
    # bool は int のサブクラスだが, ここでは無効として扱う.
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ProcessorValidationError(
            f"{processor_label}: block_size must be an int, got "
            f"{type(block_size).__name__} ({block_size!r})"
        )
    if block_size < 3 or block_size % 2 == 0:
        raise ProcessorValidationError(
            f"{processor_label}: block_size must be an odd integer >= 3, "
            f"got {block_size}"
        )


class GaussianAdaptiveBinarizationValidator(BaseValidator):
    """
    ガウシアン適応的2値化用のバリデータ.

    Args:
        config (dict[str, Any]): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        GaussianAdaptiveBinarizationValidatorのコンストラクタ.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: ``block_size`` が奇数かつ 3 以上でない場合.
        """
        self.config = config
        _validate_adaptive_block_size(config, "GaussianAdaptiveBinarization")

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証します.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        self.validate_image_type_and_nonempty(image)

        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorValidationError(
                "Input image for GaussianAdaptiveBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )


class MeanAdaptiveBinarizationValidator(BaseValidator):
    """
    平均適応的2値化用のバリデータ.

    Args:
        config (dict[str, Any]): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        MeanAdaptiveBinarizationValidatorのコンストラクタ.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: ``block_size`` が奇数かつ 3 以上でない場合.
        """
        self.config = config
        _validate_adaptive_block_size(config, "MeanAdaptiveBinarization")

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証します.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        self.validate_image_type_and_nonempty(image)

        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorValidationError(
                "Input image for MeanAdaptiveBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )
