"""大津の2値化バリデータの実装モジュール."""

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class OtsuBinarizationValidator(BaseValidator):
    """
    大津の2値化用のバリデータ.

    Args:
        config (dict[str, int]): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: dict[str, int]) -> None:
        """
        OtsuBinarizationValidatorのコンストラクタ.

        Args:
            config (dict[str, int]): バリデーション対象の設定辞書.
        """
        self.config = config

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
                "Input image for OtsuBinarization must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )
