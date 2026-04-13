"""大津の2値化バリデータの実装モジュール."""

from typing import Any

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
        self.validate_config(dict(config))

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        大津の2値化はパラメータを持たないため検証不要.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.
        """
        return None

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
