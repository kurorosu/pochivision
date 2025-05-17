"""大津の2値化バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class OtsuBinarizationValidator(BaseValidator):
    """
    大津の2値化用のバリデータ.

    Args:
        config (Dict[str, Any]): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        OtsuBinarizationValidatorのコンストラクタ.

        Args:
            config (Dict[str, Any]): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate(self) -> None:
        """
        設定パラメータのバリデーションを実行します.

        BaseValidatorの抽象メソッドを満たすために定義されています.
        """
        self.validate_config()

    def validate_config(self) -> None:
        """
        設定値のバリデーションを実行する.

        大津の2値化は設定パラメータを持たないため、このメソッドは何も行いません.
        """
        pass  # 設定パラメータに関するバリデーションは無い

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
