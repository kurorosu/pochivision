"""バイラテラルフィルタ用バリデータの実装モジュール."""

from typing import Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class BilateralFilterValidator(BaseValidator):
    """バイラテラルフィルタ用のバリデータ."""

    def __init__(self, config: Dict[str, int], image: np.ndarray) -> None:
        """
        BilateralFilterValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
            image (np.ndarray): 入力画像.
        """
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータや画像が検出された場合.
        """
        if self.image is not None:
            self.validate_image_type_and_nonempty(self.image)
        d = self.config.get("d", 9)
        sigmaColor = self.config.get("sigmaColor", 75)
        sigmaSpace = self.config.get("sigmaSpace", 75)
        if not (isinstance(d, int) and d > 0):
            raise ProcessorValidationError("d must be a positive integer. Example: 9")
        if not (isinstance(sigmaColor, (int, float)) and sigmaColor > 0):
            raise ProcessorValidationError(
                "sigmaColor must be a positive number. Example: 75"
            )
        if not (isinstance(sigmaSpace, (int, float)) and sigmaSpace > 0):
            raise ProcessorValidationError(
                "sigmaSpace must be a positive number. Example: 75"
            )
