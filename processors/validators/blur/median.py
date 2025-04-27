"""メディアンブラー用バリデータの実装モジュール."""

from typing import Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class MedianBlurValidator(BaseValidator):
    """メディアンブラー用のバリデータ."""

    def __init__(self, config: Dict[str, int], image: np.ndarray) -> None:
        """
        MedianBlurValidatorのコンストラクタ.

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
        kernel_size = self.config.get("kernel_size", 5)
        if not (
            isinstance(kernel_size, int) and kernel_size > 0 and kernel_size % 2 == 1
        ):
            raise ProcessorValidationError(
                "kernel_size must be a positive odd integer. Example: 5"
            )
