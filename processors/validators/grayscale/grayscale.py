"""グレースケール入力バリデータの実装モジュール."""

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class GrayscaleInputValidator(BaseValidator):
    """グレースケール入力用のバリデータ."""

    def __init__(self, image: np.ndarray) -> None:
        """
        GrayscaleInputValidatorのコンストラクタ.

        Args:
            image (np.ndarray): 入力画像.
        """
        self.image = image

    def validate(self) -> None:
        """
        入力画像のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合.
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)
        # チャンネル数チェック
        if self.image.ndim != 3 or self.image.shape[2] not in (3, 4):
            raise ProcessorValidationError(
                "Input image must be a color image (BGR/BGRA) with 3 or 4 channels."
            )
