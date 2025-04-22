from typing import Any
from processors.validators.base import BaseValidator
from exceptions import ProcessorValidationError


class GrayscaleInputValidator(BaseValidator):
    """
    グレースケール変換用の入力画像バリデータ。

    Args:
        image (np.ndarray): 入力画像

    Raises:
        ProcessorValidationError: 不正な画像が渡された場合
    """

    def __init__(self, image: Any) -> None:
        self.image = image

    def validate(self) -> None:
        """
        入力画像のバリデーションを実行する。

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)
        # チャンネル数チェック
        if self.image.ndim != 3 or self.image.shape[2] not in (3, 4):
            raise ProcessorValidationError(
                "Input image must be a color image (BGR/BGRA) with 3 or 4 channels.")
