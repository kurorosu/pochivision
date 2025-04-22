from typing import Dict
from processors.validators.base import BaseValidator
import numpy as np
from exceptions import ProcessorValidationError


class MedianBlurValidator(BaseValidator):
    """
    メディアンブラーの設定値・画像バリデーションを担当するクラス。

    Args:
        config (dict): バリデーション対象の設定辞書
        image (np.ndarray, optional): 入力画像

    Raises:
        ProcessorValidationError: 不正なパラメータや画像が検出された場合
    """

    def __init__(self, config: Dict[str, int], image: np.ndarray = None) -> None:
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する。

        Raises:
            ProcessorValidationError: 不正なパラメータや画像が検出された場合
        """
        if self.image is not None:
            self.validate_image_type_and_nonempty(self.image)
        kernel_size = self.config.get("kernel_size", 5)
        if not (isinstance(kernel_size, int) and kernel_size > 0 and kernel_size % 2 == 1):
            raise ProcessorValidationError(
                "kernel_size must be a positive odd integer. Example: 5")
