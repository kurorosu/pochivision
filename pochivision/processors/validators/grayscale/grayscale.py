"""グレースケール変換バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class GrayscaleValidator(BaseValidator):
    """グレースケール変換用のバリデータ."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        GrayscaleValidatorのコンストラクタ.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像のバリデーションを実行する.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合.
        """
        # 基本的な画像バリデーション
        self.validate_image_type_and_nonempty(image)

        # dtype チェック
        if image.dtype != np.uint8:
            raise ProcessorValidationError("Input image must be of type np.uint8")

        # 1チャンネル(グレースケール)または3チャンネル(BGRカラー)の画像であることを確認
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (1, 3, 4))):
            raise ProcessorValidationError(
                "Input image must be 2D grayscale or 3/4 channel color "
                "image (BGR/BGRA)."
            )
