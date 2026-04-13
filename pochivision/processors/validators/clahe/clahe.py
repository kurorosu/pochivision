"""CLAHE入力バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class CLAHEInputValidator(BaseValidator):
    """CLAHE入力用のバリデータ."""

    processor_name = "clahe"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        CLAHEInputValidatorのコンストラクタ.

        Args:
            config (dict[str, Any] | None, optional): 設定パラメータ. デフォルトはNone.
        """
        self.config = config or {}

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像のバリデーションを実行する.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合.
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(image)

        # 画像データ型チェック
        if image.dtype != np.uint8:
            raise ProcessorValidationError(
                self._format_error(
                    f"Image data type must be np.uint8, got {image.dtype}"
                )
            )
