"""リサイズプロセッサー用バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class ResizeConfigValidator(BaseValidator):
    """リサイズプロセッサー用のバリデータ."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        ResizeConfigValidatorのコンストラクタ.

        Args:
            config (Dict[str, Any]): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate_config(self) -> None:
        """
        設定値のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータが検出された場合.
        """
        width = self.config.get("width")
        height = self.config.get("height")
        preserve_aspect_ratio = self.config.get("preserve_aspect_ratio", False)
        aspect_ratio_mode = self.config.get("aspect_ratio_mode", "width")

        # widthとheightの少なくとも一方が指定されていることを確認
        if width is None and height is None:
            raise ProcessorValidationError(
                "Either width or height (or both) must be specified"
            )

        # widthが指定されている場合は正の整数であることを確認
        if width is not None and not (isinstance(width, int) and width > 0):
            raise ProcessorValidationError("width must be a positive integer")

        # heightが指定されている場合は正の整数であることを確認
        if height is not None and not (isinstance(height, int) and height > 0):
            raise ProcessorValidationError("height must be a positive integer")

        # preserve_aspect_ratioはbool型であること
        if not isinstance(preserve_aspect_ratio, bool):
            raise ProcessorValidationError("preserve_aspect_ratio must be a boolean")

        # aspect_ratio_modeは'width'または'height'であること
        if aspect_ratio_mode not in ["width", "height"]:
            raise ProcessorValidationError(
                "aspect_ratio_mode must be either 'width' or 'height'"
            )

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像のバリデーションを実行する.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合.
        """
        # 画像バリデーション
        self.validate_image_type_and_nonempty(image)
