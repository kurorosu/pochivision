"""リサイズプロセッサー用バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class ResizeConfigValidator(BaseValidator):
    """リサイズプロセッサー用のバリデータ."""

    def __init__(self, config: Dict[str, Any], image: np.ndarray = None) -> None:
        """
        ResizeConfigValidatorのコンストラクタ.

        Args:
            config (Dict[str, Any]): バリデーション対象の設定辞書.
            image (np.ndarray, optional): 入力画像. Defaults to None.
        """
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータや画像が検出された場合.
        """
        # 画像バリデーション（imageが指定されている場合のみ）
        if self.image is not None:
            self.validate_image_type_and_nonempty(self.image)

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
