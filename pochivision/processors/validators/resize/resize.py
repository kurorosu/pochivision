"""リサイズプロセッサー用バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class ResizeConfigValidator(BaseValidator):
    """リサイズプロセッサー用のバリデータ."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        ResizeConfigValidatorのコンストラクタ.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        self.config = config
        self.validate_config(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        ``width`` および ``height`` は指定されている場合 1 以上の int でなければならない.
        ``aspect_ratio_mode`` は ``"width"`` または ``"height"`` でなければならない.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        for key in ("width", "height"):
            if key in config and config[key] is not None:
                value = config[key]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ProcessorValidationError(
                        f"{key} must be an int, got {value!r}"
                    )
                if value < 1:
                    raise ProcessorValidationError(f"{key} must be >= 1, got {value}")
        if "aspect_ratio_mode" in config and config["aspect_ratio_mode"] is not None:
            mode = config["aspect_ratio_mode"]
            if mode not in ("width", "height"):
                raise ProcessorValidationError(
                    "aspect_ratio_mode must be 'width' or 'height', " f"got {mode!r}"
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
