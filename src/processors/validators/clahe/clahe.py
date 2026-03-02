"""CLAHE入力バリデータの実装モジュール."""

from typing import Any, Dict, Optional

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class CLAHEInputValidator(BaseValidator):
    """CLAHE入力用のバリデータ."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        CLAHEInputValidatorのコンストラクタ.

        Args:
            config (Optional[Dict[str, Any]], optional): 設定パラメータ. デフォルトはNone.
        """
        self.config = config or {}

    def validate_config(self) -> None:
        """
        パラメータのバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータが渡された場合.
        """
        # color_modeのチェック
        color_mode = self.config.get("color_mode", "gray")
        if color_mode not in ["gray", "lab", "bgr"]:
            raise ProcessorValidationError(
                f"Invalid color_mode '{color_mode}'. "
                "Must be one of: 'gray', 'lab', 'bgr'."
            )

        # clip_limitのチェック
        clip_limit = self.config.get("clip_limit", 2.0)
        try:
            clip_limit = float(clip_limit)
            if clip_limit <= 0:
                raise ValueError("clip_limit must be a positive number")
        except (TypeError, ValueError):
            raise ProcessorValidationError(
                f"Invalid clip_limit '{clip_limit}'. " "Must be a positive number."
            )

        # tile_grid_sizeのチェック
        tile_grid_size = self.config.get("tile_grid_size", [8, 8])
        if not isinstance(tile_grid_size, list) or len(tile_grid_size) != 2:
            raise ProcessorValidationError(
                f"Invalid tile_grid_size '{tile_grid_size}'. "
                "Must be a list of 2 positive integers."
            )

        try:
            tile_grid_size = [int(x) for x in tile_grid_size]
            if any(x <= 0 for x in tile_grid_size):
                raise ValueError("tile_grid_size must contain positive integers")
        except (TypeError, ValueError):
            raise ProcessorValidationError(
                f"Invalid tile_grid_size '{tile_grid_size}'. "
                "Must be a list of 2 positive integers."
            )

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
                f"Image data type must be np.uint8, got {image.dtype}"
            )
