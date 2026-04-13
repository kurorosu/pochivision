"""CLAHE入力バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class CLAHEInputValidator(BaseValidator):
    """CLAHE入力用のバリデータ."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        CLAHEInputValidatorのコンストラクタ.

        Args:
            config (dict[str, Any] | None, optional): 設定パラメータ. デフォルトはNone.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        self.config = config or {}
        self.validate_config(self.config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        ``color_mode`` は ``"gray"`` , ``"lab"`` , ``"bgr"`` のいずれか,
        ``clip_limit`` は 0 より大きい数値,
        ``tile_grid_size`` は長さ 2 の list/tuple で両要素 1 以上の int でなければならない.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        if "color_mode" in config and config["color_mode"] is not None:
            color_mode = config["color_mode"]
            if color_mode not in ("gray", "lab", "bgr"):
                raise ProcessorValidationError(
                    "color_mode must be one of 'gray', 'lab', 'bgr', "
                    f"got {color_mode!r}"
                )
        if "clip_limit" in config and config["clip_limit"] is not None:
            clip_limit = config["clip_limit"]
            if isinstance(clip_limit, bool) or not isinstance(clip_limit, (int, float)):
                raise ProcessorValidationError(
                    f"clip_limit must be a number, got {clip_limit!r}"
                )
            if clip_limit <= 0:
                raise ProcessorValidationError(
                    f"clip_limit must be > 0, got {clip_limit}"
                )
        if "tile_grid_size" in config and config["tile_grid_size"] is not None:
            tile_grid_size = config["tile_grid_size"]
            if (
                not isinstance(tile_grid_size, (list, tuple))
                or len(tile_grid_size) != 2
            ):
                raise ProcessorValidationError(
                    "tile_grid_size must be a list/tuple of length 2, "
                    f"got {tile_grid_size!r}"
                )
            for v in tile_grid_size:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise ProcessorValidationError(
                        f"tile_grid_size elements must be int, got {v!r}"
                    )
                if v < 1:
                    raise ProcessorValidationError(
                        "tile_grid_size elements must be >= 1, "
                        f"got {tile_grid_size!r}"
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
