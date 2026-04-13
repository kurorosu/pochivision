"""ヒストグラム平坦化入力バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class EqualizeInputValidator(BaseValidator):
    """ヒストグラム平坦化入力用のバリデータ."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        EqualizeInputValidatorのコンストラクタ.

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

        ``color_mode`` は ``"gray"`` , ``"lab"`` , ``"bgr"`` のいずれかでなければならない.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: ``color_mode`` が不正な場合.
        """
        if "color_mode" not in config or config["color_mode"] is None:
            return
        color_mode = config["color_mode"]
        if color_mode not in ("gray", "lab", "bgr"):
            raise ProcessorValidationError(
                "color_mode must be one of 'gray', 'lab', 'bgr', " f"got {color_mode!r}"
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
