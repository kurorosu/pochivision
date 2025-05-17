"""ヒストグラム平坦化入力バリデータの実装モジュール."""

from typing import Any, Dict, Optional

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class EqualizeInputValidator(BaseValidator):
    """ヒストグラム平坦化入力用のバリデータ."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        EqualizeInputValidatorのコンストラクタ.

        Args:
            config (Optional[Dict[str, Any]], optional): 設定パラメータ. デフォルトはNone.
        """
        self.config = config or {}

    def validate(self) -> None:
        """
        設定パラメータのバリデーションを実行します.

        BaseValidatorの抽象メソッドを満たすために定義されています.
        """
        self.validate_config()

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
