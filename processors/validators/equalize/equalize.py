"""ヒストグラム平坦化入力バリデータの実装モジュール."""

from typing import Any, Dict, Optional

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class EqualizeInputValidator(BaseValidator):
    """ヒストグラム平坦化入力用のバリデータ."""

    def __init__(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        EqualizeInputValidatorのコンストラクタ.

        Args:
            image (np.ndarray): 入力画像.
            config (Optional[Dict[str, Any]], optional): 設定パラメータ. デフォルトはNone.
        """
        self.image = image
        self.config = config or {}

    def validate(self) -> None:
        """
        入力画像とパラメータのバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正な画像やパラメータが渡された場合.
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)

        # 画像データ型チェック
        if self.image.dtype != np.uint8:
            raise ProcessorValidationError(
                f"Image data type must be np.uint8, got {self.image.dtype}"
            )

        # color_modeのチェック
        color_mode = self.config.get("color_mode", "gray")
        if color_mode not in ["gray", "lab", "bgr"]:
            raise ProcessorValidationError(
                f"Invalid color_mode '{color_mode}'. "
                "Must be one of: 'gray', 'lab', 'bgr'."
            )
