"""リサイズプロセッサー用バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

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
