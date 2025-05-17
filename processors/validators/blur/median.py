"""メディアンブラー用バリデータの実装モジュール."""

from typing import Dict

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class MedianBlurValidator(BaseValidator):
    """メディアンブラー用のバリデータ."""

    def __init__(self, config: Dict[str, int]) -> None:
        """
        MedianBlurValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate(self) -> None:
        """
        設定値のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータが検出された場合.
        """
        kernel_size = self.config.get("kernel_size", 5)
        if not (
            isinstance(kernel_size, int) and kernel_size > 0 and kernel_size % 2 == 1
        ):
            raise ProcessorValidationError(
                "kernel_size must be a positive odd integer. Example: 5"
            )
