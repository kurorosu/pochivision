"""平均ブラー用バリデータの実装モジュール."""

from typing import Dict

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class AverageBlurValidator(BaseValidator):
    """平均ブラー用のバリデータ."""

    def __init__(self, config: Dict[str, int]) -> None:
        """
        AverageBlurValidatorのコンストラクタ.

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
        kernel_size = self.config.get("kernel_size", [5, 5])
        if (
            not isinstance(kernel_size, (list, tuple))
            or len(kernel_size) != 2
            or not all(isinstance(k, int) and k > 0 for k in kernel_size)
        ):
            raise ProcessorValidationError(
                "kernel_size must be specified as two positive integers. "
                "Example: [5, 5]"
            )
