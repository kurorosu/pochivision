"""大津の2値化バリデータの実装モジュール."""

from typing import Dict

# from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class OtsuBinarizationValidator(BaseValidator):
    """
    大津の2値化用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: Dict[str, int]) -> None:
        """
        OtsuBinarizationValidatorのコンストラクタ.

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
        # 共通バリデーション
        # self.validate_image_type_and_nonempty(self.image) # processメソッドに移動

        # 2Dまたは3/4チャンネル画像のみ許可
        # if not ( # processメソッドに移動
        #     (self.image.ndim == 2)
        #     or (self.image.ndim == 3 and self.image.shape[2] in (3, 4))
        # ):
        #     raise ProcessorValidationError(
        #         "Input image must be 2D grayscale or "
        #         "3/4 channel color image (BGR/BGRA)."
        #     )

        # 大津の2値化は特別な設定を必要としないため、追加のバリデーションは不要
        pass  # 設定パラメータに関するバリデーションは無いが、メソッドは必要
