"""標準2値化バリデータの実装モジュール."""

from typing import Dict

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class StandardBinarizationValidator(BaseValidator):
    """
    スタンダードな2値化（しきい値による通常の2値化）用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: Dict[str, int]) -> None:
        """
        StandardBinarizationValidatorのコンストラクタ.

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
        # threshold値のバリデーション
        threshold = self.config.get("threshold", 128)
        if not (isinstance(threshold, int) and 0 <= threshold <= 255):
            raise ProcessorValidationError(
                "threshold must be an integer between 0 and 255."
            )
