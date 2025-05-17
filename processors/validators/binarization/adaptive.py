"""適応的2値化バリデータの実装モジュール."""

from typing import Dict

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class GaussianAdaptiveBinarizationValidator(BaseValidator):
    """
    ガウシアン適応的2値化用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: Dict[str, int]) -> None:
        """
        GaussianAdaptiveBinarizationValidatorのコンストラクタ.

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

        # block_sizeのバリデーション
        block_size = self.config.get("block_size", 11)

        # 明示的に型と値をチェック
        if not isinstance(block_size, int):
            raise ProcessorValidationError(
                "block_size must be an integer. " "Example: 11"
            )

        # 奇数かつ3以上であることを確認
        if block_size < 3 or block_size % 2 == 0:
            raise ProcessorValidationError(
                "block_size must be an odd integer greater than or equal to 3. "
                "Example: 11"
            )

        # cのバリデーション
        c = self.config.get("c", 2)
        if not isinstance(c, (int, float)):
            raise ProcessorValidationError("c must be a number. Example: 2")


class MeanAdaptiveBinarizationValidator(BaseValidator):
    """
    平均適応的2値化用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    def __init__(self, config: Dict[str, int]) -> None:
        """
        MeanAdaptiveBinarizationValidatorのコンストラクタ.

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

        # block_sizeのバリデーション
        block_size = self.config.get("block_size", 11)

        # 明示的に型と値をチェック
        if not isinstance(block_size, int):
            raise ProcessorValidationError(
                "block_size must be an integer. " "Example: 11"
            )

        # 奇数かつ3以上であることを確認
        if block_size < 3 or block_size % 2 == 0:
            raise ProcessorValidationError(
                "block_size must be an odd integer greater than or equal to 3. "
                "Example: 11"
            )

        # cのバリデーション
        c = self.config.get("c", 2)
        if not isinstance(c, (int, float)):
            raise ProcessorValidationError("c must be a number. Example: 2")
