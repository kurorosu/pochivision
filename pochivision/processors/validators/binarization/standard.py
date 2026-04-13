"""標準2値化バリデータの実装モジュール."""

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class StandardBinarizationValidator(BaseValidator):
    """
    スタンダードな2値化（しきい値による通常の2値化）用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.

    Raises:
        ProcessorValidationError: 不正なパラメータが検出された場合.
    """

    processor_name = "std_bin"

    def __init__(self, config: dict[str, int]) -> None:
        """
        StandardBinarizationValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証します.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        self.validate_image_type_and_nonempty(image)

        # 標準2値化は最終的にグレースケール画像に対して処理を行うが、
        # to_grayscale関数でカラーからグレースケールへの変換をサポートするため、
        # ここでは to_grayscale が受け付ける形式 (2D または 3/4チャンネル) を許容する。
        # to_grayscale 内部でさらに詳細なチェックが行われる。
        if not ((image.ndim == 2) or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ProcessorValidationError(
                self._format_error(
                    "Input image for StandardBinarization must be 2D grayscale or "
                    "3/4 channel color image (BGR/BGRA), "
                    f"got ndim={image.ndim} (shape={image.shape})"
                )
            )
