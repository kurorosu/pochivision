"""大津の2値化バリデータの実装モジュール."""

from typing import Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class OtsuBinarizationValidator(BaseValidator):
    """
    大津の2値化用のバリデータ.

    Args:
        config (dict): バリデーション対象の設定辞書.
        image (np.ndarray): 入力画像.

    Raises:
        ProcessorValidationError: 不正なパラメータや画像が検出された場合.
    """

    def __init__(self, config: Dict[str, int], image: np.ndarray) -> None:
        """
        OtsuBinarizationValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
            image (np.ndarray): 入力画像.
        """
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータや画像が検出された場合.
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)

        # 2Dまたは3/4チャンネル画像のみ許可
        if not (
            (self.image.ndim == 2)
            or (self.image.ndim == 3 and self.image.shape[2] in (3, 4))
        ):
            raise ProcessorValidationError(
                "Input image must be 2D grayscale or "
                "3/4 channel color image (BGR/BGRA)."
            )

        # 大津の2値化は特別な設定を必要としないため、追加のバリデーションは不要
