"""メディアンブラー用バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class MedianBlurValidator(BaseValidator):
    """メディアンブラー用のバリデータ."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        MedianBlurValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
        """
        self.config = config
        self.kernel_size: int | None = None

    def validate(self) -> None:
        """
        設定値のバリデーションを実行する.

        BaseValidatorの抽象メソッドを満たすために定義されています.
        """
        self.validate_config()

    def validate_config(self) -> None:
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

        # バリデーション後に値を保持
        self.kernel_size = kernel_size

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像のバリデーションを実行する.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 不正な画像が渡された場合.
        """
        # 基本的な画像バリデーション (BaseValidatorのメソッドを利用)
        self.validate_image_type_and_nonempty(image)

        # 追加のバリデーション
        if image.dtype != np.uint8:
            raise ProcessorValidationError("Input image must be of type np.uint8")

        # 2Dグレースケール画像または3チャンネルのBGR画像であることを確認
        if image.ndim not in [2, 3]:
            raise ProcessorValidationError(
                "Input image must be a 2D (grayscale) or 3-channel (BGR) image"
            )

        # 3Dの場合、チャンネル数が3であることを確認
        if image.ndim == 3 and image.shape[2] not in [1, 3]:
            raise ProcessorValidationError(
                "Input image must have 1 or 3 channels for 3D images"
            )
