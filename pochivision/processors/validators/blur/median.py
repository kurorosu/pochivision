"""メディアンブラー用バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class MedianBlurValidator(BaseValidator):
    """メディアンブラー用のバリデータ."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        MedianBlurValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        self.config = config
        self.validate_config(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        kernel_size はスカラー整数で, 3 以上の奇数でなければならない.

        Args:
            config (dict): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: kernel_size が不正な場合.
        """
        if "kernel_size" not in config:
            # kernel_size 未指定時はデフォルト値を使用するため検証不要
            return

        kernel_size = config["kernel_size"]
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
            raise ProcessorValidationError(
                f"kernel_size must be an int, got {kernel_size!r}"
            )
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ProcessorValidationError(
                f"kernel_size must be an odd integer >= 3, got {kernel_size!r}"
            )

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
