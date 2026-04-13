"""ガウシアンブラー用バリデータの実装モジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class GaussianBlurValidator(BaseValidator):
    """ガウシアンブラー用のバリデータ."""

    processor_name = "gaussian_blur"

    def __init__(self, config: dict[str, Any]) -> None:
        """
        GaussianBlurValidatorのコンストラクタ.

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

        kernel_size は長さ 2 の list/tuple で, 両要素とも 3 以上の奇数でなければならない.

        Args:
            config (dict): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: kernel_size が不正な場合.
        """
        if "kernel_size" not in config:
            # kernel_size 未指定時はデフォルト値を使用するため検証不要
            return

        kernel_size = config["kernel_size"]
        if not isinstance(kernel_size, (list, tuple)) or len(kernel_size) != 2:
            raise ProcessorValidationError(
                self._format_error(
                    "kernel_size must be a list/tuple of length 2, "
                    f"got {kernel_size!r}"
                )
            )
        for v in kernel_size:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ProcessorValidationError(
                    self._format_error(f"kernel_size elements must be int, got {v!r}")
                )
            if v < 3 or v % 2 == 0:
                raise ProcessorValidationError(
                    self._format_error(
                        "kernel_size elements must be odd integers >= 3, "
                        f"got {kernel_size!r}"
                    )
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
            raise ProcessorValidationError(
                self._format_error(
                    f"Input image must be of type np.uint8, got {image.dtype}"
                )
            )

        # 2Dグレースケール画像または3チャンネルのBGR画像であることを確認
        if image.ndim not in [2, 3]:
            raise ProcessorValidationError(
                self._format_error(
                    "Input image must be a 2D (grayscale) or 3-channel (BGR) "
                    f"image, got ndim={image.ndim} (shape={image.shape})"
                )
            )

        # 3Dの場合、チャンネル数が3であることを確認
        if image.ndim == 3 and image.shape[2] not in [1, 3]:
            raise ProcessorValidationError(
                self._format_error(
                    "Input image must have 1 or 3 channels for 3D images, "
                    f"got {image.shape[2]} (shape={image.shape})"
                )
            )
