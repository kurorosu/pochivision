"""ガウシアンブラー用バリデータの実装モジュール."""

from typing import Any, Dict

import numpy as np

from exceptions import ProcessorValidationError
from processors.validators.base import BaseValidator


class GaussianBlurValidator(BaseValidator):
    """ガウシアンブラー用のバリデータ."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        GaussianBlurValidatorのコンストラクタ.

        Args:
            config (dict): バリデーション対象の設定辞書.
        """
        self.config = config
        self.kernel_width: int | None = None
        self.kernel_height: int | None = None
        self.sigma_x: float | None = None
        self.sigma_y: float | None = None

    def validate_config(self) -> None:
        """
        設定値のバリデーションを実行する.

        Raises:
            ProcessorValidationError: 不正なパラメータが検出された場合.
        """
        kernel_size = self.config.get("kernel_size", [15, 15])
        sigma = self.config.get("sigma", 0)

        # カーネルサイズのバリデーション
        if not isinstance(kernel_size, (list, tuple)) or len(kernel_size) != 2:
            raise ProcessorValidationError(
                "kernel_size must be specified as two positive odd integers. "
                "Example: [15, 15]"
            )

        k_width, k_height = kernel_size
        if not (isinstance(k_width, int) and k_width > 0 and k_width % 2 == 1):
            raise ProcessorValidationError(
                "kernel_size width must be a positive odd integer."
            )
        if not (isinstance(k_height, int) and k_height > 0 and k_height % 2 == 1):
            raise ProcessorValidationError(
                "kernel_size height must be a positive odd integer."
            )
        self.kernel_width = k_width
        self.kernel_height = k_height

        # シグマ値のバリデーション
        if not isinstance(sigma, (int, float)) or sigma < 0:
            raise ProcessorValidationError("sigma must be a non-negative number")
        self.sigma_x = float(sigma)

        # sigmaYが明示的に指定されている場合の検証
        sigma_y = self.config.get("sigmaY", sigma)
        if not isinstance(sigma_y, (int, float)) or sigma_y < 0:
            raise ProcessorValidationError(
                "sigmaY (if provided) must be a non-negative number"
            )
        self.sigma_y = float(sigma_y)

        # kernel_sizeもsigmaも両方0は不可
        if (
            self.sigma_x == 0
            and self.sigma_y == 0
            and self.kernel_width == 1
            and self.kernel_height == 1
        ):
            raise ProcessorValidationError(
                "Both kernel_size and sigma are effectively zero, "
                "which won't produce any blur."
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
