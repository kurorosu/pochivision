from typing import Dict
from processors.validators.base import BaseValidator
import numpy as np
from exceptions import ProcessorValidationError


class GaussianBlurConfigValidator(BaseValidator):
    """
    ガウシアンブラーの設定値・画像バリデーションを担当するクラス。

    Args:
        config (dict): バリデーション対象の設定辞書
        image (np.ndarray, optional): 入力画像

    Raises:
        ProcessorValidationError: 不正なパラメータや画像が検出された場合
    """

    def __init__(self, config: Dict[str, int], image: np.ndarray = None) -> None:
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する。

        Raises:
            ProcessorValidationError: 不正なパラメータや画像が検出された場合
        """
        # 画像バリデーション（imageが指定されている場合のみ）
        if self.image is not None:
            self.validate_image_type_and_nonempty(self.image)
        kernel_size = self.config.get("kernel_size", [15, 15])
        sigma = self.config.get("sigma", 0)

        # カーネルサイズのバリデーション
        if (not isinstance(kernel_size, (list, tuple)) or
            len(kernel_size) != 2 or
                not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernel_size)):
            raise ProcessorValidationError(
                "kernel_size must be specified as two positive odd integers. Example: [15, 15]")

        # シグマ値のバリデーション
        if not (isinstance(sigma, (int, float)) and sigma >= 0):
            raise ProcessorValidationError(
                "sigma must be a non-negative number")

        # kernel_sizeもsigmaも両方0は不可
        if sigma == 0 and all(k == 0 for k in kernel_size):
            raise ProcessorValidationError(
                "kernel_size and sigma cannot both be 0")
