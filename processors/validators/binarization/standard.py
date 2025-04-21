from typing import Any, Dict
from processors.validators.base import BaseValidator
import numpy as np


class StandardBinarizationValidator(BaseValidator):
    """
    スタンダードな2値化（しきい値による通常の2値化）用のバリデータ。

    Args:
        config (dict): バリデーション対象の設定辞書
        image (np.ndarray): 入力画像

    Raises:
        ValueError: 不正なパラメータや画像が検出された場合
    """

    def __init__(self, config: Dict[str, Any], image: Any) -> None:
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する。

        Raises:
            ValueError: 不正なパラメータや画像が検出された場合
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)
        # 2Dまたは3/4チャンネル画像のみ許可
        if not ((self.image.ndim == 2) or (self.image.ndim == 3 and self.image.shape[2] in (3, 4))):
            raise ValueError(
                "Input image must be 2D grayscale or 3/4 channel color image (BGR/BGRA).")
        # threshold値のバリデーション
        threshold = self.config.get("threshold", 128)
        if not (isinstance(threshold, int) and 0 <= threshold <= 255):
            raise ValueError("threshold must be an integer between 0 and 255.")
