"""リサイズプロセッサーを提供するモジュール."""

from typing import Any, Dict

import cv2
import numpy as np

from .base import BaseProcessor
from .registry import register_processor
from .validators.resize import ResizeConfigValidator


@register_processor("resize")
class ResizeProcessor(BaseProcessor):
    """
    画像をリサイズするプロセッサー.

    アスペクト比の保持オプションを提供します.
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        リサイズプロセッサーを初期化します.

        Args:
            name (str): プロセッサー名
            config (Dict[str, Any]): リサイズパラメータ
                - width (int): リサイズ後の幅
                - height (int): リサイズ後の高さ
                - preserve_aspect_ratio (bool, optional): アスペクト比を保持するかどうか
                - aspect_ratio_mode (str, optional): アスペクト比保持モード ('width' or 'height')
        """
        super().__init__(name, config)
        self.width = config.get("width", None)
        self.height = config.get("height", None)
        self.preserve_aspect_ratio = config.get("preserve_aspect_ratio", False)
        self.aspect_ratio_mode = config.get("aspect_ratio_mode", "width")

        # パラメータのバリデーション
        validator = ResizeConfigValidator(config)
        validator.validate()

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像をリサイズします.

        Args:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: リサイズされた画像
        """
        # 入力画像のバリデーション
        validator = ResizeConfigValidator(
            {
                "width": self.width,
                "height": self.height,
                "preserve_aspect_ratio": self.preserve_aspect_ratio,
                "aspect_ratio_mode": self.aspect_ratio_mode,
            },
            image,
        )
        validator.validate()

        # アスペクト比を保持しない場合は単純にリサイズ
        if not self.preserve_aspect_ratio:
            target_size = (self.width or image.shape[1], self.height or image.shape[0])
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # アスペクト比を保持する場合の処理
        orig_h, orig_w = image.shape[:2]
        orig_aspect_ratio = orig_w / orig_h

        if self.aspect_ratio_mode == "width":
            # 幅を基準にアスペクト比を保持
            if self.width:
                new_w = self.width
                new_h = int(new_w / orig_aspect_ratio)
            else:
                new_h = self.height
                new_w = int(new_h * orig_aspect_ratio)
        else:  # height mode
            # 高さを基準にアスペクト比を保持
            if self.height:
                new_h = self.height
                new_w = int(new_h * orig_aspect_ratio)
            else:
                new_w = self.width
                new_h = int(new_w / orig_aspect_ratio)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
