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
        ResizeProcessorを初期化.

        Args:
            name (str): プロセッサー名
            config (Dict[str, Any]): リサイズパラメータ
                - width (int): リサイズ後の幅
                - height (int): リサイズ後の高さ
                - preserve_aspect_ratio (bool, optional): アスペクト比を保持するかどうか
                - aspect_ratio_mode (str, optional): アスペクト比保持モード ('width' or 'height')
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = ResizeConfigValidator(config)
        self.validator.validate_config()

        # デフォルト設定を取得してDRY原則に従う
        default_config = self.get_default_config()
        self.width = config.get("width", default_config["width"])
        self.height = config.get("height", default_config["height"])
        self.preserve_aspect_ratio = config.get(
            "preserve_aspect_ratio", default_config["preserve_aspect_ratio"]
        )
        self.aspect_ratio_mode = config.get(
            "aspect_ratio_mode", default_config["aspect_ratio_mode"]
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像をリサイズします.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: リサイズされた画像.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        # 元の画像サイズを取得
        h, w = image.shape[:2]

        # リサイズ後のサイズを計算
        target_w, target_h = self._calculate_target_size(w, h)

        # リサイズ処理
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _calculate_target_size(self, orig_width: int, orig_height: int) -> tuple:
        """
        リサイズ後のサイズを計算.

        Args:
            orig_width (int): 元の画像の幅.
            orig_height (int): 元の画像の高さ.

        Returns:
            tuple: (target_width, target_height)
        """
        if self.width is None and self.height is None:
            return orig_width, orig_height

        if not self.preserve_aspect_ratio:
            return (
                self.width if self.width is not None else orig_width,
                self.height if self.height is not None else orig_height,
            )

        # アスペクト比を保持する場合
        aspect_ratio = orig_width / orig_height

        if self.aspect_ratio_mode == "width" and self.width is not None:
            target_w = self.width
            target_h = int(target_w / aspect_ratio)
        elif self.aspect_ratio_mode == "height" and self.height is not None:
            target_h = self.height
            target_w = int(target_h * aspect_ratio)
        else:
            # どちらも指定されていない場合は元のサイズを使用
            target_w = self.width if self.width is not None else orig_width
            target_h = self.height if self.height is not None else orig_height

        return target_w, target_h

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        リサイズプロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {
            "width": 1600,
            "height": 1200,
            "preserve_aspect_ratio": True,
            "aspect_ratio_mode": "width",
        }
