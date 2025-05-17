"""グレースケール変換プロセッサの実装を提供するモジュール."""

import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from utils.image import to_grayscale


@register_processor("grayscale")
class GrayscaleProcessor(BaseProcessor):
    """
    グレースケール変換を行う画像処理プロセッサ.

    このプロセッサは、カラー画像（BGR）をグレースケール画像に変換します.
    設定項目は特に不要で、変換のみを行います.

    登録名:
        "grayscale"

    設定例:
        {
            "grayscale": {}
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        グレースケール変換を実行します.

        Args:
            image (np.ndarray): 入力画像(BGRまたはグレースケール).

        Returns:
            np.ndarray: グレースケールに変換された画像.

        Raises:
            ProcessorRuntimeError: 画像変換に失敗した場合.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        try:
            return to_grayscale(image)
        except ValueError as e:
            raise ProcessorRuntimeError(f"Grayscale conversion failed: {e}")
