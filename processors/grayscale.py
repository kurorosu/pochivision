import cv2
import numpy as np

from processors.base import BaseProcessor
from processors.registry import register_processor


@register_processor("grayscale")
class GrayscaleProcessor(BaseProcessor):
    """
    グレースケール変換を行う画像処理プロセッサ。

    このプロセッサは、カラー画像（BGR）をグレースケール画像に変換します。
    設定項目は特に不要で、変換のみを行います。

    登録名:
        "grayscale"

    設定例:
        {
            "grayscale": {}
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        グレースケール変換を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）

        Returns:
            np.ndarray: グレースケールに変換された画像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
