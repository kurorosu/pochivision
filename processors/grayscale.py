import cv2
import numpy as np
from processors.base import BaseProcessor


class GrayscaleProcessor(BaseProcessor):
    """
    グレースケール画像処理クラス。
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        グレースケール変換を実行します。

        Parameters:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: グレースケール処理後の画像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
