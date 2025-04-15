import cv2
import numpy as np

from processors.base import BaseProcessor


class BlurProcessor(BaseProcessor):
    """
    ぼかし画像処理クラス。
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像にガウシアンぼかしを適用します。

        Parameters:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: ぼかし処理後の画像
        """
        kernel_size = self.config.get("kernel_size", (15, 15))
        sigma = self.config.get("sigma", 0)
        return cv2.GaussianBlur(image, kernel_size, sigma)
