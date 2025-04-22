import cv2
import numpy as np

from processors import BaseProcessor
from processors.registry import register_processor
from processors.validators.blur.gaussian import GaussianBlurConfigValidator
from processors.validators.blur.average import AverageBlurValidator
from processors.validators.blur.median import MedianBlurValidator
from processors.validators.blur.bilateral import BilateralFilterValidator
from processors.validators.blur.motion import MotionBlurValidator


@register_processor("gaussian_blur")
class GaussianBlurProcessor(BaseProcessor):
    """
    ガウシアンぼかし（Gaussian Blur）を適用する画像処理プロセッサ。

    このプロセッサは、入力画像に対してガウシアンフィルタを用いた
    ぼかし処理（Gaussian Blur）を実行します。設定ファイルでカーネルサイズやシグマ値を
    指定することができます。

    登録名:
        "gaussian_blur"

    設定例:
        {
            "kernel_size": [15, 15],
            "sigma": 0
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ガウシアンぼかし処理（Gaussian Blur）を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）

        Returns:
            np.ndarray: ガウシアンぼかしを適用した画像
        """
        GaussianBlurConfigValidator(self.config, image).validate()
        kernel_size = tuple(self.config.get("kernel_size", [15, 15]))
        sigma = self.config.get("sigma", 0)
        return cv2.GaussianBlur(image, kernel_size, sigma)


@register_processor("average_blur")
class AverageBlurProcessor(BaseProcessor):
    """
    平均値ブラー（Average Blur）を適用する画像処理プロセッサ。

    入力画像に対してカーネルサイズで指定した範囲の平均値でぼかし処理を行います。

    登録名:
        "average_blur"

    設定例:
        {
            "kernel_size": [5, 5]
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        平均値ブラー処理（cv2.blur）を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）

        Returns:
            np.ndarray: 平均値ブラーを適用した画像
        """
        AverageBlurValidator(self.config, image).validate()
        kernel_size = tuple(self.config.get("kernel_size", [5, 5]))
        return cv2.blur(image, kernel_size)


@register_processor("median_blur")
class MedianBlurProcessor(BaseProcessor):
    """
    メディアンブラー（Median Blur）を適用する画像処理プロセッサ。

    入力画像に対してカーネルサイズで指定した範囲の中央値でぼかし処理を行います。
    塩胡椒ノイズ除去に有効です。

    登録名:
        "median_blur"

    設定例:
        {
            "kernel_size": 5
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        メディアンブラー処理（cv2.medianBlur）を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式またはグレースケール）

        Returns:
            np.ndarray: メディアンブラーを適用した画像
        """
        MedianBlurValidator(self.config, image).validate()
        kernel_size = self.config.get("kernel_size", 5)
        return cv2.medianBlur(image, kernel_size)


@register_processor("bilateral_filter")
class BilateralFilterProcessor(BaseProcessor):
    """
    バイラテラルフィルタ（Bilateral Filter）を適用する画像処理プロセッサ。

    エッジを保ちながらぼかし処理を行います。
    d, sigmaColor, sigmaSpaceの3つのパラメータで調整可能です。

    登録名:
        "bilateral_filter"

    設定例:
        {
            "d": 9,
            "sigmaColor": 75,
            "sigmaSpace": 75
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        バイラテラルフィルタ処理（cv2.bilateralFilter）を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式またはグレースケール）

        Returns:
            np.ndarray: バイラテラルフィルタを適用した画像
        """
        BilateralFilterValidator(self.config, image).validate()
        d = self.config.get("d", 9)
        sigmaColor = self.config.get("sigmaColor", 75)
        sigmaSpace = self.config.get("sigmaSpace", 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


@register_processor("motion_blur")
class MotionBlurProcessor(BaseProcessor):
    """
    モーションブラー（Motion Blur）を適用する画像処理プロセッサ。

    指定した長さと角度で直線的な動きのブラーを適用します。

    登録名:
        "motion_blur"

    設定例:
        {
            "kernel_size": 15,
            "angle": 0
        }
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        モーションブラー処理（cv2.filter2D）を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式またはグレースケール）

        Returns:
            np.ndarray: モーションブラーを適用した画像
        """
        MotionBlurValidator(self.config, image).validate()
        kernel_size = self.config.get("kernel_size", 15)
        angle = self.config.get("angle", 0)
        # カーネル生成
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = np.deg2rad(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        for i in range(kernel_size):
            x = int(center + (i - center) * cos_a)
            y = int(center + (i - center) * sin_a)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        kernel /= np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)
