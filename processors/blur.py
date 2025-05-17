"""各種ブラー（ぼかし）処理プロセッサの実装を提供するモジュール."""

from typing import Any, Dict

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from processors.validators.blur.average import AverageBlurValidator
from processors.validators.blur.bilateral import BilateralFilterValidator
from processors.validators.blur.gaussian import GaussianBlurConfigValidator
from processors.validators.blur.median import MedianBlurValidator
from processors.validators.blur.motion import MotionBlurValidator


@register_processor("gaussian_blur")
class GaussianBlurProcessor(BaseProcessor):
    """
    ガウシアンぼかし（Gaussian Blur）を適用する画像処理プロセッサ.

    このプロセッサは、入力画像に対してガウシアンフィルタを用いた
    ぼかし処理（Gaussian Blur）を実行します. 設定ファイルでカーネルサイズやシグマ値を
    指定することができます.

    登録名:
        "gaussian_blur"

    設定例:
        {
            "kernel_size": [15, 15],
            "sigma": 0
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        GaussianBlurProcessorを初期化.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = GaussianBlurConfigValidator(self.config)
        self.validator.validate()
        self.kernel_size = tuple(self.config.get("kernel_size", [15, 15]))
        self.sigma = self.config.get("sigma", 0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ガウシアンぼかし処理（Gaussian Blur）を実行します.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            np.ndarray: ガウシアンぼかしを適用した画像.
        """
        # 画像自体のバリデーション（必要であれば）
        # self.validator.validate_image_type_and_nonempty(
        # image)
        # BaseValidatorにメソッドがある場合
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )

        return cv2.GaussianBlur(image, self.kernel_size, self.sigma)


@register_processor("average_blur")
class AverageBlurProcessor(BaseProcessor):
    """
    平均値ブラー（Average Blur）を適用する画像処理プロセッサ.

    入力画像に対してカーネルサイズで指定した範囲の平均値でぼかし処理を行います.
    設定ファイルでカーネルサイズを指定することができます.

    登録名:
        "average_blur"

    設定例:
        {
            "kernel_size": [5, 5]
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Averageblurprocessor を初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = AverageBlurValidator(self.config)
        self.validator.validate()
        self.kernel_size = tuple(self.config.get("kernel_size", [5, 5]))

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        平均値ブラー処理（cv2.blur）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式).

        Returns:
            np.ndarray: 平均値ブラーを適用した画像.
        """
        # 画像自体のバリデーション（必要であれば）
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        return cv2.blur(image, self.kernel_size)


@register_processor("median_blur")
class MedianBlurProcessor(BaseProcessor):
    """
    メディアンブラー（Median Blur）を適用する画像処理プロセッサ.

    入力画像に対してカーネルサイズで指定した範囲の中央値でぼかし処理を行います.
    塩胡椒ノイズ除去に有効です.
    設定ファイルでカーネルサイズを指定することができます.

    登録名:
        "median_blur"

    設定例:
        {
            "kernel_size": 5
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Medianblurprocessor を初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = MedianBlurValidator(self.config)
        self.validator.validate()
        self.kernel_size = self.config.get("kernel_size", 5)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        メディアンブラー処理（cv2.medianBlur）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: メディアンブラーを適用した画像.
        """
        # 画像自体のバリデーション（必要であれば）
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        return cv2.medianBlur(image, self.kernel_size)


@register_processor("bilateral_filter")
class BilateralFilterProcessor(BaseProcessor):
    """
    バイラテラルフィルタ（Bilateral Filter）を適用する画像処理プロセッサ.

    エッジを保ちながらぼかし処理を行います.
    d, sigmaColor, sigmaSpaceの3つのパラメータで調整可能です.

    登録名:
        "bilateral_filter"

    設定例:
        {
            "d": 9,
            "sigmaColor": 75,
            "sigmaSpace": 75
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Bilateralfilterprocessor を初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = BilateralFilterValidator(self.config)
        self.validator.validate()
        self.d = self.config.get("d", 9)
        self.sigmaColor = self.config.get("sigmaColor", 75)
        self.sigmaSpace = self.config.get("sigmaSpace", 75)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        バイラテラルフィルタ処理（cv2.bilateralFilter）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: バイラテラルフィルタを適用した画像.
        """
        # 画像自体のバリデーション（必要であれば）
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )
        return cv2.bilateralFilter(image, self.d, self.sigmaColor, self.sigmaSpace)


@register_processor("motion_blur")
class MotionBlurProcessor(BaseProcessor):
    """
    モーションブラー（Motion Blur）を適用する画像処理プロセッサ.

    指定した長さと角度で直線的な動きのブラーを適用します.

    登録名:
        "motion_blur"

    設定例:
        {
            "kernel_size": 15,
            "angle": 0
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Motionblurprocessor を初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = MotionBlurValidator(self.config)
        self.validator.validate()
        self.kernel_size = self.config.get("kernel_size", 15)
        self.angle = self.config.get("angle", 0)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        モーションブラー処理（cv2.filter2D）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: モーションブラーを適用した画像.
        """
        # 画像自体のバリデーション（必要であれば）
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ProcessorRuntimeError(
                "Input image must be a non-empty NumPy ndarray."
            )

        # カーネル生成
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        center = self.kernel_size // 2
        rad = np.deg2rad(self.angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        for i in range(self.kernel_size):
            x = int(center + (i - center) * cos_a)
            y = int(center + (i - center) * sin_a)
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1
        if np.sum(kernel) == 0:  # 全要素が0の場合、ゼロ除算を避ける
            # 例えば、kernel_sizeが非常に小さい場合や特定の角度で発生しうる
            # この場合、元画像をそのまま返すか、エラーとするか、あるいは小さな単位行列カーネルを適用するか
            # ここでは元画像を返すことにする
            return image
        kernel /= np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)
