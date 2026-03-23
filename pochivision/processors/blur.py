"""各種ブラー（ぼかし）処理プロセッサの実装を提供するモジュール."""

from typing import Any, Dict

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.validators.blur.average import AverageBlurValidator
from pochivision.processors.validators.blur.bilateral import BilateralFilterValidator
from pochivision.processors.validators.blur.gaussian import GaussianBlurValidator
from pochivision.processors.validators.blur.median import MedianBlurValidator
from pochivision.processors.validators.blur.motion import MotionBlurValidator


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        GaussianBlurProcessorを初期化.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = GaussianBlurValidator(config)
        self.validator.validate_config()

        # バリデータから検証済みの値を取得
        self.kernel_size = (self.validator.kernel_width, self.validator.kernel_height)
        self.sigma_x = self.validator.sigma_x
        self.sigma_y = self.validator.sigma_y

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ガウシアンぼかし処理（Gaussian Blur）を実行します.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            np.ndarray: ガウシアンぼかしを適用した画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: OpenCV処理中にエラーが発生した場合.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        # ガウシアンブラー処理
        try:
            return cv2.GaussianBlur(
                image, self.kernel_size, self.sigma_x, sigmaY=self.sigma_y
            )
        except cv2.error as e:
            error_msg = f"Error during GaussianBlur processing: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        ガウシアンブラープロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"kernel_size": [15, 15], "sigma": 0}


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        AverageBlurProcessorを初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = AverageBlurValidator(config)
        self.validator.validate_config()

        # バリデータから検証済みの値を取得
        self.kernel_size = (self.validator.kernel_width, self.validator.kernel_height)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        平均値ブラー処理（cv2.blur）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式).

        Returns:
            np.ndarray: 平均値ブラーを適用した画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: OpenCV処理中にエラーが発生した場合.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        # 平均値ブラー処理
        try:
            return cv2.blur(image, self.kernel_size)
        except cv2.error as e:
            error_msg = f"Error during Average Blur processing: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        平均値ブラープロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"kernel_size": [5, 5]}


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

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        MedianBlurProcessorを初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = MedianBlurValidator(config)
        self.validator.validate_config()

        # バリデータから検証済みの値を取得
        self.kernel_size = self.validator.kernel_size

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        メディアンブラー処理（cv2.medianBlur）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: メディアンブラーを適用した画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: OpenCV処理中にエラーが発生した場合.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        # メディアンブラー処理
        try:
            return cv2.medianBlur(image, self.kernel_size)
        except cv2.error as e:
            error_msg = f"Error during Median Blur processing: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        メディアンブラープロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"kernel_size": 5}


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
        """
        BilateralFilterProcessorを初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = BilateralFilterValidator(config)
        self.validator.validate_config()

        # バリデータから検証済みの値を取得
        self.d = self.validator.d
        self.sigma_color = self.validator.sigma_color
        self.sigma_space = self.validator.sigma_space

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        バイラテラルフィルタ処理（cv2.bilateralFilter）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: バイラテラルフィルタを適用した画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: OpenCV処理中にエラーが発生した場合.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        # バイラテラルフィルタ処理
        try:
            # パラメータを確実に利用可能にする（バリデーションが成功していれば通常はNoneではない）
            d = self.d
            sigma_color = self.sigma_color
            sigma_space = self.sigma_space

            if d is None or sigma_color is None or sigma_space is None:
                error_msg = (
                    f"Bilateral filter parameters were not properly validated: d={d}, "
                    f"sigma_color={sigma_color}, sigma_space={sigma_space}"
                )
                raise ProcessorRuntimeError(error_msg)

            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        except cv2.error as e:
            error_msg = f"Error during Bilateral Filter processing: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        バイラテラルフィルタプロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}


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
        MotionBlurProcessorを初期化します.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        # パラメータのバリデーション
        self.validator = MotionBlurValidator(config)
        self.validator.validate_config()

        # バリデータから検証済みの値を取得
        self.kernel_size = self.validator.kernel_size
        self.angle = self.validator.angle

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        モーションブラー処理（cv2.filter2D）を実行します.

        Args:
            image (np.ndarray): 入力画像(BGR形式またはグレースケール).

        Returns:
            np.ndarray: モーションブラーを適用した画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: OpenCV処理中にエラーが発生した場合.
        """
        # 入力画像のバリデーション
        self.validator.validate_image(image)

        try:
            # パラメータを確実に利用可能にする（バリデーションが成功していれば通常はNoneではない）
            kernel_size = self.kernel_size
            angle = self.angle

            if kernel_size is None or angle is None:
                error_msg = (
                    f"Motion blur parameters were not properly "
                    f"validated: kernel_size={kernel_size}, angle={angle}"
                )
                raise ProcessorRuntimeError(error_msg)

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
            if np.sum(kernel) == 0:  # 全要素が0の場合、ゼロ除算を避ける
                # 例えば、kernel_sizeが非常に小さい場合や特定の角度で発生しうる
                # この場合、元画像をそのまま返すか、エラーとするか、あるいは小さな単位行列カーネルを適用するか
                # ここでは元画像を返すことにする
                return image
            kernel /= np.sum(kernel)
            return cv2.filter2D(image, -1, kernel)
        except cv2.error as e:
            error_msg = f"Error during Motion Blur processing: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        モーションブラープロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"kernel_size": 15, "angle": 0}
