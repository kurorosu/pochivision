import cv2

from processors.base import BaseProcessor
from processors.registry import register_processor


@register_processor("blur")
class BlurProcessor(BaseProcessor):
    """
    ガウシアンぼかしを適用する画像処理プロセッサ。

    このプロセッサは、入力画像に対してガウシアンフィルタを用いた
    ぼかし処理を実行します。設定ファイルでカーネルサイズやシグマ値を
    指定することができます。

    登録名:
        "blur"

    設定例:
        {
            "kernel_size": [15, 15],
            "sigma": 0
        }
    """

    def process(self, image):
        """
        ガウシアンぼかし処理を実行します。

        Args:
            image (np.ndarray): 入力画像（BGR形式）

        Returns:
            np.ndarray: ガウシアンぼかしを適用した画像
        """
        kernel_size = tuple(self.config.get("kernel_size", [15, 15]))
        sigma = self.config.get("sigma", 0)
        return cv2.GaussianBlur(image, kernel_size, sigma)
