from typing import Any, Dict
from processors.validators.base import BaseValidator


class MotionBlurValidator(BaseValidator):
    """
    モーションブラー（Motion Blur）用のバリデータ。

    Args:
        config (dict): バリデーション対象の設定辞書
        image (np.ndarray): 入力画像

    Raises:
        ValueError: 不正なパラメータや画像が検出された場合
    """

    def __init__(self, config: Dict[str, Any], image: Any) -> None:
        self.config = config
        self.image = image

    def validate(self) -> None:
        """
        設定値と画像のバリデーションを実行する。

        Raises:
            ValueError: 不正なパラメータや画像が検出された場合
        """
        # 共通バリデーション
        self.validate_image_type_and_nonempty(self.image)
        # kernel_size: 正の奇数整数
        kernel_size = self.config.get("kernel_size", 15)
        if not (isinstance(kernel_size, int) and kernel_size > 0 and kernel_size % 2 == 1):
            raise ValueError(
                "kernel_size must be a positive odd integer. Example: 15")
        # angle: 0-359の整数
        angle = self.config.get("angle", 0)
        if not (isinstance(angle, int) and 0 <= angle < 360):
            raise ValueError(
                "angle must be an integer between 0 and 359. Example: 0")
