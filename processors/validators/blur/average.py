from typing import Any, Dict
from processors.validators.base import BaseValidator


class AverageBlurValidator(BaseValidator):
    """
    平均値ブラー（Average Blur）用のバリデータ。

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
        # カーネルサイズのバリデーション
        kernel_size = self.config.get("kernel_size", [5, 5])
        if (not isinstance(kernel_size, (list, tuple)) or
            len(kernel_size) != 2 or
                not all(isinstance(k, int) and k > 0 for k in kernel_size)):
            raise ValueError(
                "kernel_size must be specified as two positive integers. Example: [5, 5]")
