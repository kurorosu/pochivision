from typing import Any, Dict
from processors.validators.base import BaseValidator


class BilateralFilterValidator(BaseValidator):
    """
    バイラテラルフィルタ（Bilateral Filter）用のバリデータ。

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
        # d, sigmaColor, sigmaSpaceのバリデーション
        d = self.config.get("d", 9)
        sigmaColor = self.config.get("sigmaColor", 75)
        sigmaSpace = self.config.get("sigmaSpace", 75)
        if not (isinstance(d, int) and d > 0):
            raise ValueError("d must be a positive integer. Example: 9")
        if not (isinstance(sigmaColor, (int, float)) and sigmaColor > 0):
            raise ValueError(
                "sigmaColor must be a positive number. Example: 75")
        if not (isinstance(sigmaSpace, (int, float)) and sigmaSpace > 0):
            raise ValueError(
                "sigmaSpace must be a positive number. Example: 75")
