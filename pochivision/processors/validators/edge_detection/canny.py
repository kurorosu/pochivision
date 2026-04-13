"""Cannyエッジ検出プロセッサーの設定バリデーターを定義します."""

from typing import Any

import numpy as np  # numpy をインポート

# exceptions から ProcessorValidationError をインポート
from pochivision.exceptions import ProcessorValidationError

from ..base import BaseValidator

# from ..edge_detection import CannyEdgeProcessor # 循環参照を避けるためにコメントアウト


class CannyEdgeValidator(BaseValidator):  # クラス名を CannyEdgeValidator に変更
    """CannyEdgeProcessor設定および入力画像のバリデーター."""

    processor_name = "canny_edge"

    def __init__(self, config: dict[str, Any]):
        """
        CannyEdgeValidatorを初期化します.

        Args:
            config (dict[str, Any]): 検証対象の設定.
        """
        self.config: dict[str, Any] = config

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証します.

        Cannyエッジ検出はグレースケール画像または3チャンネルのカラー画像を期待します.
        また、入力はnp.ndarray型で空でないことを基本バリデーションで確認します.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        self.validate_image_type_and_nonempty(image)  # 基本的な型と空のチェック

        if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
            raise ProcessorValidationError(
                self._format_error(
                    "Input image for CannyEdgeProcessor must be 2D grayscale "
                    "or 3-channel color image, "
                    f"got ndim={image.ndim} (shape={image.shape})"
                )
            )
        # uint8 への変換可能性のチェックは process メソッド内で行うため、ここでは不要
        # (cv2.Canny が内部で uint8 を要求するため)
