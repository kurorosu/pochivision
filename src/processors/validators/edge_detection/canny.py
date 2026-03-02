"""Cannyエッジ検出プロセッサーの設定バリデーターを定義します."""

from typing import Any, Dict

import numpy as np  # numpy をインポート

# exceptions から ProcessorValidationError をインポート
from exceptions import ProcessorValidationError

from ..base import BaseValidator

# from ..edge_detection import CannyEdgeProcessor # 循環参照を避けるためにコメントアウト


class CannyEdgeValidator(BaseValidator):  # クラス名を CannyEdgeValidator に変更
    """CannyEdgeProcessor設定および入力画像のバリデーター."""

    def __init__(self, config: Dict[str, Any]):
        """
        CannyEdgeValidatorを初期化します.

        Args:
            config (Dict[str, Any]): 検証対象の設定.
        """
        self.config: Dict[str, Any] = config

    def validate_config(self) -> None:
        """
        Cannyエッジ検出設定を検証します.

        configにキーが存在する場合にのみ、その値を検証します.

        Raises:
            ProcessorValidationError: 設定が無効な場合.
        """
        if "threshold1" in self.config:
            threshold1 = self.config["threshold1"]
            if not isinstance(threshold1, (int, float)):
                raise ProcessorValidationError(
                    f"Canny 'threshold1' must be a number, got {threshold1}."
                )
            if threshold1 < 0:
                raise ProcessorValidationError(
                    f"Canny 'threshold1' must be non-negative. Got {threshold1}."
                )

        if "threshold2" in self.config:
            threshold2 = self.config["threshold2"]
            if not isinstance(threshold2, (int, float)):
                raise ProcessorValidationError(
                    f"Canny 'threshold2' must be a number, got {threshold2}."
                )
            if threshold2 < 0:
                raise ProcessorValidationError(
                    f"Canny 'threshold2' must be non-negative. Got {threshold2}."
                )

        if "threshold1" in self.config and "threshold2" in self.config:
            threshold1_val = self.config["threshold1"]
            threshold2_val = self.config["threshold2"]
            if (
                isinstance(threshold1_val, (int, float))
                and isinstance(threshold2_val, (int, float))
                and threshold1_val > threshold2_val
            ):
                raise ProcessorValidationError(
                    "Canny 'threshold1' should not be greater than 'threshold2'. "
                    f"Got threshold1={threshold1_val}, threshold2={threshold2_val}."
                )

        if "aperture_size" in self.config:
            aperture_size = self.config["aperture_size"]
            if not isinstance(aperture_size, int):
                raise ProcessorValidationError(
                    f"Canny 'aperture_size' must be an integer, got {aperture_size}."
                )
            if aperture_size not in [3, 5, 7]:
                raise ProcessorValidationError(
                    "Canny 'aperture_size' must be 3, 5, or 7. " f"Got {aperture_size}."
                )

        if "l2_gradient" in self.config:
            l2_gradient = self.config["l2_gradient"]
            if not isinstance(l2_gradient, bool):
                raise ProcessorValidationError(
                    f"Canny 'l2_gradient' must be a boolean, got {l2_gradient}."
                )

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
                "Input image for CannyEdgeProcessor must be 2D grayscale "
                "or 3-channel color image."
            )
        # uint8 への変換可能性のチェックは process メソッド内で行うため、ここでは不要
        # (cv2.Canny が内部で uint8 を要求するため)
