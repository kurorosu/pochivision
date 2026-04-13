"""Cannyエッジ検出プロセッサーの設定バリデーターを定義します."""

from typing import Any

import numpy as np  # numpy をインポート

# exceptions から ProcessorValidationError をインポート
from pochivision.exceptions import ProcessorValidationError

from ..base import BaseValidator

# from ..edge_detection import CannyEdgeProcessor # 循環参照を避けるためにコメントアウト


class CannyEdgeValidator(BaseValidator):  # クラス名を CannyEdgeValidator に変更
    """CannyEdgeProcessor設定および入力画像のバリデーター."""

    def __init__(self, config: dict[str, Any]):
        """
        CannyEdgeValidatorを初期化します.

        Args:
            config (dict[str, Any]): 検証対象の設定.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        self.config: dict[str, Any] = config
        self.validate_config(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        ``aperture_size`` は 3, 5, 7 のいずれか (奇数) ,
        ``threshold1`` および ``threshold2`` は 0 以上の数値で
        ``threshold1 <= threshold2`` でなければならない.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        if "aperture_size" in config and config["aperture_size"] is not None:
            aperture_size = config["aperture_size"]
            if isinstance(aperture_size, bool) or not isinstance(aperture_size, int):
                raise ProcessorValidationError(
                    f"aperture_size must be an int, got {aperture_size!r}"
                )
            if aperture_size < 3 or aperture_size > 7 or aperture_size % 2 == 0:
                raise ProcessorValidationError(
                    "aperture_size must be an odd integer in [3, 7], "
                    f"got {aperture_size}"
                )
        threshold1 = config.get("threshold1")
        threshold2 = config.get("threshold2")
        for name, value in (("threshold1", threshold1), ("threshold2", threshold2)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProcessorValidationError(
                    f"{name} must be a number, got {value!r}"
                )
            if value < 0:
                raise ProcessorValidationError(f"{name} must be >= 0, got {value}")
        if (
            threshold1 is not None
            and threshold2 is not None
            and not isinstance(threshold1, bool)
            and not isinstance(threshold2, bool)
            and isinstance(threshold1, (int, float))
            and isinstance(threshold2, (int, float))
            and threshold1 > threshold2
        ):
            raise ProcessorValidationError(
                f"threshold1 ({threshold1}) must be <= threshold2 ({threshold2})"
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
