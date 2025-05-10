"""画像処理に関する共通ユーティリティ関数を提供するモジュール."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    RGB/BGR画像をグレースケールに変換する.既にグレースケールの場合はそのまま返す.

    Args:
        image (np.ndarray): 入力画像（BGRまたはグレースケール）

    Returns:
        np.ndarray: グレースケール画像

    Raises:
        ValueError: サポートされていない画像形式の場合
    """
    if image.ndim == 2:
        # 既にグレースケール
        return image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        # カラー画像（3または4チャンネル）
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Unsupported image format: shape={image.shape}. "
            "Only 2D (grayscale) or 3D (BGR/BGRA) images are supported."
        )
