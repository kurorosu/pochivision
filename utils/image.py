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


def to_bgr(image: np.ndarray) -> np.ndarray:
    """
    任意の形状の画像をBGR形式に変換する.

    Args:
        image (np.ndarray): 入力画像（グレースケールまたはBGR）

    Returns:
        np.ndarray: BGR形式の画像（3チャンネル）

    Raises:
        ValueError: サポートされていない画像形式の場合
    """
    if image.ndim == 2:
        # 2次元グレースケール → BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # 3次元1チャンネル → 2次元 → BGR
            return cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            # 既にBGR（そのまま返す）
            return image
        elif image.shape[2] == 4:
            # BGRA → BGR
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                f"Unsupported channel count: {image.shape[2]}. "
                "Only 1, 3, or 4 channels are supported."
            )
    else:
        raise ValueError(
            f"Unsupported image format: shape={image.shape}. "
            "Only 2D (grayscale) or 3D images are supported."
        )


def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    任意の形状の画像をRGB形式に変換する.

    Args:
        image (np.ndarray): 入力画像（グレースケールまたはBGR）

    Returns:
        np.ndarray: RGB形式の画像（3チャンネル）

    Raises:
        ValueError: サポートされていない画像形式の場合
    """
    # まずBGRに変換
    bgr_image = to_bgr(image)

    # BGR → RGB変換
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
