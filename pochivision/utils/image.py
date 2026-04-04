"""画像処理に関する共通ユーティリティ関数を提供するモジュール."""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def load_image(image_path: Path) -> np.ndarray | None:
    """画像ファイルを読み込む.

    Args:
        image_path: 画像ファイルのパス.

    Returns:
        読み込んだ画像. 失敗時は None.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image


def get_image_files(
    directory: Path,
    extensions: list[str] | None = None,
    case_sensitive: bool = False,
) -> list[Path]:
    """ディレクトリから画像ファイルのパスリストを取得する.

    Args:
        directory: 検索対象ディレクトリ.
        extensions: 対象拡張子のリスト. None の場合はデフォルト拡張子を使用.
        case_sensitive: True の場合, 拡張子の大文字小文字を区別する.

    Returns:
        画像ファイルのパスリスト (ソート済み).
    """
    if extensions is None:
        extensions = DEFAULT_IMAGE_EXTENSIONS

    image_files: list[Path] = []
    for ext in extensions:
        if case_sensitive:
            image_files.extend(directory.glob(f"*{ext}"))
        else:
            image_files.extend(directory.glob(f"*{ext.lower()}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(set(image_files))


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
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # 3次元1チャンネル → 2次元グレースケール
            return image.squeeze(axis=2)
        elif image.shape[2] == 3:
            # BGR画像
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            # BGRA画像
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                f"Unsupported channel count: {image.shape[2]}. "
                "Only 1, 3, or 4 channels are supported."
            )
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
