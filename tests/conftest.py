"""テスト共通 fixture を定義するモジュール."""

import numpy as np
import pytest


@pytest.fixture
def dummy_binary_image() -> np.ndarray:
    """100x100 の2値画像 (0/255, uint8) を返す."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255
    return image


@pytest.fixture
def dummy_binary_color_image() -> np.ndarray:
    """100x100 の2値カラー画像 (BGR, 全チャンネル同値, uint8) を返す."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 255
    return np.stack([gray, gray, gray], axis=-1)
