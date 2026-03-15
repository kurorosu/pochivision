"""テスト共通 fixture を定義するモジュール."""

import numpy as np
import pytest


@pytest.fixture
def dummy_color_image() -> np.ndarray:
    """100x100 のカラー画像 (BGR, uint8) を返す."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_grayscale_image() -> np.ndarray:
    """100x100 のグレースケール画像 (uint8) を返す."""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


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
