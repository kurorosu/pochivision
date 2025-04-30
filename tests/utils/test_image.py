"""utils.imageモジュールのユニットテスト."""

import numpy as np
import pytest

from utils.image import to_grayscale


def test_to_grayscale():
    """to_grayscale関数のテスト."""
    # グレースケール画像（変換なし）
    gray_img = np.zeros((10, 10), dtype=np.uint8)
    result = to_grayscale(gray_img)
    assert result.shape == (10, 10)
    assert result.ndim == 2
    assert np.array_equal(result, gray_img)

    # カラー画像（BGR）
    color_img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = to_grayscale(color_img)
    assert result.shape == (10, 10)
    assert result.ndim == 2

    # アルファチャンネル付きカラー画像（BGRA）
    color_alpha_img = np.zeros((10, 10, 4), dtype=np.uint8)
    result = to_grayscale(color_alpha_img)
    assert result.shape == (10, 10)
    assert result.ndim == 2

    # 無効な画像形式
    invalid_img = np.zeros((10, 10, 5), dtype=np.uint8)  # 5チャンネル
    with pytest.raises(ValueError):
        to_grayscale(invalid_img)
