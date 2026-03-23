"""utils.imageモジュールのユニットテスト."""

import numpy as np
import pytest

from pochivision.utils.image import to_bgr, to_grayscale, to_rgb


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


def test_to_bgr():
    """to_bgr関数のテスト."""
    # 2次元グレースケール画像
    gray_2d = np.full((10, 10), 128, dtype=np.uint8)
    result = to_bgr(gray_2d)
    assert result.shape == (10, 10, 3)
    assert result.ndim == 3
    # グレースケールなのでB=G=R=128になるはず
    assert np.all(result[:, :, 0] == 128)  # B
    assert np.all(result[:, :, 1] == 128)  # G
    assert np.all(result[:, :, 2] == 128)  # R

    # 3次元1チャンネル画像
    gray_3d_1ch = np.full((10, 10, 1), 64, dtype=np.uint8)
    result = to_bgr(gray_3d_1ch)
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 64)  # B
    assert np.all(result[:, :, 1] == 64)  # G
    assert np.all(result[:, :, 2] == 64)  # R

    # 既にBGR画像（変換なし）
    bgr_img = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_img[:, :, 0] = 255  # B
    bgr_img[:, :, 1] = 128  # G
    bgr_img[:, :, 2] = 64  # R
    result = to_bgr(bgr_img)
    assert result.shape == (10, 10, 3)
    assert np.array_equal(result, bgr_img)

    # BGRA画像（アルファチャンネル除去）
    bgra_img = np.zeros((10, 10, 4), dtype=np.uint8)
    bgra_img[:, :, 0] = 255  # B
    bgra_img[:, :, 1] = 128  # G
    bgra_img[:, :, 2] = 64  # R
    bgra_img[:, :, 3] = 255  # A
    result = to_bgr(bgra_img)
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 255)  # B
    assert np.all(result[:, :, 1] == 128)  # G
    assert np.all(result[:, :, 2] == 64)  # R

    # 無効な画像形式
    invalid_img = np.zeros((10, 10, 5), dtype=np.uint8)  # 5チャンネル
    with pytest.raises(ValueError):
        to_bgr(invalid_img)


def test_to_rgb():
    """to_rgb関数のテスト."""
    # 2次元グレースケール画像
    gray_2d = np.full((10, 10), 128, dtype=np.uint8)
    result = to_rgb(gray_2d)
    assert result.shape == (10, 10, 3)
    assert result.ndim == 3
    # グレースケールなのでR=G=B=128になるはず
    assert np.all(result[:, :, 0] == 128)  # R
    assert np.all(result[:, :, 1] == 128)  # G
    assert np.all(result[:, :, 2] == 128)  # B

    # 3次元1チャンネル画像
    gray_3d_1ch = np.full((10, 10, 1), 64, dtype=np.uint8)
    result = to_rgb(gray_3d_1ch)
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 64)  # R
    assert np.all(result[:, :, 1] == 64)  # G
    assert np.all(result[:, :, 2] == 64)  # B

    # BGR画像（BGR→RGB変換）
    bgr_img = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_img[:, :, 0] = 255  # B
    bgr_img[:, :, 1] = 128  # G
    bgr_img[:, :, 2] = 64  # R
    result = to_rgb(bgr_img)
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 64)  # R (元のR)
    assert np.all(result[:, :, 1] == 128)  # G (元のG)
    assert np.all(result[:, :, 2] == 255)  # B (元のB)

    # BGRA画像（アルファチャンネル除去 + BGR→RGB変換）
    bgra_img = np.zeros((10, 10, 4), dtype=np.uint8)
    bgra_img[:, :, 0] = 255  # B
    bgra_img[:, :, 1] = 128  # G
    bgra_img[:, :, 2] = 64  # R
    bgra_img[:, :, 3] = 255  # A
    result = to_rgb(bgra_img)
    assert result.shape == (10, 10, 3)
    assert np.all(result[:, :, 0] == 64)  # R (元のR)
    assert np.all(result[:, :, 1] == 128)  # G (元のG)
    assert np.all(result[:, :, 2] == 255)  # B (元のB)

    # 無効な画像形式
    invalid_img = np.zeros((10, 10, 5), dtype=np.uint8)  # 5チャンネル
    with pytest.raises(ValueError):
        to_rgb(invalid_img)


def test_grayscale_consistency():
    """グレースケール変換の一貫性テスト."""
    # 同じグレースケール値で異なる形状の画像を作成
    gray_value = 100

    # 2次元グレースケール
    gray_2d = np.full((20, 20), gray_value, dtype=np.uint8)

    # 3次元1チャンネル
    gray_3d_1ch = np.full((20, 20, 1), gray_value, dtype=np.uint8)

    # 3次元3チャンネル（グレースケール）
    gray_3d_3ch = np.full((20, 20, 3), gray_value, dtype=np.uint8)

    # すべてをBGRに変換
    bgr_from_2d = to_bgr(gray_2d)
    bgr_from_3d_1ch = to_bgr(gray_3d_1ch)
    bgr_from_3d_3ch = to_bgr(gray_3d_3ch)

    # すべて同じ結果になるはず
    assert np.array_equal(bgr_from_2d, bgr_from_3d_1ch)
    assert np.array_equal(bgr_from_3d_1ch, bgr_from_3d_3ch)

    # すべてをRGBに変換
    rgb_from_2d = to_rgb(gray_2d)
    rgb_from_3d_1ch = to_rgb(gray_3d_1ch)
    rgb_from_3d_3ch = to_rgb(gray_3d_3ch)

    # すべて同じ結果になるはず
    assert np.array_equal(rgb_from_2d, rgb_from_3d_1ch)
    assert np.array_equal(rgb_from_3d_1ch, rgb_from_3d_3ch)

    # RGB値がすべて元のグレースケール値と同じになるはず
    assert np.all(rgb_from_2d[:, :, 0] == gray_value)  # R
    assert np.all(rgb_from_2d[:, :, 1] == gray_value)  # G
    assert np.all(rgb_from_2d[:, :, 2] == gray_value)  # B
