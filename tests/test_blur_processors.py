import numpy as np
import pytest  # noqa: F401

from processors.blur import (
    AverageBlurProcessor,
    BilateralFilterProcessor,
    GaussianBlurProcessor,
    MedianBlurProcessor,
    MotionBlurProcessor,
)

# テスト用の画像データ
DUMMY_IMAGE = np.ones((10, 10, 3), dtype=np.uint8) * 100

# GaussianBlur


def test_gaussian_blur_valid():
    """ガウシアンブラーの基本機能をテスト"""
    config = {"kernel_size": [3, 3], "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# AverageBlur


def test_average_blur_valid():
    """平均値ブラーの基本機能をテスト"""
    config = {"kernel_size": [3, 3]}
    processor = AverageBlurProcessor(name="average_blur", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# MedianBlur


def test_median_blur_valid():
    """メディアンブラーの基本機能をテスト"""
    config = {"kernel_size": 3}  # 奇数を指定
    processor = MedianBlurProcessor(name="median_blur", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# BilateralFilter


def test_bilateral_filter_valid():
    """バイラテラルフィルタの基本機能をテスト"""
    config = {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}
    processor = BilateralFilterProcessor(name="bilateral_filter", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# MotionBlur


def test_motion_blur_valid():
    """モーションブラーの基本機能をテスト"""
    config = {"kernel_size": 5, "angle": 45}
    processor = MotionBlurProcessor(name="motion_blur", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8
