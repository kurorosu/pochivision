"""ブラープロセッサのテストモジュール."""

import re

import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.blur import (
    AverageBlurProcessor,
    BilateralFilterProcessor,
    GaussianBlurProcessor,
    MedianBlurProcessor,
    MotionBlurProcessor,
)

# テスト用の画像データ
# 3チャンネルuint8画像 (BGR)
DUMMY_IMAGE_UINT8_3CH = np.ones((10, 10, 3), dtype=np.uint8) * 100
# 1チャンネルuint8画像 (グレースケール)
DUMMY_IMAGE_UINT8_1CH = np.ones((10, 10), dtype=np.uint8) * 100
# 不正な型のテスト用 (float32)
DUMMY_IMAGE_FLOAT32_3CH = np.ones((10, 10, 3), dtype=np.float32) * 100.0
# 不正なチャンネル数のテスト用
DUMMY_IMAGE_UINT8_2CH = np.ones((10, 10, 2), dtype=np.uint8) * 100
DUMMY_IMAGE_UINT8_4CH = np.ones((10, 10, 4), dtype=np.uint8) * 100
# 不正な次元数のテスト用
DUMMY_IMAGE_UINT8_4D = np.ones((10, 10, 10, 3), dtype=np.uint8) * 100


# GaussianBlur
def test_gaussian_blur_valid():
    """ガウシアンブラーの基本機能をテスト."""
    config = {"kernel_size": [3, 3], "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur", config=config)

    # 3チャンネル画像
    result_3ch = processor.process(DUMMY_IMAGE_UINT8_3CH.copy())
    assert result_3ch.shape == DUMMY_IMAGE_UINT8_3CH.shape
    assert result_3ch.dtype == np.uint8

    # 1チャンネル（グレースケール）画像
    result_1ch = processor.process(DUMMY_IMAGE_UINT8_1CH.copy())
    assert result_1ch.shape == DUMMY_IMAGE_UINT8_1CH.shape
    assert result_1ch.dtype == np.uint8


@pytest.mark.parametrize(
    "invalid_config, error_message_part",
    [
        (
            {"kernel_size": 3, "sigma": 1},
            "kernel_size must be specified as two positive odd",
        ),
        (
            {"kernel_size": [2, 3], "sigma": 1},
            "kernel_size width must be a positive odd integer",
        ),
        (
            {"kernel_size": [3, 2], "sigma": 1},
            "kernel_size height must be a positive odd integer",
        ),
        (
            {"kernel_size": [-1, 3], "sigma": 1},
            "kernel_size width must be a positive odd integer",
        ),
        (
            {"kernel_size": [3, -1], "sigma": 1},
            "kernel_size height must be a positive odd integer",
        ),
        ({"kernel_size": [3, 3], "sigma": -1}, "sigma must be a non-negative number"),
        (
            {"kernel_size": [3, 3], "sigmaY": -1},
            re.escape("sigmaY (if provided) must be a non-negative number"),
        ),
        (
            {"kernel_size": [1, 1], "sigma": 0},
            "Both kernel_size and sigma are effectively zero",
        ),
    ],
)
def test_gaussian_blur_invalid_config(invalid_config, error_message_part):
    """ガウシアンブラーの不正な設定値をテスト."""
    with pytest.raises(ProcessorValidationError, match=error_message_part):
        GaussianBlurProcessor(name="gaussian_blur_invalid", config=invalid_config)


@pytest.mark.parametrize(
    "invalid_image, error_message_part",
    [
        (DUMMY_IMAGE_FLOAT32_3CH, "Input image must be of type np.uint8"),
        (np.array([], dtype=np.uint8), "input image is empty"),
        (DUMMY_IMAGE_UINT8_2CH, "Input image must have 1 or 3 channels"),
        (DUMMY_IMAGE_UINT8_4CH, "Input image must have 1 or 3 channels"),
        (DUMMY_IMAGE_UINT8_4D, "Input image must be a 2D .* or 3-channel"),
        ("not_an_image", "image must be of type numpy.ndarray"),
    ],
)
def test_gaussian_blur_invalid_image_input(invalid_image, error_message_part):
    """ガウシアンブラーの不正な入力画像をテスト."""
    config = {"kernel_size": [3, 3], "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur_invalid_img", config=config)
    with pytest.raises(ProcessorValidationError, match=error_message_part):
        processor.process(invalid_image)


# AverageBlur
def test_average_blur_valid():
    """平均値ブラーの基本機能をテスト."""
    config = {"kernel_size": [3, 3]}
    processor = AverageBlurProcessor(name="average_blur", config=config)

    # 3チャンネル画像
    result_3ch = processor.process(DUMMY_IMAGE_UINT8_3CH.copy())
    assert result_3ch.shape == DUMMY_IMAGE_UINT8_3CH.shape
    assert result_3ch.dtype == np.uint8

    # 1チャンネル（グレースケール）画像
    result_1ch = processor.process(DUMMY_IMAGE_UINT8_1CH.copy())
    assert result_1ch.shape == DUMMY_IMAGE_UINT8_1CH.shape
    assert result_1ch.dtype == np.uint8


@pytest.mark.parametrize(
    "invalid_config, error_message_part",
    [
        (
            {"kernel_size": 3},
            "kernel_size must be specified as two positive integers",
        ),
        (
            {"kernel_size": [0, 3]},
            "kernel_size must be specified as two positive integers",
        ),
        (
            {"kernel_size": [-1, 3]},
            "kernel_size must be specified as two positive integers",
        ),
        (
            {"kernel_size": [3, 0]},
            "kernel_size must be specified as two positive integers",
        ),
        (
            {"kernel_size": [3.5, 3]},
            "kernel_size must be specified as two positive integers",
        ),
        (
            {"kernel_size": [3]},
            "kernel_size must be specified as two positive integers",
        ),
    ],
)
def test_average_blur_invalid_config(invalid_config, error_message_part):
    """平均値ブラーの不正な設定値をテスト."""
    with pytest.raises(ProcessorValidationError, match=error_message_part):
        AverageBlurProcessor(name="average_blur_invalid", config=invalid_config)


@pytest.mark.parametrize(
    "invalid_image, error_message_part",
    [
        (DUMMY_IMAGE_FLOAT32_3CH, "Input image must be of type np.uint8"),
        (np.array([], dtype=np.uint8), "input image is empty"),
        (DUMMY_IMAGE_UINT8_2CH, "Input image must have 1 or 3 channels"),
        (DUMMY_IMAGE_UINT8_4CH, "Input image must have 1 or 3 channels"),
        (DUMMY_IMAGE_UINT8_4D, "Input image must be a 2D .* or 3-channel"),
        ("not_an_image", "image must be of type numpy.ndarray"),
    ],
)
def test_average_blur_invalid_image_input(invalid_image, error_message_part):
    """平均値ブラーの不正な入力画像をテスト."""
    config = {"kernel_size": [3, 3]}
    processor = AverageBlurProcessor(name="average_blur_invalid_img", config=config)
    with pytest.raises(ProcessorValidationError, match=error_message_part):
        processor.process(invalid_image)


# MedianBlur
def test_median_blur_valid():
    """メディアンブラーの基本機能をテスト."""
    config = {"kernel_size": 3}  # 奇数を指定
    processor = MedianBlurProcessor(name="median_blur", config=config)

    result = processor.process(DUMMY_IMAGE_UINT8_3CH.copy())

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE_UINT8_3CH.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# BilateralFilter
def test_bilateral_filter_valid():
    """バイラテラルフィルタの基本機能をテスト."""
    config = {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}
    processor = BilateralFilterProcessor(name="bilateral_filter", config=config)

    result = processor.process(DUMMY_IMAGE_UINT8_3CH.copy())

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE_UINT8_3CH.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


# MotionBlur
def test_motion_blur_valid():
    """モーションブラーの基本機能をテスト."""
    config = {"kernel_size": 5, "angle": 45}
    processor = MotionBlurProcessor(name="motion_blur", config=config)

    result = processor.process(DUMMY_IMAGE_UINT8_3CH.copy())

    # 画像サイズが維持されていることを確認
    assert result.shape == DUMMY_IMAGE_UINT8_3CH.shape
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8
