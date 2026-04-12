"""ブラープロセッサのテストモジュール."""

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


# GaussianBlur kernel_size バリデーション
@pytest.mark.parametrize(
    "kernel_size",
    [[3, 3], [5, 5], [15, 15], (3, 3)],
)
def test_gaussian_blur_valid_kernel_size(kernel_size):
    """GaussianBlur: 3 以上の奇数ペアで初期化が通ることを確認."""
    config = {"kernel_size": kernel_size, "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur_ok", config=config)
    assert processor.kernel_size == (kernel_size[0], kernel_size[1])


@pytest.mark.parametrize(
    "kernel_size",
    [
        [4, 4],  # 偶数
        [3, 4],  # 片方偶数
        [1, 1],  # 3 未満
        [0, 0],  # 0
        [-1, 3],  # 負数
        [3],  # 長さ不正
        5,  # list/tuple でない
    ],
)
def test_gaussian_blur_invalid_kernel_size(kernel_size):
    """GaussianBlur: 偶数・0・負数・長さ不正で ProcessorValidationError が発生."""
    config = {"kernel_size": kernel_size, "sigma": 1}
    with pytest.raises(ProcessorValidationError, match="kernel_size"):
        GaussianBlurProcessor(name="gaussian_blur_ng", config=config)


# MedianBlur kernel_size バリデーション
@pytest.mark.parametrize("kernel_size", [3, 5, 15])
def test_median_blur_valid_kernel_size(kernel_size):
    """MedianBlur: 3 以上の奇数スカラーで初期化が通ることを確認."""
    config = {"kernel_size": kernel_size}
    processor = MedianBlurProcessor(name="median_blur_ok", config=config)
    assert processor.kernel_size == kernel_size


@pytest.mark.parametrize(
    "kernel_size",
    [4, 0, 1, -1, -3, [3, 3]],
)
def test_median_blur_invalid_kernel_size(kernel_size):
    """MedianBlur: 偶数・0・1・負数・非 int で ProcessorValidationError が発生."""
    config = {"kernel_size": kernel_size}
    with pytest.raises(ProcessorValidationError, match="kernel_size"):
        MedianBlurProcessor(name="median_blur_ng", config=config)


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
