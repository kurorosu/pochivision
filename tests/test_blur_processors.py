import numpy as np
import pytest

from processors.blur import (
    GaussianBlurProcessor,
    AverageBlurProcessor,
    MedianBlurProcessor,
    BilateralFilterProcessor,
    MotionBlurProcessor,
)
from exceptions import ProcessorRuntimeError

# ダミー画像
DUMMY_IMAGE = np.ones((10, 10, 3), dtype=np.uint8)

# GaussianBlur


def test_gaussian_blur_valid():
    config = {"kernel_size": [3, 3], "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur", config=config)
    result = processor.process(DUMMY_IMAGE)
    assert result.shape == DUMMY_IMAGE.shape


def test_gaussian_blur_invalid_kernel():
    config = {"kernel_size": [0, 0], "sigma": 1}
    processor = GaussianBlurProcessor(name="gaussian_blur", config=config)
    with pytest.raises(ProcessorRuntimeError):
        processor.process(DUMMY_IMAGE)

# AverageBlur


def test_average_blur_invalid_kernel():
    config = {"kernel_size": [0, 0]}
    processor = AverageBlurProcessor(name="average_blur", config=config)
    with pytest.raises(ProcessorRuntimeError):
        processor.process(DUMMY_IMAGE)

# MedianBlur


def test_median_blur_invalid_kernel():
    config = {"kernel_size": 0}
    processor = MedianBlurProcessor(name="median_blur", config=config)
    with pytest.raises(ProcessorRuntimeError):
        processor.process(DUMMY_IMAGE)

# BilateralFilter


def test_bilateral_blur_invalid_d():
    config = {"d": 0, "sigmaColor": 75, "sigmaSpace": 75}
    processor = BilateralFilterProcessor(
        name="bilateral_filter", config=config)
    with pytest.raises(ProcessorRuntimeError):
        processor.process(DUMMY_IMAGE)

# MotionBlur


def test_motion_blur_invalid_kernel():
    config = {"kernel_size": 0, "angle": 0}
    processor = MotionBlurProcessor(name="motion_blur", config=config)
    with pytest.raises(ProcessorRuntimeError):
        processor.process(DUMMY_IMAGE)
