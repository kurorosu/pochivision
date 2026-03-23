import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.grayscale import GrayscaleProcessor

# テスト用の画像データ
DUMMY_COLOR = np.ones((10, 10, 3), dtype=np.uint8) * 100


def test_grayscale_valid():
    """カラー画像からグレースケールへの変換をテスト."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    result = processor.process(DUMMY_COLOR)

    # チャンネル数が減っていることを確認（3→1）
    assert result.ndim == 2
    # 元のサイズ（高さ・幅）が維持されていることを確認
    assert result.shape == (10, 10)
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


def test_grayscale_already_grayscale():
    """既にグレースケールの画像はそのまま返される."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    gray_image = np.ones((10, 10), dtype=np.uint8) * 100
    result = processor.process(gray_image)

    assert result.ndim == 2
    assert result.shape == (10, 10)


def test_grayscale_empty_image():
    """空画像で ProcessorValidationError."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    with pytest.raises(ProcessorValidationError, match="empty"):
        processor.process(np.array([], dtype=np.uint8))


def test_grayscale_non_ndarray():
    """numpy.ndarray 以外で ProcessorValidationError."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    with pytest.raises(ProcessorValidationError, match="numpy.ndarray"):
        processor.process([[1, 2], [3, 4]])


def test_grayscale_wrong_dtype():
    """uint8 以外の dtype で ProcessorValidationError."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    float_image = np.ones((10, 10, 3), dtype=np.float32)
    with pytest.raises(ProcessorValidationError, match="np.uint8"):
        processor.process(float_image)
