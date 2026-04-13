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
    # 早期リターンにより同一オブジェクトが返されること (冗長変換が発生しない)
    assert result is gray_image


def test_grayscale_already_grayscale_3d_single_channel():
    """3 次元 1 チャンネル画像は 2 次元グレースケールに整形されて返される."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    gray_image = np.ones((10, 10, 1), dtype=np.uint8) * 100
    result = processor.process(gray_image)

    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
    # 値が保持されていること
    assert np.array_equal(result, gray_image.squeeze(axis=2))


def test_grayscale_double_apply_is_idempotent():
    """grayscale を 2 回適用しても例外にならず同じ結果を返す."""
    processor = GrayscaleProcessor(name="grayscale", config={})
    first = processor.process(DUMMY_COLOR)
    second = processor.process(first)

    assert second.ndim == 2
    assert second.shape == (10, 10)
    assert np.array_equal(first, second)


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
