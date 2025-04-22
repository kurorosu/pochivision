import numpy as np
import pytest
from processors.binarization import StandardBinarizationProcessor

# テスト用の画像データ
DUMMY_COLOR = np.ones((10, 10, 3), dtype=np.uint8) * 100
DUMMY_GRAY = np.ones((10, 10), dtype=np.uint8) * 100


def test_binarization_valid_gray():
    """グレースケール画像からの2値化をテスト"""
    config = {"threshold": 50}
    processor = StandardBinarizationProcessor(
        name="standard_binarization", config=config)

    # グレースケール画像を与えると
    result = processor.process(DUMMY_GRAY)

    # 2値化された画像が返される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
    # 閾値より大きいピクセルは255になる
    assert np.all(result == 255)  # DUMMY_GRAYは100なのでしきい値50より大きい


def test_binarization_valid_color():
    """カラー画像からの2値化をテスト"""
    config = {"threshold": 50}
    processor = StandardBinarizationProcessor(
        name="standard_binarization", config=config)

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
