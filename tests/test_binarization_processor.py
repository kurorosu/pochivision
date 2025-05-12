import numpy as np
import pytest  # noqa: F401

from processors.binarization import (
    OtsuBinarizationProcessor,
    StandardBinarizationProcessor,
)

# テスト用の画像データ
DUMMY_COLOR = np.ones((10, 10, 3), dtype=np.uint8) * 100
DUMMY_GRAY = np.ones((10, 10), dtype=np.uint8) * 100


def test_binarization_valid_gray():
    """グレースケール画像からの2値化をテスト"""
    config = {"threshold": 50}
    processor = StandardBinarizationProcessor(
        name="standard_binarization", config=config
    )

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
        name="standard_binarization", config=config
    )

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_valid_gray():
    """グレースケール画像からの大津の2値化をテスト"""
    config = {}  # 大津の2値化は設定不要
    processor = OtsuBinarizationProcessor(name="otsu_binarization", config=config)

    # グレースケール画像を与えると
    result = processor.process(DUMMY_GRAY)

    # 2値化された画像が返される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_valid_color():
    """カラー画像からの大津の2値化をテスト"""
    config = {}  # 大津の2値化は設定不要
    processor = OtsuBinarizationProcessor(name="otsu_binarization", config=config)

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_invalid_input():
    """不正な入力画像に対するテスト"""
    config = {}
    processor = OtsuBinarizationProcessor(name="otsu_binarization", config=config)

    # 1チャンネルの画像は不正
    invalid_image = np.ones((10, 10, 1), dtype=np.uint8) * 100
    with pytest.raises(Exception):
        processor.process(invalid_image)

    # 空の画像は不正
    empty_image = np.array([], dtype=np.uint8)
    with pytest.raises(Exception):
        processor.process(empty_image)
