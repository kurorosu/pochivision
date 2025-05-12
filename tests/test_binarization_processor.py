import numpy as np
import pytest  # noqa: F401

from exceptions import ProcessorRuntimeError
from processors.binarization import (
    GaussianAdaptiveBinarizationProcessor,
    MeanAdaptiveBinarizationProcessor,
    OtsuBinarizationProcessor,
    StandardBinarizationProcessor,
)

# テスト用の画像データ
DUMMY_COLOR = np.ones((10, 10, 3), dtype=np.uint8) * 100
DUMMY_GRAY = np.ones((10, 10), dtype=np.uint8) * 100


def test_binarization_valid_gray():
    """グレースケール画像からの2値化をテスト"""
    config = {"threshold": 50}
    processor = StandardBinarizationProcessor(name="std_bin", config=config)

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
    processor = StandardBinarizationProcessor(name="std_bin", config=config)

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_valid_gray():
    """グレースケール画像からの大津の2値化をテスト"""
    config = {}  # 大津の2値化は設定不要
    processor = OtsuBinarizationProcessor(name="otsu_bin", config=config)

    # グレースケール画像を与えると
    result = processor.process(DUMMY_GRAY)

    # 2値化された画像が返される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_valid_color():
    """カラー画像からの大津の2値化をテスト"""
    config = {}  # 大津の2値化は設定不要
    processor = OtsuBinarizationProcessor(name="otsu_bin", config=config)

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


def test_gaussian_adaptive_binarization_valid_gray():
    """グレースケール画像からのガウシアン適応的2値化をテスト"""
    config = {"block_size": 5, "c": 2}
    processor = GaussianAdaptiveBinarizationProcessor(
        name="gauss_adapt_bin", config=config
    )

    # グレースケール画像を与えると
    result = processor.process(DUMMY_GRAY)

    # 2値化された画像が返される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_gaussian_adaptive_binarization_valid_color():
    """カラー画像からのガウシアン適応的2値化をテスト"""
    config = {"block_size": 5, "c": 2}
    processor = GaussianAdaptiveBinarizationProcessor(
        name="gauss_adapt_bin", config=config
    )

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_mean_adaptive_binarization_valid_gray():
    """グレースケール画像からの平均適応的2値化をテスト"""
    config = {"block_size": 5, "c": 2}
    processor = MeanAdaptiveBinarizationProcessor(name="mean_adapt_bin", config=config)

    # グレースケール画像を与えると
    result = processor.process(DUMMY_GRAY)

    # 2値化された画像が返される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_mean_adaptive_binarization_valid_color():
    """カラー画像からの平均適応的2値化をテスト"""
    config = {"block_size": 5, "c": 2}
    processor = MeanAdaptiveBinarizationProcessor(name="mean_adapt_bin", config=config)

    # カラー画像を与えると
    result = processor.process(DUMMY_COLOR)

    # 自動的にグレースケール変換後、2値化される
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_adaptive_binarization_invalid_input():
    """不正な入力画像に対するテスト"""
    # ガウシアン適応的2値化
    gaussian_config = {"block_size": 5, "c": 2}
    gaussian_processor = GaussianAdaptiveBinarizationProcessor(
        name="gauss_adapt_bin", config=gaussian_config
    )

    # 1チャンネルの画像は不正
    invalid_image = np.ones((10, 10, 1), dtype=np.uint8) * 100
    with pytest.raises(Exception):
        gaussian_processor.process(invalid_image)

    # 空の画像は不正
    empty_image = np.array([], dtype=np.uint8)
    with pytest.raises(Exception):
        gaussian_processor.process(empty_image)

    # 平均適応的2値化
    mean_config = {"block_size": 5, "c": 2}
    mean_processor = MeanAdaptiveBinarizationProcessor(
        name="mean_adapt_bin", config=mean_config
    )

    # 1チャンネルの画像は不正
    with pytest.raises(Exception):
        mean_processor.process(invalid_image)

    # 空の画像は不正
    with pytest.raises(Exception):
        mean_processor.process(empty_image)


def test_adaptive_binarization_invalid_config():
    """不正な設定に対するテスト"""
    # ガウシアン適応的2値化
    with pytest.raises(ProcessorRuntimeError):
        GaussianAdaptiveBinarizationProcessor(
            name="gauss_adapt_bin", config={"block_size": 4}  # 偶数は不正
        )

    # 平均適応的2値化
    with pytest.raises(ProcessorRuntimeError):
        MeanAdaptiveBinarizationProcessor(
            name="mean_adapt_bin", config={"block_size": 4}  # 偶数は不正
        )
