import numpy as np
import pytest  # noqa: F401

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.binarization import (
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
    result = processor.process(DUMMY_GRAY)
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_valid_color():
    """カラー画像からの大津の2値化をテスト"""
    config = {}  # 大津の2値化は設定不要
    processor = OtsuBinarizationProcessor(name="otsu_bin", config=config)
    result = processor.process(DUMMY_COLOR)
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_otsu_binarization_invalid_input():
    """OtsuBinarizationProcessor の不正な入力画像に対するテスト."""
    config = {}
    processor = OtsuBinarizationProcessor(name="otsu_binarization", config=config)

    # 不正な型の画像
    invalid_image_type = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_type)  # type: ignore
    assert "image must be of type numpy.ndarray" in str(excinfo.value)

    # 空の画像
    empty_image = np.array([], dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(empty_image)
    assert "input image is empty" in str(excinfo.value)

    # 不正な次元の画像 (例: 0次元)
    invalid_image_0dim = np.array(0, dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_0dim)
    assert (
        "Input image for OtsuBinarization must be 2D grayscale or 3/4 channel"
        in str(excinfo.value)
    )

    # 1チャンネルの画像 (現状のバリデータではエラーになるのでテストケースとして残す)
    invalid_image_1channel = np.ones((10, 10, 1), dtype=np.uint8) * 100
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_1channel)
    assert (
        "Input image for OtsuBinarization must be 2D grayscale or 3/4 channel"
        in str(excinfo.value)
    )


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
    result = processor.process(DUMMY_GRAY)
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_mean_adaptive_binarization_valid_color():
    """カラー画像からの平均適応的2値化をテスト"""
    config = {"block_size": 5, "c": 2}
    processor = MeanAdaptiveBinarizationProcessor(name="mean_adapt_bin", config=config)
    result = processor.process(DUMMY_COLOR)
    assert result.ndim == 2
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_adaptive_binarization_invalid_input():
    """AdaptiveBinarizationProcessor の不正な入力画像に対するテスト."""
    gaussian_config = {"block_size": 11, "c": 2}  # block_sizeを奇数に修正
    gaussian_processor = GaussianAdaptiveBinarizationProcessor(
        name="gauss_adapt_bin", config=gaussian_config
    )
    mean_config = {"block_size": 11, "c": 2}  # block_sizeを奇数に修正
    mean_processor = MeanAdaptiveBinarizationProcessor(
        name="mean_adapt_bin", config=mean_config
    )

    processors_to_test = {
        "GaussianAdaptiveBinarization": gaussian_processor,
        "MeanAdaptiveBinarization": mean_processor,
    }

    for processor_name, processor_instance in processors_to_test.items():
        # 不正な型の画像
        invalid_image_type = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor_instance.process(invalid_image_type)  # type: ignore
        assert "image must be of type numpy.ndarray" in str(
            excinfo.value
        ), f"{processor_name} failed type check"

        # 空の画像
        empty_image = np.array([], dtype=np.uint8)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor_instance.process(empty_image)
        assert "input image is empty" in str(
            excinfo.value
        ), f"{processor_name} failed empty check"

        # 不正な次元の画像 (例: 0次元)
        invalid_image_0dim = np.array(0, dtype=np.uint8)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor_instance.process(invalid_image_0dim)
        assert (
            f"Input image for {processor_name} must be 2D grayscale or 3/4 channel"
            in str(excinfo.value)
        ), f"{processor_name} failed 0dim check"

        # 1チャンネルの画像
        invalid_image_1channel = np.ones((10, 10, 1), dtype=np.uint8) * 100
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor_instance.process(invalid_image_1channel)
        assert (
            f"Input image for {processor_name} must be 2D grayscale or 3/4 channel"
            in str(excinfo.value)
        ), f"{processor_name} failed 1channel check"


def test_adaptive_binarization_invalid_config():
    """不正な設定に対するテスト.

    block_size の偶数/奇数チェックは Pydantic スキーマの対象外のため,
    型の不一致をテストする.
    """
    from pochivision.processors.registry import get_processor

    # block_size に文字列を渡す (StrictInt で拒否される)
    with pytest.raises(ValueError):
        get_processor("gauss_adapt_bin", {"block_size": "4", "c": 2.0})

    with pytest.raises(ValueError):
        get_processor("mean_adapt_bin", {"block_size": "2", "c": 2.0})


def test_standard_binarization_invalid_threshold_config():
    """StandardBinarizationProcessor の不正なしきい値設定に対するテスト.

    threshold の範囲チェック (0-255) は Pydantic スキーマの対象外のため,
    型の不一致のみテストする.
    """
    from pochivision.processors.registry import get_processor

    # 文字列は StrictInt で拒否される
    with pytest.raises(ValueError):
        get_processor("std_bin", {"threshold": "abc"})


def test_standard_binarization_invalid_input_type():
    """StandardBinarizationProcessor の不正な型の入力画像に対するテスト."""
    config = {"threshold": 128}
    processor = StandardBinarizationProcessor(name="std_bin", config=config)
    invalid_image_type = [[1, 2, 3], [4, 5, 6]]  # list型
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_type)  # type: ignore
    assert "image must be of type numpy.ndarray" in str(excinfo.value)


def test_standard_binarization_empty_input_image():
    """StandardBinarizationProcessor の空の入力画像に対するテスト."""
    config = {"threshold": 128}
    processor = StandardBinarizationProcessor(name="std_bin", config=config)
    empty_image = np.array([])
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(empty_image)
    assert "input image is empty" in str(excinfo.value)


def test_standard_binarization_invalid_input_shape():
    """StandardBinarizationProcessor の不正な形状の入力画像に対するテスト."""
    config = {"threshold": 128}
    processor = StandardBinarizationProcessor(name="std_bin", config=config)
    # 不正な次元の画像 (例: 0次元)
    invalid_image_0dim = np.array(0, dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_0dim)
    assert (
        "Input image for StandardBinarization must be 2D grayscale or 3/4 channel"
        in str(excinfo.value)
    )

    # 不正なチャンネル数の画像 (例: 5チャンネル)
    invalid_image_5channel = np.ones((10, 10, 5), dtype=np.uint8) * 100
    with pytest.raises(ProcessorValidationError) as excinfo:
        processor.process(invalid_image_5channel)
    assert (
        "Input image for StandardBinarization must be 2D grayscale or 3/4 channel"
        in str(excinfo.value)
    )
