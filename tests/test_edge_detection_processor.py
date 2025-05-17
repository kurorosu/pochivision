import numpy as np
import pytest

from exceptions import ProcessorRuntimeError
from processors.edge_detection import CannyEdgeProcessor
from processors.validators.base import ProcessorValidationError


# Helper function to create a simple image
def create_sample_image(height: int, width: int, channels: int = 3) -> np.ndarray:
    """テスト用の簡単な画像を作成します."""
    if channels == 3:
        return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    elif channels == 1:
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    raise ValueError("Channels must be 1 or 3")


class TestCannyEdgeProcessor:
    """
    CannyEdgeProcessorのテストクラス.
    """

    def test_initialization_default_config(self):
        """デフォルト設定でのプロセッサ初期化をテストします."""
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_test", config=config)
        assert processor.name == "canny_test"
        assert processor.config == config
        assert processor._threshold1 == 100.0
        assert processor._threshold2 == 200.0
        assert processor._aperture_size == 3
        assert not processor._l2_gradient

    def test_initialization_custom_config(self):
        """カスタム設定でのプロセッサ初期化をテストします."""
        custom_config = {
            "threshold1": 50.0,
            "threshold2": 150.0,
            "aperture_size": 5,
            "l2_gradient": True,
        }
        processor = CannyEdgeProcessor(name="canny_custom", config=custom_config)
        assert processor._threshold1 == 50.0
        assert processor._threshold2 == 150.0
        assert processor._aperture_size == 5
        assert processor._l2_gradient

    def test_process_bgr_image(self):
        """BGR画像の処理をテストします."""
        image = create_sample_image(100, 100, 3)
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_bgr", config=config)
        processed_image = processor.process(image.copy())

        assert processed_image.shape == (100, 100)  # Output is grayscale
        assert processed_image.dtype == np.uint8
        # Check if Canny was applied (at least some pixels should be non-zero or zero)
        assert np.any(processed_image == 0) or np.any(processed_image == 255)

    def test_process_grayscale_image(self):
        """グレースケール画像の処理をテストします."""
        image = create_sample_image(100, 100, 1)
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_gray", config=config)
        processed_image = processor.process(image.copy())

        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8
        assert np.any(processed_image == 0) or np.any(processed_image == 255)

    def test_process_float_image_0_1_range(self):
        """[0, 1]範囲の浮動小数点数画像を処理するテスト."""
        image_float = np.random.rand(100, 100).astype(np.float32)  # Grayscale float
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_float", config=config)
        processed_image = processor.process(image_float.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8

    def test_process_float_image_0_255_range(self):
        """[0, 255]範囲の浮動小数点数画像を処理するテスト."""
        image_float = (np.random.rand(100, 100) * 255).astype(
            np.float32
        )  # Grayscale float
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_float255", config=config)
        processed_image = processor.process(image_float.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8

    def test_process_uint16_image(self):
        """uint16画像の処理をテストします."""
        image_uint16 = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_uint16", config=config)
        processed_image = processor.process(image_uint16.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8

    def test_invalid_config_missing_thresholds(self):
        """設定でthresholdが欠落している場合、デフォルト値が使用されることをテスト."""
        # このテストは、以前はエラーを期待していましたが、
        # 設計思想の変更により、thresholdが存在しない場合はデフォルト値が使用されるため、
        # エラーが発生しないこと、およびデフォルト値が適用されることを確認します。
        config = {}
        processor = CannyEdgeProcessor(name="canny_no_thresholds", config=config)
        default_config = CannyEdgeProcessor.get_default_config()
        assert processor._threshold1 == default_config["threshold1"]
        assert processor._threshold2 == default_config["threshold2"]
        # aperture_size と l2_gradient も同様にデフォルト値が設定されることを確認
        assert processor._aperture_size == default_config["aperture_size"]
        assert processor._l2_gradient == default_config["l2_gradient"]

    def test_invalid_config_threshold_type(self):
        """不正な設定（しきい値の型誤り）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid", config={"threshold1": "100", "threshold2": 200}
            )
        assert "'threshold1' must be a number" in str(excinfo.value)

    def test_invalid_config_negative_threshold(self):
        """不正な設定（負のしきい値）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid", config={"threshold1": -10, "threshold2": 200}
            )
        assert "Canny 'threshold1' must be non-negative. Got -10." in str(excinfo.value)

    def test_invalid_config_threshold_order(self):
        """不正な設定（threshold1 > threshold2）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid", config={"threshold1": 200, "threshold2": 100}
            )
        assert "'threshold1' should not be greater than 'threshold2'" in str(
            excinfo.value
        )

    def test_invalid_config_aperture_size_type(self):
        """不正な設定（aperture_sizeの型誤り）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid",
                config={
                    "threshold1": 100,
                    "threshold2": 200,
                    "aperture_size": "3",
                },
            )
        assert "'aperture_size' must be an integer" in str(excinfo.value)

    def test_invalid_config_aperture_size_value(self):
        """不正な設定（aperture_sizeの値誤り）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid",
                config={"threshold1": 100, "threshold2": 200, "aperture_size": 4},
            )
        assert "'aperture_size' must be 3, 5, or 7" in str(excinfo.value)
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid",
                config={"threshold1": 100, "threshold2": 200, "aperture_size": 1},
            )
        assert "'aperture_size' must be 3, 5, or 7" in str(excinfo.value)

    def test_invalid_config_l2_gradient_type(self):
        """不正な設定（l2_gradientの型誤り）のテスト."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            CannyEdgeProcessor(
                name="canny_invalid",
                config={
                    "threshold1": 100,
                    "threshold2": 200,
                    "l2_gradient": "true",
                },
            )
        assert "'l2_gradient' must be a boolean" in str(excinfo.value)

    def test_invalid_input_image_shape(self):
        """不正な形状の画像を与えたテスト（例: 4チャンネル）。"""
        image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)  # 4 channels
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_invalid_shape", config=config)
        with pytest.raises(ProcessorRuntimeError) as excinfo:
            processor.process(image)
        assert (
            "Input image for CannyEdgeProcessor must be 2D grayscale or "
            "3-channel color image." in str(excinfo.value)
        )
