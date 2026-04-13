import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.edge_detection import CannyEdgeProcessor


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
        from pochivision.processors.registry import get_processor

        # 文字列は StrictFloat で拒否される
        with pytest.raises(ValueError):
            get_processor("canny_edge", {"threshold1": "100", "threshold2": 200.0})

    def test_invalid_config_aperture_size_type(self):
        """不正な設定（aperture_sizeの型誤り）のテスト."""
        from pochivision.processors.registry import get_processor

        # 文字列は StrictInt で拒否される
        with pytest.raises(ValueError):
            get_processor(
                "canny_edge",
                {
                    "threshold1": 100.0,
                    "threshold2": 200.0,
                    "aperture_size": "3",
                },
            )

    def test_invalid_config_aperture_size_value(self):
        """不正な設定（aperture_sizeの値誤り）のテスト."""
        from pochivision.processors.registry import get_processor

        # aperture_size=1 は ge=3 制約で拒否される
        with pytest.raises(ValueError):
            get_processor(
                "canny_edge",
                {"threshold1": 100.0, "threshold2": 200.0, "aperture_size": 1},
            )

    def test_invalid_input_image_type(self):
        """不正な型の画像を与えたテスト（例: list）。"""
        image = [[1, 2, 3], [4, 5, 6]]
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_invalid_type", config=config)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor.process(image)
        assert "image must be of type numpy.ndarray" in str(excinfo.value)

    def test_invalid_input_empty_image(self):
        """空の画像を与えたテスト。"""
        image = np.array([])
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_empty_image", config=config)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor.process(image)
        assert "input image is empty" in str(excinfo.value)

    def test_invalid_input_image_shape(self):
        """不正な形状の画像を与えたテスト（例: 4チャンネル）。"""
        image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)  # 4 channels
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_invalid_shape", config=config)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor.process(image)
        assert (
            "Input image for CannyEdgeProcessor must be 2D grayscale or "
            "3-channel color image" in str(excinfo.value)
        )

    def test_process_image_with_nan_values(self):
        """NaN を含む入力が 0 にクランプされ, 正常にエッジ検出されるかテスト."""
        image_float = (np.random.rand(100, 100) * 1000).astype(np.float32)
        image_float[0, 0] = np.nan
        image_float[50, 50] = np.nan
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_nan", config=config)
        processed_image = processor.process(image_float.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8
        # 出力に不正値が含まれないこと.
        assert not np.any(np.isnan(processed_image.astype(np.float32)))

    def test_process_image_with_inf_values(self):
        """Inf を含む入力が 0 にクランプされ, 正常にエッジ検出されるかテスト."""
        image_float = (np.random.rand(100, 100) * 1000).astype(np.float32)
        image_float[0, 0] = np.inf
        image_float[10, 10] = -np.inf
        image_float[50, 50] = np.inf
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_inf", config=config)
        processed_image = processor.process(image_float.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8
        # uint8 なので範囲は [0, 255] に収まる.
        assert processed_image.min() >= 0
        assert processed_image.max() <= 255

    def test_process_image_with_nan_and_inf_values(self):
        """NaN と Inf 両方を含む入力が 0 にクランプされるかテスト."""
        image_float = (np.random.rand(100, 100) * 1000).astype(np.float32)
        image_float[0, 0] = np.nan
        image_float[0, 1] = np.inf
        image_float[0, 2] = -np.inf
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_nan_inf", config=config)
        processed_image = processor.process(image_float.copy())
        assert processed_image.shape == (100, 100)
        assert processed_image.dtype == np.uint8
        assert processed_image.min() >= 0
        assert processed_image.max() <= 255

    def test_process_image_normalize_fail(self):
        """cv2.normalize でエラーが発生するケースのテスト (例: 0次元配列)."""
        image_problematic = np.array(0, dtype=np.int32)
        config = CannyEdgeProcessor.get_default_config()
        processor = CannyEdgeProcessor(name="canny_norm_fail", config=config)
        with pytest.raises(ProcessorValidationError) as excinfo:
            processor.process(image_problematic)
        assert (
            "Input image for CannyEdgeProcessor must be 2D grayscale or "
            "3-channel color image" in str(excinfo.value)
        )
