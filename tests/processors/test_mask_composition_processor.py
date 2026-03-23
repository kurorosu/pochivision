"""MaskCompositionProcessor のテスト."""

import numpy as np
import pytest

from pochivision.exceptions import ProcessorRuntimeError, ProcessorValidationError
from pochivision.processors.mask_composition import MaskCompositionProcessor


class TestMaskCompositionProcessorInit:
    """初期化とバリデーションのテスト."""

    def test_default_config(self):
        """デフォルト設定で初期化できる."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        assert processor.target_image_name == "original"
        assert processor.use_white_pixels is True
        assert processor.enable_cropping is False
        assert processor.crop_margin == 5

    def test_custom_config(self):
        """カスタム設定で初期化できる."""
        config = {
            "target_image": "grayscale",
            "use_white_pixels": False,
            "enable_cropping": True,
            "crop_margin": 10,
        }
        processor = MaskCompositionProcessor(name="mask_composition", config=config)
        assert processor.target_image_name == "grayscale"
        assert processor.use_white_pixels is False
        assert processor.enable_cropping is True
        assert processor.crop_margin == 10

    def test_invalid_target_image_type(self):
        """target_image が文字列でない場合 ProcessorValidationError."""
        with pytest.raises(ProcessorValidationError, match="target_image"):
            MaskCompositionProcessor(
                name="mask_composition", config={"target_image": 123}
            )

    def test_invalid_use_white_pixels_type(self):
        """use_white_pixels が bool でない場合 ProcessorValidationError."""
        with pytest.raises(ProcessorValidationError, match="use_white_pixels"):
            MaskCompositionProcessor(
                name="mask_composition", config={"use_white_pixels": "yes"}
            )

    def test_invalid_crop_margin_negative(self):
        """crop_margin が負の場合 ProcessorValidationError."""
        with pytest.raises(ProcessorValidationError, match="crop_margin"):
            MaskCompositionProcessor(
                name="mask_composition", config={"crop_margin": -1}
            )


class TestMaskCompositionProcessorProcess:
    """process メソッドのテスト."""

    @pytest.fixture
    def processor(self) -> MaskCompositionProcessor:
        """デフォルト設定の MaskCompositionProcessor を返す."""
        return MaskCompositionProcessor(name="mask_composition", config={})

    def test_process_without_target_image(
        self, processor: MaskCompositionProcessor, dummy_binary_image: np.ndarray
    ):
        """ターゲット画像未設定で ProcessorRuntimeError."""
        with pytest.raises(ProcessorRuntimeError, match="not set"):
            processor.process(dummy_binary_image)

    def test_process_with_white_pixels(self, dummy_binary_image: np.ndarray):
        """白ピクセル部分をターゲット画像で置き換える."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        target = np.full((100, 100, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(dummy_binary_image)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_process_with_black_pixels(self, dummy_binary_image: np.ndarray):
        """use_white_pixels=False で黒ピクセル部分を置き換える."""
        processor = MaskCompositionProcessor(
            name="mask_composition", config={"use_white_pixels": False}
        )
        target = np.full((100, 100, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(dummy_binary_image)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_process_with_color_binary_mask(self, dummy_binary_color_image: np.ndarray):
        """カラー2値画像をマスクとして使用できる."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        target = np.full((100, 100, 3), 200, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(dummy_binary_color_image)

        assert result.shape == (100, 100, 3)

    def test_process_with_size_mismatch(self, dummy_binary_image: np.ndarray):
        """マスクとターゲット画像のサイズが異なる場合にリサイズされる."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        target = np.full((200, 200, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(dummy_binary_image)

        assert result.shape[:2] == dummy_binary_image.shape[:2]

    def test_process_with_cropping(self):
        """enable_cropping=True でトリミングが実行される."""
        processor = MaskCompositionProcessor(
            name="mask_composition",
            config={"enable_cropping": True, "crop_margin": 2},
        )
        # 中央に白い領域のある2値画像
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        target = np.full((100, 100, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(mask)

        # トリミングされるので元画像より小さい
        assert result.shape[0] < 100
        assert result.shape[1] < 100

    def test_process_empty_image(self, processor: MaskCompositionProcessor):
        """空画像で ProcessorValidationError."""
        processor.set_target_image(np.full((100, 100, 3), 128, dtype=np.uint8))
        with pytest.raises(ProcessorValidationError, match="empty"):
            processor.process(np.array([], dtype=np.uint8))

    def test_process_non_binary_image(self, processor: MaskCompositionProcessor):
        """非2値画像で ProcessorValidationError."""
        processor.set_target_image(np.full((100, 100, 3), 128, dtype=np.uint8))
        non_binary = np.full((100, 100), 100, dtype=np.uint8)
        with pytest.raises(ProcessorValidationError, match="binary"):
            processor.process(non_binary)


class TestMaskCompositionProcessorPipelineMode:
    """パイプラインモード関連のテスト."""

    def test_set_pipeline_mode_parallel_raises(self):
        """パラレルモード設定で ProcessorRuntimeError."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        with pytest.raises(ProcessorRuntimeError, match="pipeline mode"):
            processor.set_pipeline_mode("parallel")

    def test_set_pipeline_mode_pipeline_ok(self):
        """パイプラインモード設定は例外を発生しない."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        processor.set_pipeline_mode("pipeline")


class TestMaskCompositionProcessorDefaultConfig:
    """get_default_config のテスト."""

    def test_default_config_keys(self):
        """デフォルト設定に必要なキーが含まれている."""
        config = MaskCompositionProcessor.get_default_config()
        assert "target_image" in config
        assert "use_white_pixels" in config
        assert "enable_cropping" in config
        assert "crop_margin" in config
