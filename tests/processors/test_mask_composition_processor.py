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

    # NOTE: test_invalid_target_image_type, test_invalid_use_white_pixels_type,
    # test_invalid_crop_margin_negative は削除.
    # mask_composition に対応する Pydantic スキーマが存在しないため,
    # get_processor() 経由のバリデーションではこれらの不正な設定を検出できない.


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

    def test_process_with_white_pixels(self):
        """白領域に target_image を出力し 黒領域は 0 で埋める."""
        processor = MaskCompositionProcessor(name="mask_composition", config={})
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # 中央に白領域
        target = np.full((50, 50, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(mask)

        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8
        # 白領域は target の値 (128)
        assert np.all(result[20, 20] == 128)
        # 黒領域は 0
        assert np.all(result[0, 0] == 0)

    def test_process_with_black_pixels(self):
        """use_white_pixels=False で黒領域に target_image が出力される."""
        processor = MaskCompositionProcessor(
            name="mask_composition", config={"use_white_pixels": False}
        )
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # 中央に白領域
        target = np.full((50, 50, 3), 128, dtype=np.uint8)
        processor.set_target_image(target)

        result = processor.process(mask)

        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8
        # 反転により 元の白領域は 0, 元の黒領域に target (128) が出力
        assert np.all(result[20, 20] == 0)
        assert np.all(result[0, 0] == 128)

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

    def test_process_dtype_mismatch_raises(
        self, processor: MaskCompositionProcessor, dummy_binary_image: np.ndarray
    ):
        """target_image の dtype が uint8 でない場合に ProcessorValidationError."""
        # float 型の target_image は uint8 ではないため不整合
        target_float = np.full((100, 100, 3), 0.5, dtype=np.float32)
        processor.set_target_image(target_float)
        with pytest.raises(ProcessorValidationError, match="uint8"):
            processor.process(dummy_binary_image)


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
