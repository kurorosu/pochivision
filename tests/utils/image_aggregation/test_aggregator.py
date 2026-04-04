"""ImageAggregator のテスト."""

import cv2
import numpy as np

from pochivision.utils.image_aggregation.aggregator import ImageAggregator
from pochivision.utils.image_aggregation.operations import OperationMode
from pochivision.workspace import OutputManager


def _create_test_image_file(path, width: int = 10, height: int = 10) -> None:
    """テスト用画像ファイルを作成する."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


class TestImageAggregatorCopy:
    """ImageAggregator の copy モードテスト."""

    def test_aggregate_copies_images(self, tmp_path):
        """copy モードで画像が集約される."""
        # 入力構造: input/date/processor/image.jpg
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image_file(input_dir / "20260401_0" / "resize" / "img1.jpg")
        _create_test_image_file(input_dir / "20260401_0" / "resize" / "img2.jpg")

        output_dir = tmp_path / "output"
        output_manager = OutputManager(str(output_dir))

        aggregator = ImageAggregator(
            str(input_dir), mode=OperationMode.COPY, output_manager=output_manager
        )
        count = aggregator.aggregate()

        assert count == 2
        # 元ファイルが残っている (copy)
        assert (input_dir / "20260401_0" / "resize" / "img1.jpg").exists()

    def test_aggregate_empty_directory(self, tmp_path):
        """空のディレクトリで 0 を返す."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_manager = OutputManager(str(output_dir))

        aggregator = ImageAggregator(
            str(input_dir), mode=OperationMode.COPY, output_manager=output_manager
        )
        count = aggregator.aggregate()

        assert count == 0

    def test_aggregate_multiple_processor_types(self, tmp_path):
        """複数の処理タイプを集約できる."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "original").mkdir(parents=True)
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image_file(input_dir / "20260401_0" / "original" / "img1.jpg")
        _create_test_image_file(input_dir / "20260401_0" / "resize" / "img1.jpg")

        output_dir = tmp_path / "output"
        output_manager = OutputManager(str(output_dir))

        aggregator = ImageAggregator(
            str(input_dir), mode=OperationMode.COPY, output_manager=output_manager
        )
        count = aggregator.aggregate()

        assert count == 2

    def test_aggregate_duplicate_filenames(self, tmp_path):
        """重複ファイル名が連番で解決される."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        (input_dir / "20260402_0" / "resize").mkdir(parents=True)
        # 同名ファイルを2つの日付フォルダに作成
        _create_test_image_file(input_dir / "20260401_0" / "resize" / "img.jpg")
        _create_test_image_file(input_dir / "20260402_0" / "resize" / "img.jpg")

        output_dir = tmp_path / "output"
        output_manager = OutputManager(str(output_dir))

        aggregator = ImageAggregator(
            str(input_dir), mode=OperationMode.COPY, output_manager=output_manager
        )
        count = aggregator.aggregate()

        assert count == 2


class TestImageAggregatorMove:
    """ImageAggregator の move モードテスト."""

    def test_aggregate_moves_images(self, tmp_path):
        """move モードで画像が移動される."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image_file(input_dir / "20260401_0" / "resize" / "img1.jpg")

        output_dir = tmp_path / "output"
        output_manager = OutputManager(str(output_dir))

        aggregator = ImageAggregator(
            str(input_dir), mode=OperationMode.MOVE, output_manager=output_manager
        )
        count = aggregator.aggregate()

        assert count == 1
        # 元ファイルが消えている (move)
        assert not (input_dir / "20260401_0" / "resize" / "img1.jpg").exists()
