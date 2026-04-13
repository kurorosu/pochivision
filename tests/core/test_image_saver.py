"""ImageSaver のテスト."""

import threading

import cv2
import numpy as np

from pochivision.core.image_saver import ImageSaver


def _create_test_image(width: int = 100, height: int = 80) -> np.ndarray:
    """テスト用画像を生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestImageSaver:
    """ImageSaver のテスト."""

    def test_get_processing_dir_creates_directory(self, tmp_path):
        """サブディレクトリが作成される."""
        saver = ImageSaver(tmp_path)
        result = saver.get_processing_dir("test_dir")

        assert result == tmp_path / "test_dir"
        assert result.exists()

    def test_get_processing_dir_nested(self, tmp_path):
        """ネストされたディレクトリが作成される."""
        saver = ImageSaver(tmp_path)
        result = saver.get_processing_dir("nested/dir")

        assert result.exists()

    def test_save_creates_file(self, tmp_path):
        """画像ファイルが保存される."""
        saver = ImageSaver(tmp_path)
        image = _create_test_image()

        saver.save(image, "test_processor")

        saved_files = list((tmp_path / "test_processor").glob("*"))
        assert len(saved_files) == 1

    def test_save_image_content(self, tmp_path):
        """保存された画像が読み込み可能."""
        saver = ImageSaver(tmp_path)
        image = _create_test_image(width=64, height=48)

        saver.save(image, "content_test")

        saved_files = list((tmp_path / "content_test").glob("*"))
        loaded = cv2.imread(str(saved_files[0]))
        assert loaded is not None
        assert loaded.shape == (48, 64, 3)

    def test_save_multiple_images(self, tmp_path):
        """複数の画像が連番で保存される."""
        saver = ImageSaver(tmp_path)

        for _ in range(3):
            saver.save(_create_test_image(), "multi_test")

        saved_files = list((tmp_path / "multi_test").glob("*"))
        assert len(saved_files) == 3

    def test_save_is_thread_safe(self, tmp_path):
        """並行呼び出しでも全画像が重複なく保存される.

        threading.Lock による file_naming_manager 保護を検証する.
        """
        saver = ImageSaver(tmp_path)
        num_threads = 8
        per_thread = 5

        def _worker() -> None:
            """スレッドから複数回 save を呼び出す."""
            for _ in range(per_thread):
                saver.save(_create_test_image(), "concurrent")

        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        saved_files = list((tmp_path / "concurrent").glob("*"))
        # 全件がユニークなファイル名で保存されている.
        assert len(saved_files) == num_threads * per_thread
        assert len({f.name for f in saved_files}) == num_threads * per_thread
