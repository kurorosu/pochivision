"""RecordingManager / VideoFormat のテスト."""

import pytest

from pochivision.capturelib.recording_manager import RecordingManager, VideoFormat


class TestVideoFormat:
    """VideoFormat のテスト."""

    def test_get_available_formats(self):
        """利用可能な形式一覧を取得できる."""
        formats = VideoFormat.get_available_formats()
        assert isinstance(formats, dict)
        assert "mjpg" in formats
        assert "mp4v" in formats
        assert len(formats) == 6

    def test_get_format_info_valid(self):
        """有効な形式の情報を取得できる."""
        info = VideoFormat.get_format_info("mjpg")
        assert info is not None
        fourcc, ext, description = info
        assert fourcc == "MJPG"
        assert ext == ".avi"

    def test_get_format_info_case_insensitive(self):
        """大文字小文字を区別しない."""
        info = VideoFormat.get_format_info("MJPG")
        assert info is not None

    def test_get_format_info_invalid(self):
        """無効な形式で None を返す."""
        info = VideoFormat.get_format_info("nonexistent")
        assert info is None


class TestRecordingManagerInit:
    """RecordingManager の初期化テスト."""

    def test_default_format(self):
        """デフォルト形式で初期化できる."""
        manager = RecordingManager()
        assert manager.video_format == "mjpg"
        assert manager.is_recording is False
        assert manager.video_writer is None

    def test_custom_format(self):
        """指定形式で初期化できる."""
        manager = RecordingManager(default_format="mp4v")
        assert manager.video_format == "mp4v"

    def test_invalid_format_fallback(self):
        """無効な形式の場合 mjpg にフォールバックする."""
        manager = RecordingManager(default_format="invalid_format")
        assert manager.video_format == "mjpg"

    def test_get_current_format(self):
        """現在の形式を取得できる."""
        manager = RecordingManager(default_format="ffv1")
        assert manager.get_current_format() == "ffv1"


class TestRecordingManagerState:
    """RecordingManager の状態管理テスト."""

    def test_initial_recording_status(self):
        """初期状態は録画停止中."""
        manager = RecordingManager()
        assert manager.get_recording_status() is False

    def test_stop_recording_when_not_recording(self):
        """録画中でない場合に停止すると False を返す."""
        manager = RecordingManager()
        assert manager.stop_recording() is False

    def test_start_recording_already_recording(self, tmp_path):
        """録画中に開始すると False を返す."""
        manager = RecordingManager()
        manager.is_recording = True
        assert manager.start_recording(tmp_path) is False

    def test_cleanup_without_recording(self):
        """録画していない状態でクリーンアップが正常に完了する."""
        manager = RecordingManager()
        manager.cleanup()
        assert manager.video_writer is None
        assert manager.is_recording is False

    def test_add_frame_when_not_recording(self):
        """録画中でない場合にフレーム追加すると False を返す."""
        import numpy as np

        manager = RecordingManager()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert manager.add_frame(frame) is False
