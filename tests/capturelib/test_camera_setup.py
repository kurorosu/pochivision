"""CameraSetup のテスト."""

from unittest.mock import patch

import pytest

from pochivision.capturelib.camera_setup import (
    CameraSetup,
    _get_default_backend,
)
from pochivision.capturelib.log_manager import LogManager
from pochivision.constants import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
)


def _minimal_config() -> dict:
    """テスト用の最小設定を返す."""
    return {
        "cameras": {
            "0": {
                "width": 1920,
                "height": 1080,
                "fps": 60,
                "backend": "DSHOW",
            }
        },
        "selected_camera_index": 0,
    }


class TestCameraSetupLoadConfig:
    """CameraSetup.load_camera_config のテスト."""

    def test_load_valid_config(self):
        """正常な設定を読み込める."""
        config = _minimal_config()
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")
        setup.load_camera_config()

        assert setup.width == 1920
        assert setup.height == 1080
        assert setup.fps == 60
        assert setup.backend == "DSHOW"

    def test_load_missing_profile_uses_defaults(self):
        """存在しないプロファイルの場合デフォルト値を使用する."""
        config = _minimal_config()
        setup = CameraSetup(
            config, LogManager(), camera_index=0, profile_name="nonexistent"
        )
        setup.load_camera_config()

        assert setup.width == DEFAULT_CAMERA_WIDTH
        assert setup.height == DEFAULT_CAMERA_HEIGHT
        assert setup.fps == DEFAULT_CAMERA_FPS

    def test_load_partial_config(self):
        """一部のみ設定されている場合, 残りはデフォルト値."""
        config = {"cameras": {"0": {"width": 800}}}
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")
        setup.load_camera_config()

        assert setup.width == 800
        assert setup.height == DEFAULT_CAMERA_HEIGHT
        assert setup.fps == DEFAULT_CAMERA_FPS

    def test_load_no_cameras_key(self):
        """cameras キーがない場合デフォルト値."""
        config = {}
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")
        setup.load_camera_config()

        assert setup.width == DEFAULT_CAMERA_WIDTH


class TestAutoBackendSelection:
    """OS 自動検出によるバックエンド自動選択のテスト."""

    def test_get_default_backend_windows(self):
        """Windows では DSHOW が返される."""
        with patch(
            "pochivision.capturelib.camera_setup.platform.system",
            return_value="Windows",
        ):
            assert _get_default_backend() == "DSHOW"

    def test_get_default_backend_linux(self):
        """Linux では V4L2 が返される."""
        with patch(
            "pochivision.capturelib.camera_setup.platform.system",
            return_value="Linux",
        ):
            assert _get_default_backend() == "V4L2"

    def test_get_default_backend_darwin(self):
        """macOS では AVFOUNDATION が返される."""
        with patch(
            "pochivision.capturelib.camera_setup.platform.system",
            return_value="Darwin",
        ):
            assert _get_default_backend() == "AVFOUNDATION"

    def test_get_default_backend_unknown_os(self):
        """未知の OS では None が返される."""
        with patch(
            "pochivision.capturelib.camera_setup.platform.system",
            return_value="FreeBSD",
        ):
            assert _get_default_backend() is None

    def test_explicit_backend_takes_priority(self):
        """config で backend を明示指定した場合はそちらが優先される."""
        config = {
            "cameras": {"0": {"backend": "MSMF"}},
        }
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")
        setup.load_camera_config()

        assert setup.backend == "MSMF"

    def test_auto_backend_when_not_specified(self):
        """config で backend 未指定の場合, OS に応じて自動選択される."""
        config = {"cameras": {"0": {"width": 640}}}
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")

        with patch(
            "pochivision.capturelib.camera_setup._get_default_backend",
            return_value="V4L2",
        ):
            setup.load_camera_config()

        assert setup.backend == "V4L2"

    def test_auto_backend_none_on_unknown_os(self):
        """未知の OS では backend が None のまま."""
        config = {"cameras": {"0": {}}}
        setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")

        with patch(
            "pochivision.capturelib.camera_setup._get_default_backend",
            return_value=None,
        ):
            setup.load_camera_config()

        assert setup.backend is None
