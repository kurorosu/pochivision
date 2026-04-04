"""CameraSetup のテスト."""

import pytest

from pochivision.capturelib.camera_setup import CameraSetup
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
