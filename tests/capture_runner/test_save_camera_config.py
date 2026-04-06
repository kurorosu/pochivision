"""_save_camera_config のテスト."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from pochivision.capture_runner.viewer import LivePreviewRunner
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.log_manager import LogManager


def _make_runner(
    tmp_path: Path,
    camera_setup: CameraSetup | None = None,
) -> LivePreviewRunner:
    """テスト用の LivePreviewRunner を生成する."""
    cap = MagicMock()
    cap.getBackendName.return_value = "DSHOW"
    cap.get.return_value = 0.0

    pipeline = MagicMock()
    pipeline.output_dir = tmp_path

    return LivePreviewRunner(cap, pipeline, camera_setup=camera_setup)


def _make_camera_setup() -> CameraSetup:
    """テスト用の CameraSetup を生成する."""
    config = {"cameras": {"0": {"width": 1920, "height": 1080, "fps": 30}}}
    setup = CameraSetup(config, LogManager(), camera_index=0, profile_name="0")
    setup.load_camera_config()
    return setup


class TestSaveCameraConfigViewer:
    """_save_camera_config のテスト."""

    def test_saves_config(self, tmp_path):
        """camera_setup が設定されている場合, JSON が保存される."""
        setup = _make_camera_setup()
        runner = _make_runner(tmp_path, camera_setup=setup)

        runner._save_camera_config()

        config_path = tmp_path / "camera_config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["camera_index"] == 0
        assert data["profile_name"] == "0"

    def test_skips_when_no_camera_setup(self, tmp_path):
        """camera_setup=None の場合, 保存されない."""
        runner = _make_runner(tmp_path, camera_setup=None)

        runner._save_camera_config()

        config_path = tmp_path / "camera_config.json"
        assert not config_path.exists()

    def test_oserror_does_not_raise(self, tmp_path):
        """保存失敗時に例外が発生しない."""
        setup = _make_camera_setup()
        runner = _make_runner(tmp_path, camera_setup=setup)
        runner.pipeline.output_dir = "Z:\\nonexistent\\path"

        runner._save_camera_config()

    def test_requested_width_from_config(self, tmp_path):
        """requested_width/height が設定ファイルの要求値である."""
        setup = _make_camera_setup()
        # simulate actual != requested
        setup.width = 1280
        setup.height = 720

        runner = _make_runner(tmp_path, camera_setup=setup)
        runner._save_camera_config()

        config_path = tmp_path / "camera_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["requested_width"] == 1920
        assert data["requested_height"] == 1080
