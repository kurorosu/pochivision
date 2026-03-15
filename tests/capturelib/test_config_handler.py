"""ConfigHandler / CameraConfigHandler のテスト."""

import json

import pytest

from capturelib.config_handler import (
    CameraConfigError,
    CameraConfigHandler,
    ConfigHandler,
    ConfigLoadError,
)
from exceptions.config import ConfigValidationError


def _minimal_config() -> dict:
    """バリデーションが通る最小限の設定を返す."""
    return {
        "cameras": {
            "0": {
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "backend": "DSHOW",
                "label": "Test_Cam",
                "processors": ["resize"],
                "mode": "pipeline",
                "resize": {
                    "width": 800,
                    "preserve_aspect_ratio": True,
                    "aspect_ratio_mode": "width",
                },
            }
        },
        "recording": {
            "select_format": "mjpg",
            "available_formats": {
                "mjpg": "Motion JPEG",
            },
        },
        "selected_camera_index": 0,
        "id_interval": 1,
        "preview": {"width": 1280, "height": 720},
    }


class TestConfigHandlerLoad:
    """ConfigHandler.load のテスト."""

    def test_load_valid_config(self, tmp_path):
        """正常な設定ファイルを読み込める."""
        config = _minimal_config()
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")

        result = ConfigHandler.load(str(config_file))

        assert "cameras" in result
        assert "0" in result["cameras"]

    def test_load_file_not_found(self, tmp_path):
        """存在しないファイルで ConfigLoadError."""
        with pytest.raises(ConfigLoadError, match="not found"):
            ConfigHandler.load(str(tmp_path / "nonexistent.json"))

    def test_load_invalid_json(self, tmp_path):
        """不正な JSON で ConfigLoadError."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="decode"):
            ConfigHandler.load(str(config_file))

    def test_load_validation_error(self, tmp_path):
        """スキーマ違反で ConfigValidationError."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"invalid": "data"}), encoding="utf-8")

        with pytest.raises(ConfigValidationError, match="バリデーション"):
            ConfigHandler.load(str(config_file))


class TestConfigHandlerSave:
    """ConfigHandler.save のテスト."""

    def test_save_creates_file(self, tmp_path):
        """設定をファイルに保存できる."""
        config = {"cameras": {}, "test_key": "test_value"}
        ConfigHandler.save(config, tmp_path)

        saved_files = list(tmp_path.glob("*_config.json"))
        assert len(saved_files) == 1

        with open(saved_files[0], encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["test_key"] == "test_value"
        assert "timestamp" in saved


class TestCameraConfigHandlerGetCameraConfig:
    """CameraConfigHandler.get_camera_config のテスト."""

    def test_get_camera_config_by_index(self):
        """カメラインデックスで設定を取得できる."""
        config = _minimal_config()
        result = CameraConfigHandler.get_camera_config(config, 0)

        assert result["width"] == 1920
        assert result["height"] == 1080

    def test_get_camera_config_no_cameras_key(self):
        """cameras キーがない場合 CameraConfigError."""
        with pytest.raises(CameraConfigError, match="No camera configurations"):
            CameraConfigHandler.get_camera_config({}, 0)

    def test_get_camera_config_missing_index(self):
        """指定インデックスが存在しない場合 CameraConfigError."""
        config = _minimal_config()
        with pytest.raises(CameraConfigError, match="No configuration found"):
            CameraConfigHandler.get_camera_config(config, 99)

    def test_get_camera_config_none_index_uses_selected(self):
        """camera_index=None で selected_camera_index を使用する."""
        config = _minimal_config()
        result = CameraConfigHandler.get_camera_config(config, None)

        assert result["width"] == 1920

    def test_get_camera_config_none_index_no_selected(self):
        """camera_index=None かつ selected_camera_index なしで CameraConfigError."""
        config = _minimal_config()
        del config["selected_camera_index"]

        with pytest.raises(CameraConfigError, match="No selected camera index"):
            CameraConfigHandler.get_camera_config(config, None)


class TestCameraConfigHandlerGetProcessors:
    """CameraConfigHandler.get_camera_processors のテスト."""

    def test_get_processors(self):
        """プロセッサ設定を取得できる."""
        config = _minimal_config()
        processors, proc_configs, mode = CameraConfigHandler.get_camera_processors(
            config, "0"
        )

        assert processors == ["resize"]
        assert "resize" in proc_configs
        assert mode == "pipeline"

    def test_get_processors_no_cameras(self):
        """cameras キーがない場合 CameraConfigError."""
        with pytest.raises(CameraConfigError, match="No camera configurations"):
            CameraConfigHandler.get_camera_processors({}, "0")

    def test_get_processors_missing_profile(self):
        """存在しないプロファイル名で CameraConfigError."""
        config = _minimal_config()
        with pytest.raises(CameraConfigError, match="No configuration found"):
            CameraConfigHandler.get_camera_processors(config, "nonexistent")

    def test_get_processors_no_processors_key(self):
        """プロファイルに processors キーがない場合 CameraConfigError."""
        config = _minimal_config()
        del config["cameras"]["0"]["processors"]

        with pytest.raises(CameraConfigError, match="No processors defined"):
            CameraConfigHandler.get_camera_processors(config, "0")

    def test_get_processors_empty_list(self):
        """processors が空リストの場合 CameraConfigError."""
        config = _minimal_config()
        config["cameras"]["0"]["processors"] = []

        with pytest.raises(CameraConfigError, match="Empty processors list"):
            CameraConfigHandler.get_camera_processors(config, "0")

    def test_get_processors_default_mode_parallel(self):
        """mode 未指定の場合デフォルトで parallel."""
        config = _minimal_config()
        del config["cameras"]["0"]["mode"]

        _, _, mode = CameraConfigHandler.get_camera_processors(config, "0")
        assert mode == "parallel"


class TestCameraConfigHandlerUtilities:
    """ユーティリティメソッドのテスト."""

    def test_get_all_camera_indices(self):
        """全カメラインデックスを取得できる."""
        config = _minimal_config()
        indices = CameraConfigHandler.get_all_camera_indices(config)

        assert indices == [0]

    def test_get_all_camera_indices_no_cameras(self):
        """cameras キーがない場合 CameraConfigError."""
        with pytest.raises(CameraConfigError):
            CameraConfigHandler.get_all_camera_indices({})

    def test_get_selected_camera_index(self):
        """選択されたカメラインデックスを取得できる."""
        config = _minimal_config()
        index = CameraConfigHandler.get_selected_camera_index(config)

        assert index == 0

    def test_get_selected_camera_index_fallback(self):
        """selected_camera_index なしの場合, 最初のカメラを返す."""
        config = _minimal_config()
        del config["selected_camera_index"]

        index = CameraConfigHandler.get_selected_camera_index(config)
        assert index == 0

    def test_get_selected_camera_index_no_cameras(self):
        """カメラ設定がない場合 CameraConfigError."""
        with pytest.raises(CameraConfigError, match="No selected camera index"):
            CameraConfigHandler.get_selected_camera_index({})
