"""camera_config_saver のテスト."""

import json
from unittest.mock import MagicMock

from pochivision.capturelib.camera_config_saver import save_camera_config


def _make_mock_cap(
    width: float = 1920.0,
    height: float = 1080.0,
    fps: float = 30.0,
    backend: str = "DSHOW",
    brightness: float = 128.0,
    contrast: float = 32.0,
    exposure: float = -6.0,
) -> MagicMock:
    """テスト用の cv2.VideoCapture モックを生成する."""
    cap = MagicMock()
    cap.getBackendName.return_value = backend

    prop_values = {
        3: width,  # CAP_PROP_FRAME_WIDTH
        4: height,  # CAP_PROP_FRAME_HEIGHT
        5: fps,  # CAP_PROP_FPS
        10: brightness,  # CAP_PROP_BRIGHTNESS
        11: contrast,  # CAP_PROP_CONTRAST
        12: 0.0,  # CAP_PROP_SATURATION
        13: 0.0,  # CAP_PROP_HUE
        14: 0.0,  # CAP_PROP_GAIN
        15: exposure,  # CAP_PROP_EXPOSURE
        44: 0.0,  # CAP_PROP_WB_TEMPERATURE
        20: 0.0,  # CAP_PROP_SHARPNESS
        22: 0.0,  # CAP_PROP_GAMMA
        28: 0.0,  # CAP_PROP_FOCUS
        39: 0.0,  # CAP_PROP_AUTOFOCUS
        21: 0.0,  # CAP_PROP_AUTO_EXPOSURE
        44: 0.0,  # CAP_PROP_AUTO_WB
    }
    cap.get.side_effect = lambda prop_id: prop_values.get(prop_id, 0.0)
    return cap


class TestSaveCameraConfig:
    """save_camera_config のテスト."""

    def test_creates_json_file(self, tmp_path):
        """JSON ファイルが作成される."""
        cap = _make_mock_cap()
        result = save_camera_config(
            cap,
            tmp_path,
            camera_index=0,
            profile_name="0",
            requested_width=1920,
            requested_height=1080,
        )
        assert result.exists()
        assert result.name == "camera_config.json"

    def test_contains_basic_fields(self, tmp_path):
        """基本フィールドが含まれる."""
        cap = _make_mock_cap()
        path = save_camera_config(
            cap,
            tmp_path,
            camera_index=0,
            profile_name="test_profile",
            requested_width=1920,
            requested_height=1080,
        )
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["camera_index"] == 0
        assert data["profile_name"] == "test_profile"
        assert data["backend"] == "DSHOW"
        assert data["requested_width"] == 1920
        assert data["requested_height"] == 1080
        assert data["actual_width"] == 1920
        assert data["actual_height"] == 1080
        assert data["fps"] == 30.0

    def test_contains_camera_properties(self, tmp_path):
        """カメラプロパティが含まれる."""
        cap = _make_mock_cap(brightness=150.0, contrast=50.0, exposure=-4.0)
        path = save_camera_config(
            cap,
            tmp_path,
            camera_index=0,
            profile_name="0",
            requested_width=640,
            requested_height=480,
        )
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["brightness"] == 150.0
        assert data["contrast"] == 50.0
        assert data["exposure"] == -4.0

    def test_requested_vs_actual_differ(self, tmp_path):
        """要求解像度と実際の解像度が異なる場合, 両方保存される."""
        cap = _make_mock_cap(width=1280.0, height=720.0)
        path = save_camera_config(
            cap,
            tmp_path,
            camera_index=0,
            profile_name="0",
            requested_width=1920,
            requested_height=1080,
        )
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data["requested_width"] == 1920
        assert data["requested_height"] == 1080
        assert data["actual_width"] == 1280
        assert data["actual_height"] == 720

    def test_creates_parent_directory(self, tmp_path):
        """親ディレクトリが存在しない場合, 自動作成される."""
        cap = _make_mock_cap()
        nested_dir = tmp_path / "a" / "b"
        result = save_camera_config(
            cap,
            nested_dir,
            camera_index=0,
            profile_name="0",
            requested_width=640,
            requested_height=480,
        )
        assert result.exists()

    def test_valid_json(self, tmp_path):
        """出力が有効な JSON である."""
        cap = _make_mock_cap()
        path = save_camera_config(
            cap,
            tmp_path,
            camera_index=0,
            profile_name="0",
            requested_width=640,
            requested_height=480,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
