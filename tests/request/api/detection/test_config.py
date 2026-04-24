"""DetectConfig / load_detect_config のテスト."""

import json
from pathlib import Path

import pytest

from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError
from pochivision.request.api.detection.config import (
    DetectConfig,
    load_detect_config,
)


def _write_config(tmp_path: Path, data: dict) -> Path:
    """テスト用の JSON 設定ファイルを書き出す."""
    path = tmp_path / "detect_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadDetectConfig:
    """load_detect_config のテスト."""

    def test_minimal_config(self, tmp_path):
        path = _write_config(tmp_path, {"base_url": "http://localhost:8000"})
        config = load_detect_config(str(path))

        assert isinstance(config, DetectConfig)
        assert config.base_url == "http://localhost:8000"
        assert config.image_format == "raw"
        assert config.score_threshold == 0.5
        assert config.timeout == 5.0
        assert config.jpeg_quality == 90
        assert config.detect_fps == 5.0

    def test_full_config(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "base_url": "http://localhost:9000",
                "image_format": "raw",
                "score_threshold": 0.3,
                "timeout": 10.0,
                "jpeg_quality": 75,
            },
        )
        config = load_detect_config(str(path))

        assert config.base_url == "http://localhost:9000"
        assert config.image_format == "raw"
        assert config.score_threshold == 0.3
        assert config.timeout == 10.0
        assert config.jpeg_quality == 75

    def test_missing_base_url_raises(self, tmp_path):
        path = _write_config(tmp_path, {"image_format": "jpeg"})
        with pytest.raises(ConfigValidationError, match="base_url"):
            load_detect_config(str(path))

    def test_invalid_url_scheme_raises(self, tmp_path):
        path = _write_config(tmp_path, {"base_url": "localhost:8000"})
        with pytest.raises(ConfigValidationError, match="http"):
            load_detect_config(str(path))

    def test_non_string_url_raises(self, tmp_path):
        path = _write_config(tmp_path, {"base_url": 12345})
        with pytest.raises(ConfigValidationError, match="http"):
            load_detect_config(str(path))

    def test_invalid_image_format_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "image_format": "png"}
        )
        with pytest.raises(ConfigValidationError, match="image_format"):
            load_detect_config(str(path))

    def test_invalid_score_threshold_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "score_threshold": 1.5}
        )
        with pytest.raises(ConfigValidationError, match="score_threshold"):
            load_detect_config(str(path))

    def test_invalid_timeout_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "timeout": -1}
        )
        with pytest.raises(ConfigValidationError, match="timeout"):
            load_detect_config(str(path))

    def test_invalid_jpeg_quality_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "jpeg_quality": 0}
        )
        with pytest.raises(ConfigValidationError, match="jpeg_quality"):
            load_detect_config(str(path))

    def test_jpeg_quality_over_100_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "jpeg_quality": 101}
        )
        with pytest.raises(ConfigValidationError, match="jpeg_quality"):
            load_detect_config(str(path))

    def test_score_threshold_boundaries_valid(self, tmp_path):
        for v in (0.0, 1.0):
            path = _write_config(
                tmp_path, {"base_url": "http://localhost:8000", "score_threshold": v}
            )
            assert load_detect_config(str(path)).score_threshold == v

    def test_timeout_zero_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "timeout": 0}
        )
        with pytest.raises(ConfigValidationError, match="timeout"):
            load_detect_config(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            load_detect_config(str(tmp_path / "nonexistent.json"))

    def test_detect_fps_custom(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"base_url": "http://localhost:8000", "detect_fps": 10.0},
        )
        config = load_detect_config(str(path))
        assert config.detect_fps == 10.0

    def test_legacy_mode_key_ignored_with_warning(self, tmp_path, caplog):
        """廃止された `mode` キーが JSON に残っていても warning を出して無視する."""
        path = _write_config(
            tmp_path,
            {"base_url": "http://localhost:8000", "mode": "detect"},
        )
        with caplog.at_level("WARNING"):
            config = load_detect_config(str(path))

        # mode 属性は DetectConfig に存在しない
        assert not hasattr(config, "mode")
        # 廃止アナウンスの warning が記録されている
        assert any(
            "mode" in rec.message and "--detect" in rec.message
            for rec in caplog.records
        )

    def test_detect_fps_zero_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "detect_fps": 0}
        )
        with pytest.raises(ConfigValidationError, match="detect_fps"):
            load_detect_config(str(path))

    def test_detect_fps_negative_raises(self, tmp_path):
        path = _write_config(
            tmp_path, {"base_url": "http://localhost:8000", "detect_fps": -1.0}
        )
        with pytest.raises(ConfigValidationError, match="detect_fps"):
            load_detect_config(str(path))

    def test_metrics_interval_default_disabled(self, tmp_path):
        path = _write_config(tmp_path, {"base_url": "http://localhost:8000"})
        config = load_detect_config(str(path))
        assert config.metrics_interval_s == 0.0

    def test_metrics_interval_custom(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"base_url": "http://localhost:8000", "metrics_interval_s": 2.5},
        )
        config = load_detect_config(str(path))
        assert config.metrics_interval_s == 2.5

    def test_metrics_interval_negative_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            {"base_url": "http://localhost:8000", "metrics_interval_s": -1.0},
        )
        with pytest.raises(ConfigValidationError, match="metrics_interval_s"):
            load_detect_config(str(path))


class TestDetectConfig:
    """DetectConfig のテスト."""

    def test_frozen(self):
        config = DetectConfig(base_url="http://localhost:8000")
        with pytest.raises(Exception):
            config.base_url = "http://other:8000"  # type: ignore[misc]
