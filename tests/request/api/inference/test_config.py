"""推論設定ファイルのモデルとローダーのテスト."""

import json

import pytest

from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError
from pochivision.request.api.inference.config import (
    InferConfig,
    ResizeConfig,
    load_infer_config,
)


def _write_config(tmp_path, data):
    """テスト用設定ファイルを書き出す."""
    path = tmp_path / "infer_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


class TestLoadInferConfigSuccess:
    """正常系のテスト."""

    def test_full_config(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://192.168.1.100:8000",
                "format": "raw",
                "resize": {
                    "width": 224,
                    "height": 224,
                    "padding_color": [128, 128, 128],
                },
            },
        )
        config = load_infer_config(path)

        assert config.url == "http://192.168.1.100:8000"
        assert config.format == "raw"
        assert config.resize is not None
        assert config.resize.width == 224
        assert config.resize.height == 224
        assert config.resize.padding_color == (128, 128, 128)

    def test_minimal_config(self, tmp_path):
        path = _write_config(tmp_path, {"url": "http://localhost:8000"})
        config = load_infer_config(path)

        assert config.url == "http://localhost:8000"
        assert config.format == "jpeg"
        assert config.resize is None

    def test_resize_without_padding_color(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"width": 320, "height": 240},
            },
        )
        config = load_infer_config(path)

        assert config.resize is not None
        assert config.resize.width == 320
        assert config.resize.height == 240
        assert config.resize.padding_color == (0, 0, 0)

    def test_padding_color_min_boundary(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {
                    "width": 224,
                    "height": 224,
                    "padding_color": [0, 0, 0],
                },
            },
        )
        config = load_infer_config(path)
        assert config.resize is not None
        assert config.resize.padding_color == (0, 0, 0)

    def test_padding_color_max_boundary(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {
                    "width": 224,
                    "height": 224,
                    "padding_color": [255, 255, 255],
                },
            },
        )
        config = load_infer_config(path)
        assert config.resize is not None
        assert config.resize.padding_color == (255, 255, 255)

    def test_format_jpeg_explicit(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "format": "jpeg",
            },
        )
        config = load_infer_config(path)
        assert config.format == "jpeg"

    def test_format_raw(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "format": "raw",
            },
        )
        config = load_infer_config(path)
        assert config.format == "raw"


class TestLoadInferConfigError:
    """異常系のテスト."""

    def test_file_not_found(self):
        with pytest.raises(ConfigLoadError):
            load_infer_config("/nonexistent/path.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ConfigLoadError):
            load_infer_config(str(path))

    def test_missing_url(self, tmp_path):
        path = _write_config(tmp_path, {"format": "jpeg"})
        with pytest.raises(ConfigValidationError, match="url"):
            load_infer_config(path)

    def test_invalid_format(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "format": "png",
            },
        )
        with pytest.raises(ConfigValidationError, match="format"):
            load_infer_config(path)

    def test_resize_missing_width(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"height": 224},
            },
        )
        with pytest.raises(ConfigValidationError, match="width"):
            load_infer_config(path)

    def test_resize_missing_height(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"width": 224},
            },
        )
        with pytest.raises(ConfigValidationError, match="height"):
            load_infer_config(path)

    def test_resize_zero_width(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"width": 0, "height": 224},
            },
        )
        with pytest.raises(ConfigValidationError, match="width"):
            load_infer_config(path)

    def test_resize_negative_width(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"width": -1, "height": 224},
            },
        )
        with pytest.raises(ConfigValidationError, match="width"):
            load_infer_config(path)

    def test_resize_zero_height(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {"width": 224, "height": 0},
            },
        )
        with pytest.raises(ConfigValidationError, match="height"):
            load_infer_config(path)

    def test_resize_invalid_padding_color(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {
                    "width": 224,
                    "height": 224,
                    "padding_color": [256, 0, 0],
                },
            },
        )
        with pytest.raises(ConfigValidationError, match="padding_color"):
            load_infer_config(path)

    def test_resize_padding_color_wrong_length(self, tmp_path):
        path = _write_config(
            tmp_path,
            {
                "url": "http://localhost:8000",
                "resize": {
                    "width": 224,
                    "height": 224,
                    "padding_color": [0, 0],
                },
            },
        )
        with pytest.raises(ConfigValidationError, match="padding_color"):
            load_infer_config(path)


class TestDataclassFrozen:
    """frozen dataclass のテスト."""

    def test_infer_config_frozen(self, tmp_path):
        path = _write_config(tmp_path, {"url": "http://localhost:8000"})
        config = load_infer_config(path)
        with pytest.raises(AttributeError):
            config.url = "http://other:8000"  # type: ignore[misc]

    def test_resize_config_frozen(self):
        resize = ResizeConfig(width=224, height=224)
        with pytest.raises(AttributeError):
            resize.width = 320  # type: ignore[misc]
