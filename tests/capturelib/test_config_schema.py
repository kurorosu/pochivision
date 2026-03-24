"""PreviewConfig のバリデーションテスト."""

import pytest
from pydantic import ValidationError

from pochivision.capturelib.config_schema import PreviewConfig


def test_preview_config_defaults():
    """preview 未設定時のデフォルト値を検証."""
    config = PreviewConfig()
    assert config.width == 1280
    assert config.height == 720


def test_preview_config_custom():
    """カスタム値の設定を検証."""
    config = PreviewConfig(width=1920, height=1080)
    assert config.width == 1920
    assert config.height == 1080


def test_preview_config_invalid_zero_width():
    """width が 0 でバリデーションエラーが発生することを検証."""
    with pytest.raises(ValidationError):
        PreviewConfig(width=0, height=480)


def test_preview_config_invalid_negative_height():
    """height が負の値でバリデーションエラーが発生することを検証."""
    with pytest.raises(ValidationError):
        PreviewConfig(width=640, height=-1)
