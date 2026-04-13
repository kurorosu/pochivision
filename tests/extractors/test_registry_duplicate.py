"""feature_extractors/registry.py の重複登録チェックに関するテスト."""

from collections.abc import Generator

import numpy as np
import pytest

from pochivision.exceptions import ExtractorRegistrationError
from pochivision.feature_extractors.base import BaseFeatureExtractor
from pochivision.feature_extractors.registry import (
    FEATURE_EXTRACTOR_REGISTRY,
    register_feature_extractor,
)


class _DummyExtractorA(BaseFeatureExtractor):
    """テスト用ダミー特徴量抽出器 A."""

    def extract(self, image: np.ndarray) -> dict[str, float | int]:
        """空辞書を返す."""
        return {}

    @staticmethod
    def get_default_config() -> dict:
        """デフォルト設定を返す."""
        return {}

    @staticmethod
    def get_feature_names() -> list[str]:
        """特徴量名リストを返す."""
        return []


class _DummyExtractorB(BaseFeatureExtractor):
    """テスト用ダミー特徴量抽出器 B."""

    def extract(self, image: np.ndarray) -> dict[str, float | int]:
        """空辞書を返す."""
        return {}

    @staticmethod
    def get_default_config() -> dict:
        """デフォルト設定を返す."""
        return {}

    @staticmethod
    def get_feature_names() -> list[str]:
        """特徴量名リストを返す."""
        return []


@pytest.fixture
def temp_name() -> Generator[str, None, None]:
    """登録名を一意にし, テスト後にレジストリから除去するフィクスチャ."""
    name = "__test_dup_extractor__"
    FEATURE_EXTRACTOR_REGISTRY.pop(name, None)
    yield name
    FEATURE_EXTRACTOR_REGISTRY.pop(name, None)


def test_register_extractor_first_time_succeeds(temp_name: str) -> None:
    """初回登録は成功する."""
    register_feature_extractor(temp_name)(_DummyExtractorA)
    assert FEATURE_EXTRACTOR_REGISTRY[temp_name] is _DummyExtractorA


def test_register_extractor_duplicate_raises(temp_name: str) -> None:
    """重複登録時に ExtractorRegistrationError が送出される."""
    register_feature_extractor(temp_name)(_DummyExtractorA)
    with pytest.raises(ExtractorRegistrationError) as exc_info:
        register_feature_extractor(temp_name)(_DummyExtractorB)
    assert temp_name in str(exc_info.value)
    assert FEATURE_EXTRACTOR_REGISTRY[temp_name] is _DummyExtractorA


def test_register_extractor_override_true_overwrites(temp_name: str) -> None:
    """override=True を指定すると上書きが許可される."""
    register_feature_extractor(temp_name)(_DummyExtractorA)
    register_feature_extractor(temp_name, override=True)(_DummyExtractorB)
    assert FEATURE_EXTRACTOR_REGISTRY[temp_name] is _DummyExtractorB
