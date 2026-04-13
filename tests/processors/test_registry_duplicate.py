"""processors/registry.py の重複登録チェックに関するテスト."""

from collections.abc import Generator

import numpy as np
import pytest

from pochivision.exceptions import ProcessorRegistrationError
from pochivision.processors.base import BaseProcessor
from pochivision.processors.registry import PROCESSOR_REGISTRY, register_processor


class _DummyProcessorA(BaseProcessor):
    """テスト用ダミープロセッサ A."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """入力画像をそのまま返す."""
        return image

    @staticmethod
    def get_default_config() -> dict:
        """デフォルト設定を返す."""
        return {}


class _DummyProcessorB(BaseProcessor):
    """テスト用ダミープロセッサ B."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """入力画像をそのまま返す."""
        return image

    @staticmethod
    def get_default_config() -> dict:
        """デフォルト設定を返す."""
        return {}


@pytest.fixture
def temp_name() -> Generator[str, None, None]:
    """登録名を一意にし, テスト後にレジストリから除去するフィクスチャ."""
    name = "__test_dup_processor__"
    PROCESSOR_REGISTRY.pop(name, None)
    yield name
    PROCESSOR_REGISTRY.pop(name, None)


def test_register_processor_first_time_succeeds(temp_name: str) -> None:
    """初回登録は成功する."""
    register_processor(temp_name)(_DummyProcessorA)
    assert PROCESSOR_REGISTRY[temp_name] is _DummyProcessorA


def test_register_processor_duplicate_raises(temp_name: str) -> None:
    """重複登録時に ProcessorRegistrationError が送出される."""
    register_processor(temp_name)(_DummyProcessorA)
    with pytest.raises(ProcessorRegistrationError) as exc_info:
        register_processor(temp_name)(_DummyProcessorB)
    assert temp_name in str(exc_info.value)
    # 既存登録は維持される
    assert PROCESSOR_REGISTRY[temp_name] is _DummyProcessorA


def test_register_processor_override_true_overwrites(temp_name: str) -> None:
    """override=True を指定すると上書きが許可される."""
    register_processor(temp_name)(_DummyProcessorA)
    register_processor(temp_name, override=True)(_DummyProcessorB)
    assert PROCESSOR_REGISTRY[temp_name] is _DummyProcessorB
