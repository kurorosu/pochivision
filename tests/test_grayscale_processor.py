import numpy as np
import pytest  # noqa: F401

from processors.grayscale import GrayscaleProcessor

# テスト用の画像データ
DUMMY_COLOR = np.ones((10, 10, 3), dtype=np.uint8) * 100


def test_grayscale_valid():
    """カラー画像からグレースケールへの変換をテスト"""
    processor = GrayscaleProcessor(name="grayscale", config={})
    result = processor.process(DUMMY_COLOR)

    # チャンネル数が減っていることを確認（3→1）
    assert result.ndim == 2
    # 元のサイズ（高さ・幅）が維持されていることを確認
    assert result.shape == (10, 10)
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8
