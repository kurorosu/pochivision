import numpy as np
import pytest

from pochivision.processors.resize import ResizeProcessor

# テスト用の画像データ
DUMMY_IMAGE = np.ones((300, 400, 3), dtype=np.uint8) * 100


def test_resize_no_aspect_ratio():
    """アスペクト比を維持しないリサイズのテスト."""
    config = {"width": 200, "height": 200, "preserve_aspect_ratio": False}
    processor = ResizeProcessor(name="resize", config=config)

    result = processor.process(DUMMY_IMAGE)

    # リサイズ後のサイズを確認
    assert result.shape == (200, 200, 3)
    # データ型が維持されていることを確認
    assert result.dtype == np.uint8


def test_resize_preserve_aspect_ratio_width():
    """幅を基準にアスペクト比を維持するリサイズのテスト."""
    config = {"width": 200, "preserve_aspect_ratio": True, "aspect_ratio_mode": "width"}
    processor = ResizeProcessor(name="resize", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 元のアスペクト比は 400:300 = 4:3
    # 幅200に対する高さは 200 * (300/400) = 150
    assert result.shape == (150, 200, 3)
    assert result.dtype == np.uint8


def test_resize_preserve_aspect_ratio_height():
    """高さを基準にアスペクト比を維持するリサイズのテスト."""
    config = {
        "height": 150,
        "preserve_aspect_ratio": True,
        "aspect_ratio_mode": "height",
    }
    processor = ResizeProcessor(name="resize", config=config)

    result = processor.process(DUMMY_IMAGE)

    # 元のアスペクト比は 400:300 = 4:3
    # 高さ150に対する幅は 150 * (400/300) = 200
    assert result.shape == (150, 200, 3)
    assert result.dtype == np.uint8


def test_resize_width_only():
    """幅のみ指定したリサイズのテスト."""
    config = {"width": 200}
    processor = ResizeProcessor(name="resize", config=config)

    result = processor.process(DUMMY_IMAGE)

    # デフォルトでアスペクト比保持、元画像400x300、幅200指定
    # アスペクト比保持で高さは 200 * (300/400) = 150
    assert result.shape == (150, 200, 3)
    assert result.dtype == np.uint8


def test_resize_height_only():
    """高さのみ指定したリサイズのテスト."""
    config = {"height": 150}
    processor = ResizeProcessor(name="resize", config=config)

    result = processor.process(DUMMY_IMAGE)

    # デフォルトでアスペクト比保持、aspect_ratio_mode="width"
    # 高さのみ指定でもwidthモードなので、デフォルト幅1600が適用される
    # 幅1600でアスペクト比保持すると高さは 1600 * (300/400) = 1200
    assert result.shape == (1200, 1600, 3)
    assert result.dtype == np.uint8


def test_resize_validation_error():
    """パラメータバリデーションのテスト.

    width/height 未指定や負の値のチェックは Pydantic スキーマの対象外のため,
    スキーマで検出可能な aspect_ratio_mode の不正値のみテストする.
    """
    from pochivision.processors.registry import get_processor

    # aspect_ratio_mode が不正な場合 (pattern 違反)
    with pytest.raises(ValueError):
        get_processor("resize", {"width": 100, "aspect_ratio_mode": "invalid"})
