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


def test_resize_preserve_aspect_ratio_width_rounding():
    """幅基準のアスペクト比保持で端数が四捨五入されることを検証.

    元画像 1599x900 を width=800 で縮小した場合,
    高さは 800 * 900 / 1599 = 450.2814... となり, 切り捨て (int) でも四捨五入でも 450.
    差が出るケースとして 1601x900 -> height=450 を別テストで検証する.

    ここでは端数 .5 以上のケースを直接確認する.
    元画像 1000x333 を width=600 で縮小すると
    target_h = 600 * 333 / 1000 = 199.8 -> 切り捨て 199, 四捨五入 200.
    """
    img = np.ones((333, 1000, 3), dtype=np.uint8) * 100
    proc = ResizeProcessor(
        name="resize",
        config={
            "width": 600,
            "preserve_aspect_ratio": True,
            "aspect_ratio_mode": "width",
        },
    )
    result = proc.process(img)
    # 600 * 333 / 1000 = 199.8 -> round -> 200 (切り捨てだと 199)
    assert result.shape == (200, 600, 3)


def test_resize_preserve_aspect_ratio_height_rounding():
    """高さ基準のアスペクト比保持で端数が四捨五入されることを検証.

    元画像 1599x900 を height=450 で縮小した場合,
    幅は 450 * 1599 / 900 = 799.5 となり, 切り捨てだと 799, 四捨五入だと 800 となる.
    """
    img = np.ones((900, 1599, 3), dtype=np.uint8) * 100
    proc = ResizeProcessor(
        name="resize",
        config={
            "height": 450,
            "preserve_aspect_ratio": True,
            "aspect_ratio_mode": "height",
        },
    )
    result = proc.process(img)
    # 450 * 1599 / 900 = 799.5 -> round -> 800 (切り捨てなら 799)
    assert result.shape == (450, 800, 3)


def test_resize_validation_error():
    """パラメータバリデーションのテスト.

    width/height 未指定や負の値のチェックは Pydantic スキーマの対象外のため,
    スキーマで検出可能な aspect_ratio_mode の不正値のみテストする.
    """
    from pochivision.processors.registry import get_processor

    # aspect_ratio_mode が不正な場合 (pattern 違反)
    with pytest.raises(ValueError):
        get_processor("resize", {"width": 100, "aspect_ratio_mode": "invalid"})
