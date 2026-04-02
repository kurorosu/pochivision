"""ヒストグラム平坦化プロセッサーのテスト."""

import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.equalize import EqualizeProcessor

# テスト用の画像データ
DUMMY_GRAY_IMAGE = np.ones((100, 100), dtype=np.uint8) * 128
DUMMY_COLOR_IMAGE = np.ones((100, 100, 3), dtype=np.uint8) * 128


def test_equalize_init():
    """初期化テスト."""
    processor = EqualizeProcessor(name="equalize", config={})
    assert processor.name == "equalize"
    assert processor.color_mode == "gray"  # デフォルトモード

    # color_modeを指定
    processor = EqualizeProcessor(name="equalize", config={"color_mode": "lab"})
    assert processor.color_mode == "lab"


def test_equalize_grayscale():
    """グレースケール画像の平坦化テスト."""
    processor = EqualizeProcessor(name="equalize", config={})

    # テスト用グレースケール画像の作成
    image = np.copy(DUMMY_GRAY_IMAGE)
    # 異なる値を持つ領域を作成
    image[30:70, 30:70] = 200

    result = processor.process(image)

    # 結果がnp.ndarrayかつuint8型であることを確認
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8

    # 形状が維持されている
    assert result.shape == image.shape

    # ヒストグラム平坦化により値が変化している
    assert not np.array_equal(result, image)


def test_equalize_color_gray_mode():
    """カラー画像の平坦化テスト（grayモード）."""
    processor = EqualizeProcessor(name="equalize", config={"color_mode": "gray"})

    # テスト用カラー画像の作成
    image = np.copy(DUMMY_COLOR_IMAGE)
    # 異なる値を持つ領域を作成
    image[30:70, 30:70, :] = 200

    result = processor.process(image)

    # 結果がnp.ndarrayかつuint8型であることを確認
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8

    # カラー画像の場合、チャンネル数は維持される
    assert len(result.shape) == 3
    assert result.shape[2] == 3

    # ヒストグラム平坦化により値が変化している
    assert not np.array_equal(result, image)


def test_equalize_color_lab_mode():
    """カラー画像の平坦化テスト（labモード）."""
    processor = EqualizeProcessor(name="equalize", config={"color_mode": "lab"})

    # テスト用カラー画像の作成
    image = np.copy(DUMMY_COLOR_IMAGE)
    # 異なる値を持つ領域を作成
    image[30:70, 30:70, :] = 200

    result = processor.process(image)

    # 結果がnp.ndarrayかつuint8型であることを確認
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8

    # カラー画像の場合、チャンネル数は維持される
    assert len(result.shape) == 3
    assert result.shape[2] == 3

    # ヒストグラム平坦化により値が変化している
    assert not np.array_equal(result, image)


def test_equalize_color_bgr_mode():
    """カラー画像の平坦化テスト（bgrモード）."""
    processor = EqualizeProcessor(name="equalize", config={"color_mode": "bgr"})

    # テスト用カラー画像の作成
    image = np.copy(DUMMY_COLOR_IMAGE)
    # 異なる値を持つ領域を作成
    image[30:70, 30:70, :] = 200

    result = processor.process(image)

    # 結果がnp.ndarrayかつuint8型であることを確認
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8

    # カラー画像の場合、チャンネル数は維持される
    assert len(result.shape) == 3
    assert result.shape[2] == 3

    # ヒストグラム平坦化により値が変化している
    assert not np.array_equal(result, image)


def test_equalize_invalid_color_mode():
    """無効なcolor_modeテスト."""
    from pochivision.processors.registry import get_processor

    with pytest.raises(ValueError):
        get_processor("equalize", {"color_mode": "invalid"})


def test_equalize_invalid_input():
    """無効な入力テスト."""
    processor = EqualizeProcessor(name="equalize", config={})

    # None入力
    with pytest.raises(ProcessorValidationError):
        processor.process(None)

    # 空の画像
    with pytest.raises(ProcessorValidationError):
        processor.process(np.array([]))

    # 無効なデータ型
    with pytest.raises(ProcessorValidationError):
        processor.process(np.ones((10, 10), dtype=np.float32))
