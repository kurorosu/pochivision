"""CLAHE（適応的ヒストグラム平坦化）プロセッサーのテスト."""

import numpy as np
import pytest

from exceptions import ProcessorValidationError
from processors.clahe import CLAHEProcessor

# テスト用の画像データ
DUMMY_GRAY_IMAGE = np.ones((100, 100), dtype=np.uint8) * 128
DUMMY_COLOR_IMAGE = np.ones((100, 100, 3), dtype=np.uint8) * 128


def test_clahe_init():
    """初期化テスト."""
    processor = CLAHEProcessor(name="clahe", config={})
    assert processor.name == "clahe"
    assert processor.color_mode == "gray"  # デフォルトモード
    assert processor.clip_limit == 2.0
    assert processor.tile_grid_size == (8, 8)

    # パラメータを指定
    processor = CLAHEProcessor(
        name="clahe",
        config={"color_mode": "lab", "clip_limit": 3.0, "tile_grid_size": [16, 16]},
    )
    assert processor.color_mode == "lab"
    assert processor.clip_limit == 3.0
    assert processor.tile_grid_size == (16, 16)


def test_clahe_grayscale():
    """グレースケール画像の平坦化テスト."""
    processor = CLAHEProcessor(name="clahe", config={})

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


def test_clahe_color_gray_mode():
    """カラー画像の平坦化テスト（grayモード）."""
    processor = CLAHEProcessor(name="clahe", config={"color_mode": "gray"})

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


def test_clahe_color_lab_mode():
    """カラー画像の平坦化テスト（labモード）."""
    processor = CLAHEProcessor(name="clahe", config={"color_mode": "lab"})

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


def test_clahe_color_bgr_mode():
    """カラー画像の平坦化テスト（bgrモード）."""
    processor = CLAHEProcessor(name="clahe", config={"color_mode": "bgr"})

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


def test_clahe_custom_params():
    """カスタムパラメータのテスト."""
    processor = CLAHEProcessor(
        name="clahe",
        config={"color_mode": "lab", "clip_limit": 1.5, "tile_grid_size": [16, 16]},
    )

    # テスト用カラー画像の作成
    image = np.copy(DUMMY_COLOR_IMAGE)
    image[30:70, 30:70, :] = 200

    result = processor.process(image)

    # 結果がnp.ndarrayかつuint8型であることを確認
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


def test_clahe_invalid_color_mode():
    """無効なcolor_modeテスト."""
    # バリデーションエラーになるはず
    with pytest.raises(ProcessorValidationError):
        processor = CLAHEProcessor(name="clahe", config={"color_mode": "invalid"})
        # テスト用カラー画像の作成
        image = np.copy(DUMMY_COLOR_IMAGE)
        processor.process(image)


def test_clahe_invalid_input():
    """無効な入力テスト."""
    processor = CLAHEProcessor(name="clahe", config={})

    # None入力
    with pytest.raises(ProcessorValidationError):
        processor.process(None)

    # 空の画像
    with pytest.raises(ProcessorValidationError):
        processor.process(np.array([]))

    # 無効なデータ型
    with pytest.raises(ProcessorValidationError):
        processor.process(np.ones((10, 10), dtype=np.float32))
