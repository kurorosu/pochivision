"""CLAHE（適応的ヒストグラム平坦化）プロセッサーのテスト."""

import cv2
import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.clahe import CLAHEProcessor

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


def test_clahe_shape_hw1():
    """shape (H, W, 1) の 1 チャンネル画像の CLAHE テスト."""
    processor = CLAHEProcessor(name="clahe", config={})

    # (H, W, 1) 形状のダミー画像を作成.
    image = np.ones((100, 100, 1), dtype=np.uint8) * 128
    image[30:70, 30:70, 0] = 200

    result = processor.process(image)

    # 出力は入力と同じ 3 次元形状 (H, W, 1) を保つ.
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == image.shape

    # CLAHE 適用結果が 2D apply と同値である (squeeze して比較).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    expected = clahe.apply(image.squeeze(axis=2))
    assert np.array_equal(result.squeeze(axis=2), expected)


def test_clahe_shape_hw1_matches_2d():
    """(H, W, 1) と (H, W) の出力値が一致することを確認."""
    processor = CLAHEProcessor(name="clahe", config={})

    image_2d = np.copy(DUMMY_GRAY_IMAGE)
    image_2d[30:70, 30:70] = 200
    image_3d = image_2d[:, :, np.newaxis].copy()

    result_2d = processor.process(image_2d)
    result_3d = processor.process(image_3d)

    assert np.array_equal(result_2d, result_3d.squeeze(axis=2))


def test_clahe_invalid_color_mode():
    """無効なcolor_modeテスト."""
    from pochivision.processors.registry import get_processor

    with pytest.raises(ValueError):
        get_processor("clahe", {"color_mode": "invalid"})


def test_clahe_update_params_clip_limit_only():
    """update_params で clip_limit だけを更新できる."""
    processor = CLAHEProcessor(name="clahe", config={})
    assert processor.clip_limit == 2.0
    assert processor.tile_grid_size == (8, 8)

    processor.update_params(clip_limit=4.0)

    assert processor.clip_limit == 4.0
    # tile_grid_size は維持される.
    assert processor.tile_grid_size == (8, 8)
    # config も同期されている.
    assert processor.config["clip_limit"] == 4.0
    assert processor.config["tile_grid_size"] == [8, 8]


def test_clahe_update_params_tile_grid_only():
    """update_params で tile_grid_size だけを更新できる."""
    processor = CLAHEProcessor(name="clahe", config={})

    processor.update_params(tile_grid_size=[16, 16])

    assert processor.clip_limit == 2.0
    assert processor.tile_grid_size == (16, 16)
    assert processor.config["tile_grid_size"] == [16, 16]


def test_clahe_update_params_both():
    """update_params で両方を同時に更新できる."""
    processor = CLAHEProcessor(name="clahe", config={})

    processor.update_params(clip_limit=3.5, tile_grid_size=(4, 4))

    assert processor.clip_limit == 3.5
    assert processor.tile_grid_size == (4, 4)


def test_clahe_update_params_reflected_in_process():
    """update_params 後の process() が新パラメータで動作する."""
    processor = CLAHEProcessor(name="clahe", config={})

    image = np.copy(DUMMY_GRAY_IMAGE)
    image[30:70, 30:70] = 200

    # 更新前の結果.
    result_before = processor.process(image)

    # パラメータ更新.
    processor.update_params(clip_limit=10.0, tile_grid_size=(4, 4))
    result_after = processor.process(image)

    # 期待結果を新パラメータで cv2 から直接計算.
    expected = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4)).apply(image)
    assert np.array_equal(result_after, expected)

    # 更新前後で出力が変化している.
    assert not np.array_equal(result_before, result_after)


def test_clahe_update_params_invalid_clip_limit():
    """不正な clip_limit は ValueError になる."""
    processor = CLAHEProcessor(name="clahe", config={})

    with pytest.raises(ValueError):
        processor.update_params(clip_limit=0.0)

    with pytest.raises(ValueError):
        processor.update_params(clip_limit=-1.0)


def test_clahe_update_params_invalid_tile_grid_size():
    """不正な tile_grid_size は ValueError になる."""
    processor = CLAHEProcessor(name="clahe", config={})

    with pytest.raises(ValueError):
        processor.update_params(tile_grid_size=[8])

    with pytest.raises(ValueError):
        processor.update_params(tile_grid_size=[8, 8, 8])

    with pytest.raises(ValueError):
        processor.update_params(tile_grid_size=[0, 8])

    with pytest.raises(ValueError):
        processor.update_params(tile_grid_size=[-1, 8])


def test_clahe_update_params_no_args_is_noop():
    """引数なしの update_params では値が変わらず CLAHE は再生成される."""
    processor = CLAHEProcessor(name="clahe", config={})
    old_clahe = processor.clahe

    processor.update_params()

    assert processor.clip_limit == 2.0
    assert processor.tile_grid_size == (8, 8)
    # 再生成されて別インスタンスになる.
    assert processor.clahe is not old_clahe


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
