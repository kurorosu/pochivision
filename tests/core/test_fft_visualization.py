"""SimpleFFTVisualizer のロジックテスト."""

import cv2
import numpy as np
import pytest

from pochivision.core.fft_visualization import SimpleFFTVisualizer


def _create_test_image(path, width=64, height=64):
    """テスト用グレースケール画像を作成して保存する."""
    image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def _create_color_image(path, width=64, height=64):
    """テスト用カラー画像を作成して保存する."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


class TestSimpleFFTVisualizer:
    """SimpleFFTVisualizer のテストクラス."""

    def test_load_grayscale_image(self, tmp_path):
        """グレースケール画像を読み込める."""
        img_path = _create_test_image(tmp_path / "gray.png")
        viz = SimpleFFTVisualizer(str(img_path))
        assert viz.load_image() is True
        assert viz.img is not None
        assert len(viz.img.shape) == 2

    def test_load_color_image_converts_to_gray(self, tmp_path):
        """カラー画像を読み込むとグレースケールに変換される."""
        img_path = _create_color_image(tmp_path / "color.png")
        viz = SimpleFFTVisualizer(str(img_path))
        assert viz.load_image() is True
        assert viz.img is not None
        assert len(viz.img.shape) == 2

    def test_load_nonexistent_image(self, tmp_path):
        """存在しない画像で False を返す."""
        viz = SimpleFFTVisualizer(str(tmp_path / "nonexistent.png"))
        assert viz.load_image() is False

    def test_compute_fft(self, tmp_path):
        """FFT 計算後にスペクトラムが生成される."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()
        viz.compute_fft()

        assert viz.fshift is not None
        assert viz.spectrum_display is not None
        assert viz.spectrum_display.dtype == np.uint8

    def test_compute_fft_without_image_raises(self):
        """画像未読み込みで compute_fft を呼ぶと ValueError."""
        viz = SimpleFFTVisualizer("dummy.png")
        with pytest.raises(ValueError):
            viz.compute_fft()

    def test_apply_filter_original(self, tmp_path):
        """original モードでは元画像がそのまま返る."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()
        viz.compute_fft()

        viz.filter_mode = "original"
        result = viz.apply_filter()
        np.testing.assert_array_equal(result, viz.img)

    def test_apply_filter_lowpass(self, tmp_path):
        """lowpass フィルタで画像が返る."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()
        viz.compute_fft()

        viz.filter_mode = "lowpass"
        viz.filter_radius = 20
        result = viz.apply_filter()

        assert result.shape == viz.img.shape
        assert result.dtype == np.uint8

    def test_apply_filter_highpass(self, tmp_path):
        """highpass フィルタで画像が返る."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()
        viz.compute_fft()

        viz.filter_mode = "highpass"
        viz.filter_radius = 20
        result = viz.apply_filter()

        assert result.shape == viz.img.shape
        assert result.dtype == np.uint8

    def test_apply_filter_without_fft_raises(self, tmp_path):
        """FFT 未計算で apply_filter を呼ぶと ValueError."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()

        with pytest.raises(ValueError):
            viz.apply_filter()

    def test_filter_radius_change(self, tmp_path):
        """フィルタ半径を変更すると結果が変わる."""
        img_path = _create_test_image(tmp_path / "img.png")
        viz = SimpleFFTVisualizer(str(img_path))
        viz.load_image()
        viz.compute_fft()
        viz.filter_mode = "lowpass"

        viz.filter_radius = 5
        result_small = viz.apply_filter()

        viz.filter_radius = 30
        result_large = viz.apply_filter()

        # 半径が異なれば結果も異なる
        assert not np.array_equal(result_small, result_large)
