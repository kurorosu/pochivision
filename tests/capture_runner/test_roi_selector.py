"""RoiSelector のテスト."""

import cv2
import numpy as np

from pochivision.capture_runner.roi_selector import RoiSelector


class TestRoiSelector:
    """RoiSelector のテスト."""

    def test_initial_state(self):
        """初期状態で ROI は None."""
        selector = RoiSelector()
        assert selector.roi is None

    def test_crop_without_roi(self):
        """ROI 未設定の場合, 元フレームがそのまま返される."""
        selector = RoiSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = selector.crop(frame)
        assert result is frame

    def test_crop_with_roi(self):
        """ROI 設定済みの場合, クロップされたフレームが返される."""
        selector = RoiSelector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:300] = 255

        selector.roi = (100, 100, 200, 100)
        result = selector.crop(frame)

        assert result.shape == (100, 200, 3)
        assert result.sum() > 0

    def test_crop_clamps_to_frame(self):
        """ROI がフレーム外にはみ出す場合, フレーム内にクランプされる."""
        selector = RoiSelector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        selector.roi = (80, 80, 50, 50)
        result = selector.crop(frame)

        assert result.shape[0] <= 100
        assert result.shape[1] <= 100

    def test_clear(self):
        """clear で ROI がリセットされる."""
        selector = RoiSelector()
        selector.roi = (10, 10, 50, 50)
        selector.clear()
        assert selector.roi is None

    def test_mouse_drag_sets_roi(self):
        """マウスドラッグで ROI が設定される."""
        selector = RoiSelector()
        selector._preview_scale = 1.0

        selector.mouse_callback(cv2.EVENT_LBUTTONDOWN, 10, 10)
        selector.mouse_callback(cv2.EVENT_MOUSEMOVE, 100, 100)
        selector.mouse_callback(cv2.EVENT_LBUTTONUP, 100, 100)

        assert selector.roi is not None
        x, y, w, h = selector.roi
        assert x == 10
        assert y == 10
        assert w == 90
        assert h == 90

    def test_mouse_drag_too_small_ignored(self):
        """5px 未満のドラッグは無視される."""
        selector = RoiSelector()
        selector._preview_scale = 1.0

        selector.mouse_callback(cv2.EVENT_LBUTTONDOWN, 10, 10)
        selector.mouse_callback(cv2.EVENT_LBUTTONUP, 12, 12)

        assert selector.roi is None

    def test_mouse_drag_reverse_direction(self):
        """右下→左上のドラッグでも正しく ROI が設定される."""
        selector = RoiSelector()
        selector._preview_scale = 1.0

        selector.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100)
        selector.mouse_callback(cv2.EVENT_LBUTTONUP, 10, 10)

        assert selector.roi is not None
        x, y, w, h = selector.roi
        assert x == 10
        assert y == 10
        assert w == 90
        assert h == 90

    def test_preview_scale_conversion(self):
        """プレビュー座標 → 元フレーム座標への変換が正しい."""
        selector = RoiSelector()
        selector.set_preview_scale(frame_w=1920, preview_w=960)

        selector.mouse_callback(cv2.EVENT_LBUTTONDOWN, 100, 100)
        selector.mouse_callback(cv2.EVENT_LBUTTONUP, 200, 200)

        assert selector.roi is not None
        x, y, w, h = selector.roi
        # scale = 1920/960 = 2.0
        assert x == 200
        assert y == 200
        assert w == 200
        assert h == 200

    def test_draw_with_roi(self):
        """ROI 設定済みの場合, プレビューに矩形が描画される."""
        selector = RoiSelector()
        selector.roi = (100, 100, 200, 100)
        selector._preview_scale = 2.0

        preview = np.zeros((240, 320, 3), dtype=np.uint8)
        selector.draw(preview)

        assert preview.sum() > 0

    def test_draw_without_roi(self):
        """ROI 未設定の場合, プレビューは変更されない."""
        selector = RoiSelector()

        preview = np.zeros((240, 320, 3), dtype=np.uint8)
        original = preview.copy()
        selector.draw(preview)

        np.testing.assert_array_equal(preview, original)

    def test_draw_during_drag(self):
        """ドラッグ中は選択中の矩形が描画される."""
        selector = RoiSelector()
        selector.mouse_callback(cv2.EVENT_LBUTTONDOWN, 10, 10)
        selector.mouse_callback(cv2.EVENT_MOUSEMOVE, 100, 100)

        preview = np.zeros((240, 320, 3), dtype=np.uint8)
        selector.draw(preview)

        assert preview.sum() > 0

    def test_set_preview_scale(self):
        """set_preview_scale でスケールが正しく設定される."""
        selector = RoiSelector()
        selector.set_preview_scale(frame_w=1280, preview_w=640)
        assert selector._preview_scale == 2.0

    def test_set_preview_scale_zero_preview(self):
        """プレビュー幅 0 の場合, スケールは変更されない."""
        selector = RoiSelector()
        original_scale = selector._preview_scale
        selector.set_preview_scale(frame_w=1280, preview_w=0)
        assert selector._preview_scale == original_scale
