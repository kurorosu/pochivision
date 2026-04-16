"""DetectionOverlay のテスト."""

import numpy as np

from pochivision.capture_runner.detection_overlay import (
    DetectionContext,
    DetectionOverlay,
)
from pochivision.request.api.detection.models import Detection, DetectionResponse


def _make_detection(
    class_id: int = 0,
    class_name: str = "pochi",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 100.0, 200.0),
) -> Detection:
    """テスト用の Detection を生成する."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )


def _make_response(
    detections: tuple[Detection, ...] = (),
    e2e_time_ms: float = 12.3,
    rtt_ms: float = 65.1,
    backend: str = "onnx",
) -> DetectionResponse:
    """テスト用の DetectionResponse を生成する."""
    if not detections:
        detections = (_make_detection(),)
    return DetectionResponse(
        detections=detections,
        e2e_time_ms=e2e_time_ms,
        backend=backend,
        rtt_ms=rtt_ms,
    )


def _make_context() -> DetectionContext:
    """テスト用の DetectionContext を生成する."""
    return DetectionContext(server_url="http://localhost:8000", image_size="640x640")


class TestState:
    """状態遷移のテスト."""

    def test_initial_state(self):
        overlay = DetectionOverlay()
        assert overlay.result is None
        assert overlay.error_message is None
        assert overlay._inferring is False

    def test_update(self):
        overlay = DetectionOverlay()
        result = _make_response()
        overlay.update(result)
        assert overlay.result is result
        assert overlay.error_message is None

    def test_update_clears_error(self):
        overlay = DetectionOverlay()
        overlay.set_error("connection error")
        overlay.update(_make_response())
        assert overlay.result is not None
        assert overlay.error_message is None

    def test_set_error_clears_result(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response())
        overlay.set_error("boom")
        assert overlay.error_message == "boom"
        assert overlay.result is None

    def test_clear(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response())
        overlay.clear()
        assert overlay.result is None
        assert overlay.error_message is None

    def test_set_inferring(self):
        overlay = DetectionOverlay()
        overlay.set_inferring(True)
        assert overlay._inferring is True


class TestGetColor:
    """get_color (決定的色割当) のテスト."""

    def test_same_class_id_same_color(self):
        overlay = DetectionOverlay()
        assert overlay.get_color(0) == overlay.get_color(0)

    def test_different_class_id_different_color(self):
        overlay = DetectionOverlay()
        assert overlay.get_color(0) != overlay.get_color(1)

    def test_class_id_wraps_around_palette(self):
        overlay = DetectionOverlay()
        palette_size = len(DetectionOverlay.PALETTE)
        assert overlay.get_color(0) == overlay.get_color(palette_size)


class TestDraw:
    """draw のテスト."""

    def test_no_result_returns_frame_unchanged(self):
        overlay = DetectionOverlay()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        original = frame.copy()
        result = overlay.draw(frame)
        np.testing.assert_array_equal(result, original)

    def test_inferring_draws_text_when_no_result(self):
        overlay = DetectionOverlay()
        overlay.set_inferring(True)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_error_draws_on_frame(self):
        overlay = DetectionOverlay(context=_make_context())
        overlay.set_error("connection refused")
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_empty_detections_draws_meta_only(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response(detections=()))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        # 空の tuple を渡すと _make_response がデフォルト値を入れてしまうので回避
        overlay.result = DetectionResponse(
            detections=(), e2e_time_ms=5.0, backend="onnx", rtt_ms=10.0
        )
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_bbox_drawn_within_frame(self):
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(50.0, 60.0, 150.0, 180.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        # bbox 領域周辺にピクセルが書かれているか
        assert result[60:180, 50:150].sum() > 0

    def test_bbox_color_matches_class_id(self):
        overlay = DetectionOverlay()
        det = _make_detection(class_id=2, bbox=(20.0, 20.0, 100.0, 100.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        expected_color = overlay.get_color(2)
        # bbox 辺上のどこかに期待色が存在する
        b, g, r = expected_color
        match = (result[..., 0] == b) & (result[..., 1] == g) & (result[..., 2] == r)
        assert match.any()

    def test_multiple_detections_different_colors(self):
        overlay = DetectionOverlay()
        dets = (
            _make_detection(class_id=0, bbox=(10.0, 10.0, 80.0, 80.0)),
            _make_detection(class_id=1, bbox=(100.0, 100.0, 180.0, 180.0)),
        )
        overlay.update(_make_response(detections=dets))
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        c0 = overlay.get_color(0)
        c1 = overlay.get_color(1)
        has_c0 = (
            (result[..., 0] == c0[0])
            & (result[..., 1] == c0[1])
            & (result[..., 2] == c0[2])
        ).any()
        has_c1 = (
            (result[..., 0] == c1[0])
            & (result[..., 1] == c1[1])
            & (result[..., 2] == c1[2])
        ).any()
        assert has_c0 and has_c1

    def test_label_y_at_top_of_frame_does_not_crash(self):
        """bbox が画面上端に張り付いてもラベル描画で例外が出ない."""
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(0.0, 0.0, 50.0, 30.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        overlay.draw(frame)
