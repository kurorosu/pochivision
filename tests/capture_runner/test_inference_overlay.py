"""InferenceOverlay のテスト."""

import numpy as np

from pochivision.capture_runner.inference_overlay import InferenceOverlay
from pochivision.request.api.inference.models import PredictResponse


def _make_result(
    confidence: float = 0.95, class_name: str = "class_a"
) -> PredictResponse:
    """テスト用の PredictResponse を生成する."""
    return PredictResponse(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        probabilities=[confidence, 1.0 - confidence],
        e2e_time_ms=12.3,
        backend="onnx",
        rtt_ms=65.1,
    )


class TestInferenceOverlay:
    """InferenceOverlay のテスト."""

    def test_initial_state(self):
        overlay = InferenceOverlay()
        assert overlay.result is None
        assert overlay._inferring is False

    def test_update(self):
        overlay = InferenceOverlay()
        result = _make_result()
        overlay.update(result)
        assert overlay.result is result

    def test_clear(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result())
        overlay.clear()
        assert overlay.result is None

    def test_draw_no_result(self):
        overlay = InferenceOverlay()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()
        result = overlay.draw(frame)
        np.testing.assert_array_equal(frame, original)
        assert result is frame

    def test_draw_with_result(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result())
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_color_high_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.9)
        assert color == (0, 200, 0)  # 緑

    def test_color_medium_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.5)
        assert color == (0, 200, 200)  # 黄

    def test_color_low_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.2)
        assert color == (0, 0, 200)  # 赤

    def test_color_boundary_high(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.7) == (0, 200, 0)  # ちょうど HIGH

    def test_color_boundary_low(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.4) == (0, 200, 200)  # ちょうど LOW

    def test_color_zero(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.0) == (0, 0, 200)  # 赤

    def test_color_one(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(1.0) == (0, 200, 0)  # 緑

    def test_draw_updates_after_new_result(self):
        overlay = InferenceOverlay()
        frame = np.zeros((100, 300, 3), dtype=np.uint8)

        overlay.update(_make_result(confidence=0.9, class_name="dog"))
        overlay.draw(frame)
        first_draw = frame.copy()

        frame2 = np.zeros((100, 300, 3), dtype=np.uint8)
        overlay.update(_make_result(confidence=0.3, class_name="cat"))
        overlay.draw(frame2)

        assert not np.array_equal(first_draw, frame2)

    def test_set_inferring(self):
        overlay = InferenceOverlay()
        overlay.set_inferring(True)
        assert overlay._inferring is True
        overlay.set_inferring(False)
        assert overlay._inferring is False

    def test_draw_inferring_no_result(self):
        overlay = InferenceOverlay()
        overlay.set_inferring(True)
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_draw_inferring_with_result(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result(confidence=0.9, class_name="dog"))
        overlay.set_inferring(True)
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        overlay.draw(frame)
        # result がある場合は前回の結果が表示される (Inferring... ではない)
        frame_result_only = np.zeros((100, 300, 3), dtype=np.uint8)
        overlay.set_inferring(False)
        overlay.draw(frame_result_only)
        np.testing.assert_array_equal(frame, frame_result_only)

    def test_draw_not_inferring_no_result(self):
        overlay = InferenceOverlay()
        overlay.set_inferring(False)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        original = frame.copy()
        overlay.draw(frame)
        np.testing.assert_array_equal(frame, original)
