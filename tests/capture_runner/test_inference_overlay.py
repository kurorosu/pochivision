"""InferenceOverlay のテスト."""

import numpy as np

from pochivision.capture_runner.inference_overlay import (
    InferenceContext,
    InferenceOverlay,
)
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


def _make_context() -> InferenceContext:
    """テスト用の InferenceContext を生成する."""
    return InferenceContext(server_url="http://localhost:8000", image_size="512x512")


class TestInferenceOverlay:
    """InferenceOverlay のテスト."""

    def test_initial_state(self):
        overlay = InferenceOverlay()
        assert overlay.result is None
        assert overlay.error_message is None
        assert overlay._inferring is False

    def test_update(self):
        overlay = InferenceOverlay()
        result = _make_result()
        overlay.update(result)
        assert overlay.result is result
        assert overlay.error_message is None

    def test_update_clears_error(self):
        overlay = InferenceOverlay()
        overlay.set_error("connection error")
        overlay.update(_make_result())
        assert overlay.result is not None
        assert overlay.error_message is None

    def test_set_error(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result())
        overlay.set_error("connection error")
        assert overlay.error_message == "connection error"
        assert overlay.result is None

    def test_clear(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result())
        overlay.clear()
        assert overlay.result is None
        assert overlay.error_message is None

    def test_clear_error(self):
        overlay = InferenceOverlay()
        overlay.set_error("error")
        overlay.clear()
        assert overlay.error_message is None

    def test_draw_no_result(self):
        overlay = InferenceOverlay()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        original = frame.copy()
        result = overlay.draw(frame)
        np.testing.assert_array_equal(frame, original)
        assert result is frame

    def test_draw_with_result(self):
        overlay = InferenceOverlay(_make_context())
        overlay.update(_make_result())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_draw_error(self):
        overlay = InferenceOverlay(_make_context())
        overlay.set_error("推論 API サーバーに接続できません")
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_draw_error_without_context(self):
        overlay = InferenceOverlay()
        overlay.set_error("connection error")
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_color_high_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.9)
        assert color == (0, 200, 0)

    def test_color_medium_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.5)
        assert color == (0, 200, 200)

    def test_color_low_confidence(self):
        overlay = InferenceOverlay()
        color = overlay._get_color(0.2)
        assert color == (0, 0, 200)

    def test_color_boundary_high(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.7) == (0, 200, 0)

    def test_color_boundary_low(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.4) == (0, 200, 200)

    def test_color_zero(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(0.0) == (0, 0, 200)

    def test_color_one(self):
        overlay = InferenceOverlay()
        assert overlay._get_color(1.0) == (0, 200, 0)

    def test_draw_updates_after_new_result(self):
        overlay = InferenceOverlay()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        overlay.update(_make_result(confidence=0.9, class_name="dog"))
        overlay.draw(frame)
        first_draw = frame.copy()

        frame2 = np.zeros((200, 400, 3), dtype=np.uint8)
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
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert frame.sum() > 0
        assert result is frame

    def test_draw_inferring_with_result(self):
        overlay = InferenceOverlay()
        overlay.update(_make_result(confidence=0.9, class_name="dog"))
        overlay.set_inferring(True)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.draw(frame)
        frame_result_only = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.set_inferring(False)
        overlay.draw(frame_result_only)
        np.testing.assert_array_equal(frame, frame_result_only)

    def test_draw_not_inferring_no_result(self):
        overlay = InferenceOverlay()
        overlay.set_inferring(False)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        original = frame.copy()
        overlay.draw(frame)
        np.testing.assert_array_equal(frame, original)

    def test_context_none(self):
        overlay = InferenceOverlay(context=None)
        overlay.update(_make_result())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.draw(frame)
        assert frame.sum() > 0

    def test_context_without_image_size(self):
        ctx = InferenceContext(server_url="http://localhost:8000")
        overlay = InferenceOverlay(context=ctx)
        overlay.update(_make_result())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.draw(frame)
        assert frame.sum() > 0

    def test_error_then_success_clears_error(self):
        overlay = InferenceOverlay(_make_context())
        overlay.set_error("connection error")
        frame_error = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.draw(frame_error)

        overlay.update(_make_result())
        frame_success = np.zeros((200, 400, 3), dtype=np.uint8)
        overlay.draw(frame_success)

        assert not np.array_equal(frame_error, frame_success)
