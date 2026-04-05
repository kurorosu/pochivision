"""_run_inference のスレッドセーフ性テスト."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from pochivision.capture_runner.viewer import LivePreviewRunner
from pochivision.request.api.inference.client import InferenceClient
from pochivision.request.api.inference.models import PredictResponse


def _make_frame() -> np.ndarray:
    """テスト用フレームを生成する."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_result() -> PredictResponse:
    """テスト用の PredictResponse を生成する."""
    return PredictResponse(
        class_id=0,
        class_name="cat",
        confidence=0.95,
        probabilities=[0.95, 0.05],
        e2e_time_ms=12.3,
        backend="onnx",
        rtt_ms=50.0,
    )


def _make_runner(tmp_path) -> LivePreviewRunner:
    """テスト用の LivePreviewRunner を生成する."""
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path

    client = InferenceClient(base_url="http://localhost:8000")

    return LivePreviewRunner(cap, pipeline, inference_client=client)


class TestRunInferenceThreadSafety:
    """_run_inference のスレッドセーフ性テスト."""

    def test_concurrent_calls_only_one_executes(self, tmp_path):
        """並行呼び出しで推論が 1 回だけ実行される."""
        runner = _make_runner(tmp_path)
        call_count = 0
        barrier = threading.Barrier(2, timeout=5)

        original_predict = runner.inference_client.predict  # type: ignore[union-attr]

        def slow_predict(frame: np.ndarray) -> PredictResponse:
            nonlocal call_count
            call_count += 1
            barrier.wait()
            return _make_result()

        with patch.object(runner.inference_client, "predict", side_effect=slow_predict):
            frame = _make_frame()

            # 2 スレッドから同時に _run_inference を呼び出す
            t1 = threading.Thread(target=runner._run_inference, args=(frame,))
            t2 = threading.Thread(target=runner._run_inference, args=(frame,))
            t1.start()
            t2.start()

            # barrier で 1 スレッドしか来ないケースに備えてタイムアウト
            t1.join(timeout=5)
            t2.join(timeout=5)

        # predict は 1 回のみ呼ばれるべき
        assert call_count == 1

        runner.inference_client.close()  # type: ignore[union-attr]

    def test_inferring_flag_reset_after_completion(self, tmp_path):
        """推論完了後に _inferring フラグがリセットされる."""
        runner = _make_runner(tmp_path)

        with patch.object(
            runner.inference_client, "predict", return_value=_make_result()
        ):
            runner._run_inference(_make_frame())

            # ワーカースレッド完了を待つ
            if runner._inference_thread:
                runner._inference_thread.join(timeout=5)

        assert runner._inferring is False

        runner.inference_client.close()  # type: ignore[union-attr]

    def test_inferring_flag_reset_on_error(self, tmp_path):
        """推論失敗時にも _inferring フラグがリセットされる."""
        runner = _make_runner(tmp_path)

        with patch.object(
            runner.inference_client,
            "predict",
            side_effect=Exception("test error"),
        ):
            runner._run_inference(_make_frame())

            if runner._inference_thread:
                runner._inference_thread.join(timeout=5)

        assert runner._inferring is False

        runner.inference_client.close()  # type: ignore[union-attr]

    def test_can_run_again_after_completion(self, tmp_path):
        """推論完了後に再度推論を実行できる."""
        runner = _make_runner(tmp_path)
        call_count = 0

        def counting_predict(frame: np.ndarray) -> PredictResponse:
            nonlocal call_count
            call_count += 1
            return _make_result()

        with patch.object(
            runner.inference_client, "predict", side_effect=counting_predict
        ):
            # 1 回目
            runner._run_inference(_make_frame())
            if runner._inference_thread:
                runner._inference_thread.join(timeout=5)

            # 2 回目
            runner._run_inference(_make_frame())
            if runner._inference_thread:
                runner._inference_thread.join(timeout=5)

        assert call_count == 2

        runner.inference_client.close()  # type: ignore[union-attr]

    def test_no_client_returns_immediately(self, tmp_path):
        """inference_client=None のとき即座に return する."""
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        runner = LivePreviewRunner(cap, pipeline)
        runner._run_inference(_make_frame())

        assert runner._inferring is False
