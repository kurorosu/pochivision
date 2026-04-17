"""検出ランタイム (スロットリング + i キートグル + ROI 無効化) のテスト."""

from unittest.mock import MagicMock, patch

import numpy as np

from pochivision.capture_runner.viewer import LivePreviewRunner
from pochivision.exceptions import DetectionConnectionError, DetectionError
from pochivision.request.api.detection.client import DetectionClient
from pochivision.request.api.detection.models import DetectionResponse


def _make_frame() -> np.ndarray:
    """テスト用フレームを生成する."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_response() -> DetectionResponse:
    """テスト用の空 DetectionResponse を生成する."""
    return DetectionResponse(
        detections=(),
        e2e_time_ms=10.0,
        backend="onnx",
        rtt_ms=20.0,
    )


def _make_runner(tmp_path, detect_fps: float = 5.0) -> LivePreviewRunner:
    """detect モードの LivePreviewRunner を生成する."""
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path
    client = DetectionClient(base_url="http://localhost:8000")
    return LivePreviewRunner(
        cap,
        pipeline,
        detection_client=client,
        detect_fps=detect_fps,
    )


class TestDetectMode:
    """モード判定と初期状態のテスト."""

    def test_is_detect_mode_true_when_client_set(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.is_detect_mode is True
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_is_detect_mode_false_when_no_client(self, tmp_path):
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        runner = LivePreviewRunner(cap, pipeline)
        assert runner.is_detect_mode is False

    def test_detection_enabled_defaults_true_in_detect_mode(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner._detection_enabled is True
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_detect_period_matches_fps(self, tmp_path):
        runner = _make_runner(tmp_path, detect_fps=5.0)
        assert runner._detect_period_s == 0.2
        runner.detection_client.close()  # type: ignore[union-attr]


class TestThrottling:
    """時間ベースのスロットリング挙動のテスト."""

    def test_first_call_fires(self, tmp_path):
        """最初の呼び出しは (前回送信履歴が無いため) 送信される."""
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client, "detect", return_value=_make_response()
        ):
            started = runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert started is True
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_within_period_is_skipped(self, tmp_path):
        """period 未満の連続呼び出しは 2 回目以降スキップされる."""
        runner = _make_runner(tmp_path, detect_fps=5.0)  # period = 200ms
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            runner.detection_client, "detect", side_effect=counting_detect
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)
            # 100ms 経過 (period 未満) では発火しない
            started = runner._maybe_detect(_make_frame(), now=0.1)

        assert started is False
        assert call_count == 1
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_after_period_fires_again(self, tmp_path):
        """period 経過後の呼び出しは再度発火する."""
        runner = _make_runner(tmp_path, detect_fps=5.0)  # period = 200ms
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            runner.detection_client, "detect", side_effect=counting_detect
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)
            # 200ms 経過で再発火
            started = runner._maybe_detect(_make_frame(), now=0.2)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert started is True
        assert call_count == 2
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_expected_rate_over_10_ticks(self, tmp_path):
        """30 FPS 入力 + 5 FPS スロットリングで 1 秒間に 約 5 回送信される."""
        runner = _make_runner(tmp_path, detect_fps=5.0)
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            runner.detection_client, "detect", side_effect=counting_detect
        ):
            # 30 FPS 相当で 30 tick = 1 秒
            for i in range(30):
                t = i * (1.0 / 30.0)
                runner._maybe_detect(_make_frame(), now=t)
                if runner._detection_thread:
                    runner._detection_thread.join(timeout=5.0)

        # 1 秒で 5 FPS なら 5-6 回程度を期待 (最初に即発火 + 0.2, 0.4, 0.6, 0.8 で発火)
        assert 5 <= call_count <= 6
        runner.detection_client.close()  # type: ignore[union-attr]


class TestInFlightGuard:
    """In-flight ガードのテスト (前リクエスト未完了なら送らない)."""

    def test_skipped_when_already_detecting(self, tmp_path):
        """`_detecting` が True のときはスキップされる."""
        runner = _make_runner(tmp_path, detect_fps=5.0)
        runner._detecting = True

        with patch.object(
            runner.detection_client, "detect", return_value=_make_response()
        ) as mock_detect:
            started = runner._maybe_detect(_make_frame(), now=0.0)

        assert started is False
        mock_detect.assert_not_called()
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_detecting_flag_reset_on_success(self, tmp_path):
        """検出成功後に `_detecting` フラグがリセットされる."""
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client, "detect", return_value=_make_response()
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert runner._detecting is False
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_detecting_flag_reset_on_error(self, tmp_path):
        """検出失敗時にも `_detecting` フラグがリセットされる."""
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client, "detect", side_effect=Exception("boom")
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert runner._detecting is False
        runner.detection_client.close()  # type: ignore[union-attr]


class TestToggle:
    """i キートグル (_toggle_detection) の挙動."""

    def test_toggle_flips_enabled(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner._detection_enabled is True
        runner._toggle_detection()
        assert runner._detection_enabled is False
        runner._toggle_detection()
        assert runner._detection_enabled is True
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_toggle_off_clears_overlay(self, tmp_path):
        """OFF に切り替えると overlay の state がクリアされる."""
        runner = _make_runner(tmp_path)
        runner.detection_overlay.update(_make_response())
        assert runner.detection_overlay.result is not None

        runner._toggle_detection()  # ON -> OFF

        assert runner._detection_enabled is False
        assert runner.detection_overlay.result is None
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_disabled_skips_send(self, tmp_path):
        """OFF のとき `_maybe_detect` は送信しない."""
        runner = _make_runner(tmp_path, detect_fps=5.0)
        runner._detection_enabled = False

        with patch.object(
            runner.detection_client, "detect", return_value=_make_response()
        ) as mock_detect:
            started = runner._maybe_detect(_make_frame(), now=0.0)

        assert started is False
        mock_detect.assert_not_called()
        runner.detection_client.close()  # type: ignore[union-attr]


class TestRoiDisabled:
    """detect モードでは ROI 選択が使われないことを確認する."""

    def test_is_detect_mode_property(self, tmp_path):
        """is_detect_mode が run() の ROI 分岐で使われることを確認するプロキシ."""
        runner = _make_runner(tmp_path)
        assert runner.is_detect_mode is True
        # run() 側で is_detect_mode により mouse callback / roi draw が
        # スキップされることは統合テストでなく run() の分岐条件で担保する.
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_full_frame_sent_to_detect(self, tmp_path):
        """detect モードではフル解像度フレームがクロップなしで送信される."""
        runner = _make_runner(tmp_path, detect_fps=5.0)
        frame = _make_frame()  # (100, 100, 3)

        with patch.object(
            runner.detection_client,
            "detect",
            return_value=_make_response(),
        ) as mock_detect:
            runner._maybe_detect(frame, now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        mock_detect.assert_called_once()
        sent_frame = mock_detect.call_args[0][0]
        assert sent_frame.shape == frame.shape
        runner.detection_client.close()  # type: ignore[union-attr]


class TestErrorHandling:
    """検出失敗時に overlay にエラーメッセージが反映される."""

    def test_connection_error_sets_overlay_error(self, tmp_path):
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client,
            "detect",
            side_effect=DetectionConnectionError("timeout"),
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert runner.detection_overlay.error_message == "Connection failed"
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_detection_error_sets_overlay_error(self, tmp_path):
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client,
            "detect",
            side_effect=DetectionError("bad response"),
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert runner.detection_overlay.error_message == "Detection failed"
        runner.detection_client.close()  # type: ignore[union-attr]

    def test_unexpected_error_sets_overlay_error(self, tmp_path):
        runner = _make_runner(tmp_path, detect_fps=5.0)

        with patch.object(
            runner.detection_client,
            "detect",
            side_effect=RuntimeError("boom"),
        ):
            runner._maybe_detect(_make_frame(), now=0.0)
            if runner._detection_thread:
                runner._detection_thread.join(timeout=5.0)

        assert runner.detection_overlay.error_message == "Unexpected error"
        runner.detection_client.close()  # type: ignore[union-attr]
