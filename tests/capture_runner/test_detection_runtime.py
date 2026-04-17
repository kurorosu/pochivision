"""検出ランタイム (スロットリング + i キートグル + ROI 無効化) のテスト."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


@pytest.fixture
def detect_runner(tmp_path):
    """detect モードの LivePreviewRunner を yield し, cleanup する fixture.

    `cap` / `pipeline` は cv2 / PipelineExecutor の重い初期化を避けるため
    MagicMock を使用 (本プロジェクトの既存テスト test_run_inference.py と同じ方針).
    `DetectionClient` は実オブジェクトを生成する.

    `_detection_enabled` はデフォルト False だが, スロットリング / worker 系の
    挙動を検証するテストでは事前に True に設定しておく (i キー押下相当).
    """
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path
    client = DetectionClient(base_url="http://localhost:8000")
    runner = LivePreviewRunner(
        cap,
        pipeline,
        detection_client=client,
        detect_fps=5.0,
    )
    runner._detection_enabled = True
    try:
        yield runner
    finally:
        if runner._detection_thread is not None:
            runner._detection_thread.join(timeout=2.0)
        assert runner.detection_client is not None
        runner.detection_client.close()


@pytest.fixture
def no_detect_runner(tmp_path):
    """detect_client なしの LivePreviewRunner (classify モード相当)."""
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path
    return LivePreviewRunner(cap, pipeline)


class TestDetectMode:
    """モード判定と初期状態のテスト."""

    def test_is_detect_mode_true_when_client_set(self, detect_runner):
        assert detect_runner.is_detect_mode is True

    def test_is_detect_mode_false_when_no_client(self, no_detect_runner):
        assert no_detect_runner.is_detect_mode is False

    def test_detection_enabled_defaults_false_on_startup(self, tmp_path):
        """起動直後は `i` 押下待ちで検出は OFF. fixture での True 上書きをバイパス."""
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        client = DetectionClient(base_url="http://localhost:8000")
        runner = LivePreviewRunner(
            cap,
            pipeline,
            detection_client=client,
            detect_fps=5.0,
        )
        try:
            assert runner._detection_enabled is False
        finally:
            client.close()

    def test_detect_period_matches_fps(self, detect_runner):
        # fixture の detect_fps=5.0 → period=0.2s
        assert detect_runner._detect_period_s == 0.2


class TestThrottling:
    """時間ベースのスロットリング挙動のテスト."""

    def test_first_call_fires(self, detect_runner):
        """最初の呼び出しは (前回送信履歴が無いため) 送信される."""
        with patch.object(
            detect_runner.detection_client, "detect", return_value=_make_response()
        ):
            started = detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert started is True

    def test_within_period_is_skipped(self, detect_runner):
        """period 未満の連続呼び出しは 2 回目以降スキップされる."""
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            detect_runner.detection_client, "detect", side_effect=counting_detect
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)
            # 100ms 経過 (period 未満) では発火しない
            started = detect_runner._maybe_detect(_make_frame(), now=0.1)

        assert started is False
        assert call_count == 1

    def test_after_period_fires_again(self, detect_runner):
        """period 経過後の呼び出しは再度発火する."""
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            detect_runner.detection_client, "detect", side_effect=counting_detect
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)
            # 200ms 経過で再発火
            started = detect_runner._maybe_detect(_make_frame(), now=0.2)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert started is True
        assert call_count == 2

    def test_expected_rate_over_10_ticks(self, detect_runner):
        """30 FPS 入力 + 5 FPS スロットリングで 1 秒間に 約 5 回送信される."""
        call_count = 0

        def counting_detect(frame: np.ndarray) -> DetectionResponse:
            nonlocal call_count
            call_count += 1
            return _make_response()

        with patch.object(
            detect_runner.detection_client, "detect", side_effect=counting_detect
        ):
            # 30 FPS 相当で 30 tick = 1 秒
            for i in range(30):
                t = i * (1.0 / 30.0)
                detect_runner._maybe_detect(_make_frame(), now=t)
                if detect_runner._detection_thread:
                    detect_runner._detection_thread.join(timeout=2.0)

        # 1 秒で 5 FPS なら 5-6 回程度を期待 (最初に即発火 + 0.2, 0.4, 0.6, 0.8 で発火)
        assert 5 <= call_count <= 6


class TestInFlightGuard:
    """In-flight ガードのテスト (前リクエスト未完了なら送らない)."""

    def test_skipped_when_already_detecting(self, detect_runner):
        """`_detecting` が True のときはスキップされる."""
        detect_runner._detecting = True

        with patch.object(
            detect_runner.detection_client, "detect", return_value=_make_response()
        ) as mock_detect:
            started = detect_runner._maybe_detect(_make_frame(), now=0.0)

        assert started is False
        mock_detect.assert_not_called()

    def test_detecting_flag_reset_on_success(self, detect_runner):
        """検出成功後に `_detecting` フラグがリセットされる."""
        with patch.object(
            detect_runner.detection_client, "detect", return_value=_make_response()
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner._detecting is False

    def test_detecting_flag_reset_on_error(self, detect_runner):
        """検出失敗時にも `_detecting` フラグがリセットされる."""
        with patch.object(
            detect_runner.detection_client, "detect", side_effect=Exception("boom")
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner._detecting is False


class TestToggle:
    """i キートグル (_toggle_detection) の挙動."""

    def test_toggle_flips_enabled(self, detect_runner):
        assert detect_runner._detection_enabled is True
        detect_runner._toggle_detection()
        assert detect_runner._detection_enabled is False
        detect_runner._toggle_detection()
        assert detect_runner._detection_enabled is True

    def test_toggle_off_clears_overlay(self, detect_runner):
        """OFF に切り替えると overlay の state がクリアされる."""
        detect_runner.detection_overlay.update(_make_response())
        assert detect_runner.detection_overlay.result is not None

        detect_runner._toggle_detection()  # ON -> OFF

        assert detect_runner._detection_enabled is False
        assert detect_runner.detection_overlay.result is None

    def test_disabled_skips_send(self, detect_runner):
        """OFF のとき `_maybe_detect` は送信しない."""
        detect_runner._detection_enabled = False

        with patch.object(
            detect_runner.detection_client, "detect", return_value=_make_response()
        ) as mock_detect:
            started = detect_runner._maybe_detect(_make_frame(), now=0.0)

        assert started is False
        mock_detect.assert_not_called()


class TestToggleWorkerRace:
    """OFF 切替中に完了した worker が stale 結果を overlay に残さないことを検証."""

    def test_toggle_off_during_detect_discards_result(self, detect_runner):
        """worker が detect() 中に toggle OFF → worker 結果は overlay に反映されない.

        Barrier で worker を detect() 中で停止させ, その間に UI 側から
        `_toggle_detection()` OFF を実行. その後 detect() を完了させると,
        overlay は clear 済みのままで result / error は反映されない.
        """
        enter_barrier = threading.Barrier(2, timeout=5)
        exit_event = threading.Event()

        def blocking_detect(frame: np.ndarray) -> DetectionResponse:
            enter_barrier.wait()
            exit_event.wait(timeout=5)
            return _make_response()

        with patch.object(
            detect_runner.detection_client, "detect", side_effect=blocking_detect
        ):
            started = detect_runner._maybe_detect(_make_frame(), now=0.0)
            assert started is True

            # worker が detect() に入ったタイミングを待つ
            enter_barrier.wait()

            # UI スレッドから OFF に切り替え (overlay.clear + flag 降ろし)
            detect_runner._toggle_detection()
            assert detect_runner._detection_enabled is False

            # worker を解放 → detect() 完了 → update が走るが, lock 下で
            # _detection_enabled=False を確認して skip されるはず
            exit_event.set()
            assert detect_runner._detection_thread is not None
            detect_runner._detection_thread.join(timeout=2.0)

        # stale な result が残っていないことを確認
        assert detect_runner.detection_overlay.result is None
        assert detect_runner.detection_overlay.error_message is None

    def test_toggle_off_during_error_discards_error(self, detect_runner):
        """worker がエラーを返す直前に toggle OFF → overlay にエラーは残らない."""
        enter_barrier = threading.Barrier(2, timeout=5)
        exit_event = threading.Event()

        def failing_detect(frame: np.ndarray) -> DetectionResponse:
            enter_barrier.wait()
            exit_event.wait(timeout=5)
            raise DetectionConnectionError("simulated timeout")

        with patch.object(
            detect_runner.detection_client, "detect", side_effect=failing_detect
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            enter_barrier.wait()

            detect_runner._toggle_detection()
            assert detect_runner._detection_enabled is False

            exit_event.set()
            assert detect_runner._detection_thread is not None
            detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner.detection_overlay.error_message is None


class TestRoiDisabled:
    """detect モードでは ROI 選択が使われないことを確認する."""

    def test_is_detect_mode_property(self, detect_runner):
        """is_detect_mode が run() の ROI 分岐で使われることを確認するプロキシ."""
        assert detect_runner.is_detect_mode is True
        # run() 側で is_detect_mode により mouse callback / roi draw が
        # スキップされることは統合テストでなく run() の分岐条件で担保する.

    def test_full_frame_sent_to_detect(self, detect_runner):
        """detect モードではフル解像度フレームがクロップなしで送信される."""
        frame = _make_frame()  # (100, 100, 3)

        with patch.object(
            detect_runner.detection_client,
            "detect",
            return_value=_make_response(),
        ) as mock_detect:
            detect_runner._maybe_detect(frame, now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        mock_detect.assert_called_once()
        sent_frame = mock_detect.call_args[0][0]
        assert sent_frame.shape == frame.shape


class TestErrorHandling:
    """検出失敗時に overlay にエラーメッセージが反映される."""

    def test_connection_error_sets_overlay_error(self, detect_runner):
        with patch.object(
            detect_runner.detection_client,
            "detect",
            side_effect=DetectionConnectionError("timeout"),
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner.detection_overlay.error_message == "Connection failed"

    def test_detection_error_sets_overlay_error(self, detect_runner):
        with patch.object(
            detect_runner.detection_client,
            "detect",
            side_effect=DetectionError("bad response"),
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner.detection_overlay.error_message == "Detection failed"

    def test_unexpected_error_sets_overlay_error(self, detect_runner):
        with patch.object(
            detect_runner.detection_client,
            "detect",
            side_effect=RuntimeError("boom"),
        ):
            detect_runner._maybe_detect(_make_frame(), now=0.0)
            if detect_runner._detection_thread:
                detect_runner._detection_thread.join(timeout=2.0)

        assert detect_runner.detection_overlay.error_message == "Unexpected error"
