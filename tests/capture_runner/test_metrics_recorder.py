"""MetricsRecorder のテスト."""

import pandas as pd
import pytest

from pochivision.capture_runner.metrics_recorder import MetricsRecorder
from pochivision.request.api.detection.models import Detection, DetectionResponse


def _make_response(
    detections: tuple[Detection, ...] = (),
    e2e_time_ms: float = 10.0,
    rtt_ms: float = 2.0,
    backend: str = "onnx-cuda",
    phase_times_ms: dict[str, float] | None = None,
    gpu_clock_mhz: int | None = 1770,
    gpu_vram_used_mb: int | None = 2048,
    gpu_temperature_c: int | None = 55,
) -> DetectionResponse:
    """テスト用の DetectionResponse を組み立てる."""
    return DetectionResponse(
        detections=detections,
        e2e_time_ms=e2e_time_ms,
        backend=backend,
        rtt_ms=rtt_ms,
        phase_times_ms=phase_times_ms or {},
        gpu_clock_mhz=gpu_clock_mhz,
        gpu_vram_used_mb=gpu_vram_used_mb,
        gpu_temperature_c=gpu_temperature_c,
    )


class TestMaybeRecord:
    """maybe_record のサンプリング挙動テスト."""

    def test_disabled_when_interval_zero(self, tmp_path):
        recorder = MetricsRecorder(interval_s=0.0, out_path=tmp_path / "m.csv")
        assert recorder.maybe_record(_make_response(), now_monotonic=100.0) is False
        assert recorder.row_count == 0

    def test_disabled_when_interval_negative(self, tmp_path):
        recorder = MetricsRecorder(interval_s=-0.5, out_path=tmp_path / "m.csv")
        assert recorder.maybe_record(_make_response(), now_monotonic=100.0) is False
        assert recorder.row_count == 0

    def test_first_call_records(self, tmp_path):
        recorder = MetricsRecorder(interval_s=1.0, out_path=tmp_path / "m.csv")
        assert recorder.maybe_record(_make_response(), now_monotonic=100.0) is True
        assert recorder.row_count == 1

    def test_second_call_before_interval_skipped(self, tmp_path):
        recorder = MetricsRecorder(interval_s=1.0, out_path=tmp_path / "m.csv")
        recorder.maybe_record(_make_response(), now_monotonic=100.0)
        assert recorder.maybe_record(_make_response(), now_monotonic=100.5) is False
        assert recorder.row_count == 1

    def test_second_call_after_interval_records(self, tmp_path):
        recorder = MetricsRecorder(interval_s=1.0, out_path=tmp_path / "m.csv")
        recorder.maybe_record(_make_response(), now_monotonic=100.0)
        assert recorder.maybe_record(_make_response(), now_monotonic=101.0) is True
        assert recorder.row_count == 2

    def test_records_all_fields(self, tmp_path):
        recorder = MetricsRecorder(interval_s=0.5, out_path=tmp_path / "m.csv")
        resp = _make_response(
            detections=(
                Detection(0, "pochi", 0.9, (1.0, 2.0, 3.0, 4.0)),
                Detection(1, "taro", 0.8, (5.0, 6.0, 7.0, 8.0)),
            ),
            e2e_time_ms=12.3,
            rtt_ms=3.4,
            backend="trt",
            phase_times_ms={
                "pipeline_preprocess_ms": 1.1,
                "pipeline_inference_ms": 8.2,
                "pipeline_inference_gpu_ms": 7.9,
                "pipeline_postprocess_ms": 0.5,
            },
            gpu_clock_mhz=1800,
            gpu_vram_used_mb=3000,
            gpu_temperature_c=62,
        )
        recorder.maybe_record(resp, now_monotonic=10.0)
        recorder.flush()

        df = pd.read_csv(tmp_path / "m.csv")
        row = df.iloc[0]
        assert row["num_detections"] == 2
        assert row["e2e_time_ms"] == 12.3
        assert row["rtt_ms"] == 3.4
        assert row["backend"] == "trt"
        assert row["phase_preprocess_ms"] == 1.1
        assert row["phase_inference_ms"] == 8.2
        assert row["phase_inference_gpu_ms"] == 7.9
        assert row["phase_postprocess_ms"] == 0.5
        assert row["gpu_clock_mhz"] == 1800
        assert row["gpu_vram_used_mb"] == 3000
        assert row["gpu_temperature_c"] == 62


class TestFlush:
    """flush のテスト."""

    def test_writes_csv_with_header(self, tmp_path):
        recorder = MetricsRecorder(interval_s=0.5, out_path=tmp_path / "m.csv")
        recorder.maybe_record(_make_response(), now_monotonic=10.0)
        recorder.maybe_record(_make_response(), now_monotonic=11.0)

        path = recorder.flush()
        assert path == tmp_path / "m.csv"

        df = pd.read_csv(path)
        assert len(df) == 2
        assert "timestamp_iso" in df.columns
        assert "elapsed_s" in df.columns

    def test_empty_buffer_returns_none(self, tmp_path):
        recorder = MetricsRecorder(interval_s=1.0, out_path=tmp_path / "m.csv")
        assert recorder.flush() is None
        assert not (tmp_path / "m.csv").exists()

    def test_flush_clears_buffer(self, tmp_path):
        recorder = MetricsRecorder(interval_s=0.5, out_path=tmp_path / "m.csv")
        recorder.maybe_record(_make_response(), now_monotonic=10.0)
        recorder.flush()
        assert recorder.row_count == 0

    def test_missing_phase_and_gpu_fields_become_empty(self, tmp_path):
        """サーバーが phase_times_ms / GPU を返さない場合は空セルで書き出す."""
        recorder = MetricsRecorder(interval_s=0.5, out_path=tmp_path / "m.csv")
        resp = _make_response(
            phase_times_ms={},
            gpu_clock_mhz=None,
            gpu_vram_used_mb=None,
            gpu_temperature_c=None,
        )
        recorder.maybe_record(resp, now_monotonic=10.0)
        recorder.flush()

        df = pd.read_csv(tmp_path / "m.csv")
        # 欠損値は NaN として読み込まれる
        assert pd.isna(df.iloc[0]["phase_inference_ms"])
        assert pd.isna(df.iloc[0]["gpu_clock_mhz"])
        assert pd.isna(df.iloc[0]["gpu_vram_used_mb"])
        assert pd.isna(df.iloc[0]["gpu_temperature_c"])

    def test_creates_parent_directory(self, tmp_path):
        """存在しない親ディレクトリも自動作成される."""
        nested = tmp_path / "nested" / "dir" / "m.csv"
        recorder = MetricsRecorder(interval_s=0.5, out_path=nested)
        recorder.maybe_record(_make_response(), now_monotonic=10.0)
        recorder.flush()
        assert nested.exists()


class TestElapsedSeconds:
    """elapsed_s カラムのテスト."""

    def test_elapsed_measured_from_init(self, tmp_path):
        """elapsed_s は recorder 生成時刻起点で計測される."""
        recorder = MetricsRecorder(interval_s=0.5, out_path=tmp_path / "m.csv")
        start = recorder._start_monotonic  # type: ignore[attr-defined]
        recorder.maybe_record(_make_response(), now_monotonic=start + 1.5)
        recorder.maybe_record(_make_response(), now_monotonic=start + 3.0)
        recorder.flush()

        df = pd.read_csv(tmp_path / "m.csv")
        assert df.iloc[0]["elapsed_s"] == pytest.approx(1.5, abs=0.01)
        assert df.iloc[1]["elapsed_s"] == pytest.approx(3.0, abs=0.01)
