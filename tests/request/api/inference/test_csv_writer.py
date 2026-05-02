"""InferenceCsvWriter のテスト."""

import csv
import re
from pathlib import Path

from pochivision.request.api.inference.csv_writer import InferenceCsvWriter
from pochivision.request.api.inference.models import PredictResponse


def _make_result(**overrides: object) -> PredictResponse:
    """テスト用の PredictResponse を生成する."""
    defaults: dict[str, object] = {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 0.95,
        "probabilities": [0.95, 0.05],
        "e2e_time_ms": 12.3,
        "backend": "onnx",
        "rtt_ms": 50.0,
    }
    defaults.update(overrides)
    return PredictResponse(**defaults)  # type: ignore[arg-type]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """CSV ファイルを辞書リストとして読み込む."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestInferenceCsvWriter:
    """InferenceCsvWriter のテスト."""

    def test_creates_file_with_header(self, tmp_path):
        """初回書き込みでヘッダ行付きの CSV が作成される."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result())

        assert writer.csv_path.exists()
        rows = _read_csv(writer.csv_path)
        assert len(rows) == 1
        assert "timestamp" in rows[0]
        assert "class_id" in rows[0]
        assert "image_file" in rows[0]

    def test_appends_rows(self, tmp_path):
        """複数回書き込みで行が追記される."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result(class_name="cat"))
        writer.write_row(_make_result(class_name="dog"))

        rows = _read_csv(writer.csv_path)
        assert len(rows) == 2
        assert rows[0]["class_name"] == "cat"
        assert rows[1]["class_name"] == "dog"

    def test_result_fields(self, tmp_path):
        """PredictResponse のフィールドが正しく書き込まれる."""
        result = _make_result(
            class_id=1,
            class_name="dog",
            confidence=0.85,
            e2e_time_ms=10.5,
            backend="torch",
            rtt_ms=30.0,
        )
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(result)

        rows = _read_csv(writer.csv_path)
        row = rows[0]
        assert row["class_id"] == "1"
        assert row["class_name"] == "dog"
        assert row["confidence"] == "0.85"
        assert row["e2e_time_ms"] == "10.5"
        assert row["backend"] == "torch"
        assert row["rtt_ms"] == "30.0"

    def test_image_file_included(self, tmp_path):
        """image_file が正しく書き込まれる."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result(), image_file="infer_20260405_120000_000000.png")

        rows = _read_csv(writer.csv_path)
        assert rows[0]["image_file"] == "infer_20260405_120000_000000.png"

    def test_image_file_empty_when_none(self, tmp_path):
        """image_file=None のとき空文字になる."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result(), image_file=None)

        rows = _read_csv(writer.csv_path)
        assert rows[0]["image_file"] == ""

    def test_timestamp_format(self, tmp_path):
        """timestamp が正しい形式で記録される."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result())

        rows = _read_csv(writer.csv_path)
        ts = rows[0]["timestamp"]
        # YYYY-MM-DD HH:MM:SS.ffffff 形式
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}", ts)

    def test_directory_auto_created(self, tmp_path):
        """出力ディレクトリが自動作成される."""
        nested_dir = tmp_path / "a" / "b"
        writer = InferenceCsvWriter(nested_dir)
        writer.write_row(_make_result())

        assert nested_dir.exists()
        assert writer.csv_path.exists()

    def test_no_duplicate_header(self, tmp_path):
        """複数回書き込みでヘッダが重複しない."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result())
        writer.write_row(_make_result())

        lines = writer.csv_path.read_text(encoding="utf-8").strip().split("\n")
        # ヘッダ 1 行 + データ 2 行 = 3 行
        assert len(lines) == 3
        assert lines[0].startswith("timestamp")

    def test_csv_filename(self, tmp_path):
        """CSV ファイル名が inference_results.csv である."""
        writer = InferenceCsvWriter(tmp_path)
        assert writer.csv_path.name == "inference_results.csv"

    def test_phase_and_gpu_columns_present(self, tmp_path):
        """phase / GPU カラムがヘッダに含まれる."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result())

        rows = _read_csv(writer.csv_path)
        row = rows[0]
        for column in (
            "total_ms",
            "api_preprocess_ms",
            "phase_preprocess_ms",
            "phase_inference_ms",
            "phase_inference_gpu_ms",
            "phase_postprocess_ms",
            "api_postprocess_ms",
            "gpu_clock_mhz",
            "gpu_vram_used_mb",
            "gpu_temperature_c",
        ):
            assert column in row

    def test_phase_and_gpu_values_written(self, tmp_path):
        """phase_times_ms / GPU メトリクスの値が CSV に書き出される."""
        result = _make_result(
            total_ms=12.5,
            phase_times_ms={
                "api_preprocess_ms": 0.4,
                "pipeline_preprocess_ms": 1.2,
                "pipeline_inference_ms": 2.5,
                "pipeline_inference_gpu_ms": 2.1,
                "pipeline_postprocess_ms": 0.8,
                "api_postprocess_ms": 0.1,
            },
            gpu_clock_mhz=1770,
            gpu_vram_used_mb=2048,
            gpu_temperature_c=55,
        )
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(result)

        rows = _read_csv(writer.csv_path)
        row = rows[0]
        assert row["total_ms"] == "12.5"
        assert row["api_preprocess_ms"] == "0.4"
        assert row["phase_preprocess_ms"] == "1.2"
        assert row["phase_inference_ms"] == "2.5"
        assert row["phase_inference_gpu_ms"] == "2.1"
        assert row["phase_postprocess_ms"] == "0.8"
        assert row["api_postprocess_ms"] == "0.1"
        assert row["gpu_clock_mhz"] == "1770"
        assert row["gpu_vram_used_mb"] == "2048"
        assert row["gpu_temperature_c"] == "55"

    def test_phase_and_gpu_empty_when_missing(self, tmp_path):
        """phase_times_ms 空 / GPU None のとき該当セルが空文字になる."""
        writer = InferenceCsvWriter(tmp_path)
        writer.write_row(_make_result())

        rows = _read_csv(writer.csv_path)
        row = rows[0]
        assert row["api_preprocess_ms"] == ""
        assert row["phase_inference_ms"] == ""
        assert row["gpu_clock_mhz"] == ""
        assert row["gpu_vram_used_mb"] == ""
        assert row["gpu_temperature_c"] == ""
