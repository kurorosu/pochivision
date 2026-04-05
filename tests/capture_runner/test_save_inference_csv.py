"""_save_inference_csv のテスト."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

from pochivision.capture_runner.viewer import LivePreviewRunner
from pochivision.request.api.inference.client import InferenceClient
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


def _make_runner(tmp_path: Path, save_csv: bool = True) -> LivePreviewRunner:
    """テスト用の LivePreviewRunner を生成する."""
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path

    client = InferenceClient(
        base_url="http://localhost:8000",
        save_csv=save_csv,
    )

    runner = LivePreviewRunner(cap, pipeline, inference_client=client)
    return runner


def _read_csv(path: Path) -> list[dict[str, str]]:
    """CSV ファイルを辞書リストとして読み込む."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestSaveInferenceCsv:
    """_save_inference_csv のテスト."""

    def test_saves_csv(self, tmp_path):
        """save_csv=True のとき CSV が保存される."""
        runner = _make_runner(tmp_path, save_csv=True)
        result = _make_result()

        runner._save_inference_csv(result, None)

        csv_path = tmp_path / "inference" / "inference_results.csv"
        assert csv_path.exists()
        rows = _read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["class_name"] == "cat"

        runner.inference_client.close()

    def test_disabled(self, tmp_path):
        """save_csv=False のときは保存されない."""
        runner = _make_runner(tmp_path, save_csv=False)
        result = _make_result()

        runner._save_inference_csv(result, None)

        csv_path = tmp_path / "inference" / "inference_results.csv"
        assert not csv_path.exists()

        runner.inference_client.close()

    def test_none_client(self, tmp_path):
        """inference_client=None のときは保存されない."""
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        runner = LivePreviewRunner(cap, pipeline)
        runner._save_inference_csv(_make_result(), None)

        csv_path = tmp_path / "inference" / "inference_results.csv"
        assert not csv_path.exists()

    def test_with_image_file(self, tmp_path):
        """image_file が CSV に記録される."""
        runner = _make_runner(tmp_path, save_csv=True)
        result = _make_result()

        runner._save_inference_csv(result, "infer_20260405.png")

        csv_path = tmp_path / "inference" / "inference_results.csv"
        rows = _read_csv(csv_path)
        assert rows[0]["image_file"] == "infer_20260405.png"

        runner.inference_client.close()

    def test_multiple_saves(self, tmp_path):
        """複数回の保存で行が追記される."""
        runner = _make_runner(tmp_path, save_csv=True)

        runner._save_inference_csv(_make_result(class_name="cat"), None)
        runner._save_inference_csv(_make_result(class_name="dog"), None)

        csv_path = tmp_path / "inference" / "inference_results.csv"
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert rows[0]["class_name"] == "cat"
        assert rows[1]["class_name"] == "dog"

        runner.inference_client.close()
