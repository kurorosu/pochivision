"""_save_inference_frame のテスト."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from pochivision.capture_runner.viewer import LivePreviewRunner
from pochivision.request.api.inference.client import InferenceClient
from pochivision.request.api.inference.config import ResizeConfig


def _make_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """テスト用フレームを生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def _make_runner(
    tmp_path: Path,
    save_frame: bool = True,
    resize: ResizeConfig | None = None,
) -> LivePreviewRunner:
    """テスト用の LivePreviewRunner を生成する."""
    cap = MagicMock()
    pipeline = MagicMock()
    pipeline.output_dir = tmp_path

    client = InferenceClient(
        base_url="http://localhost:8000",
        save_frame=save_frame,
        resize=resize,
    )

    runner = LivePreviewRunner(cap, pipeline, inference_client=client)
    return runner


class TestSaveInferenceFrame:
    """_save_inference_frame のテスト."""

    def test_saves_file(self, tmp_path):
        """save_frame=True のときファイルが保存される."""
        runner = _make_runner(tmp_path, save_frame=True)
        frame = _make_frame()

        runner._save_inference_frame(frame)

        inference_dir = tmp_path / "inference"
        assert inference_dir.exists()
        saved_files = list(inference_dir.glob("infer_*.png"))
        assert len(saved_files) == 1

        saved = cv2.imread(str(saved_files[0]))
        assert saved is not None

        runner.inference_client.close()

    def test_disabled(self, tmp_path):
        """save_frame=False のときは保存されない."""
        runner = _make_runner(tmp_path, save_frame=False)
        frame = _make_frame()

        runner._save_inference_frame(frame)

        inference_dir = tmp_path / "inference"
        assert not inference_dir.exists()

        runner.inference_client.close()

    def test_none_client(self, tmp_path):
        """inference_client=None のときは保存されない."""
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        runner = LivePreviewRunner(cap, pipeline)
        runner._save_inference_frame(_make_frame())

        inference_dir = tmp_path / "inference"
        assert not inference_dir.exists()

    def test_saves_resized_frame(self, tmp_path):
        """resize 設定ありの場合, リサイズ後の画像が保存される."""
        resize = ResizeConfig(width=64, height=64)
        runner = _make_runner(tmp_path, save_frame=True, resize=resize)
        frame = _make_frame(480, 640)

        runner._save_inference_frame(frame)

        saved_files = list((tmp_path / "inference").glob("infer_*.png"))
        assert len(saved_files) == 1

        saved = cv2.imread(str(saved_files[0]))
        assert saved.shape == (64, 64, 3)

        runner.inference_client.close()

    def test_multiple_saves(self, tmp_path):
        """複数回保存でファイル名が重複しない."""
        runner = _make_runner(tmp_path, save_frame=True)
        frame = _make_frame()

        runner._save_inference_frame(frame)
        runner._save_inference_frame(frame)

        saved_files = list((tmp_path / "inference").glob("infer_*.png"))
        assert len(saved_files) == 2

        runner.inference_client.close()

    def test_directory_auto_created(self, tmp_path):
        """inference ディレクトリが自動作成される."""
        runner = _make_runner(tmp_path, save_frame=True)
        frame = _make_frame()

        assert not (tmp_path / "inference").exists()
        runner._save_inference_frame(frame)
        assert (tmp_path / "inference").exists()

        runner.inference_client.close()

    def test_returns_filename(self, tmp_path):
        """save_frame=True のとき保存されたファイル名を返す."""
        runner = _make_runner(tmp_path, save_frame=True)
        frame = _make_frame()

        result = runner._save_inference_frame(frame)
        assert result is not None
        assert result.startswith("infer_")
        assert result.endswith(".png")

        runner.inference_client.close()

    def test_returns_none_when_disabled(self, tmp_path):
        """save_frame=False のとき None を返す."""
        runner = _make_runner(tmp_path, save_frame=False)
        frame = _make_frame()

        result = runner._save_inference_frame(frame)
        assert result is None

        runner.inference_client.close()

    def test_returns_none_when_no_client(self, tmp_path):
        """inference_client=None のとき None を返す."""
        cap = MagicMock()
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path

        runner = LivePreviewRunner(cap, pipeline)
        result = runner._save_inference_frame(_make_frame())
        assert result is None
