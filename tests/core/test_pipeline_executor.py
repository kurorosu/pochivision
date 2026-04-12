"""PipelineExecutor のテスト."""

import json

import cv2
import numpy as np
import pytest

from pochivision.core.pipeline_executor import PipelineExecutor
from pochivision.exceptions import CameraConfigError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import get_processor


class _FailingProcessor(BaseProcessor):
    """常に例外を送出するテスト用プロセッサ."""

    def process(self, image: np.ndarray) -> np.ndarray:
        """意図的に例外を送出する."""
        raise RuntimeError("intentional failure")

    @staticmethod
    def get_default_config() -> dict:
        """空のデフォルト設定を返す."""
        return {}


class _RecordingProcessor(BaseProcessor):
    """受け取った画像を記録するテスト用プロセッサ.

    parallel モードでコピー渡しされていることを検証するために,
    受け取った配列の id と実体を記録する.
    """

    def __init__(self, name: str, config: dict | None = None) -> None:
        """受け取った画像リストを初期化する."""
        super().__init__(name=name, config=config or {})
        self.received_ids: list[int] = []
        self.received_sums: list[int] = []

    def process(self, image: np.ndarray) -> np.ndarray:
        """受け取り画像を記録し, in-place 変更して返す."""
        self.received_ids.append(id(image))
        # 受け取り時点の画素合計を記録 (in-place 0 埋め前).
        self.received_sums.append(int(image.sum()))
        # in-place 変更: 共有されていれば他プロセッサにも影響する.
        image[...] = 0
        return image

    @staticmethod
    def get_default_config() -> dict:
        """空のデフォルト設定を返す."""
        return {}


class _CountingProcessor(BaseProcessor):
    """呼び出し回数を数えるテスト用プロセッサ."""

    def __init__(self, name: str, config: dict | None = None) -> None:
        """呼び出しカウンタを初期化する."""
        super().__init__(name=name, config=config or {})
        self.call_count = 0

    def process(self, image: np.ndarray) -> np.ndarray:
        """呼び出し回数を加算して画像をそのまま返す."""
        self.call_count += 1
        return image

    @staticmethod
    def get_default_config() -> dict:
        """空のデフォルト設定を返す."""
        return {}


def _create_test_image(width: int = 100, height: int = 80) -> np.ndarray:
    """テスト用画像を生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def _minimal_config() -> dict:
    """テスト用の最小設定を返す."""
    return {
        "cameras": {
            "0": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "backend": "DSHOW",
                "label": "Test_Cam",
                "processors": ["resize"],
                "mode": "pipeline",
                "resize": {
                    "width": 50,
                    "preserve_aspect_ratio": False,
                },
            }
        },
        "selected_camera_index": 0,
        "id_interval": 1,
        "preview": {"width": 1280, "height": 720},
        "recording": {
            "select_format": "mjpg",
            "available_formats": {"mjpg": "Motion JPEG"},
        },
    }


class TestPipelineExecutorInit:
    """PipelineExecutor の初期化テスト."""

    def test_invalid_mode_raises_value_error(self, tmp_path):
        """無効なモードで ValueError が発生する."""
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            PipelineExecutor(
                processors=[],
                output_dir=tmp_path,
                mode="invalid",
            )

    def test_parallel_mode(self, tmp_path):
        """parallel モードで初期化できる."""
        executor = PipelineExecutor(
            processors=[],
            output_dir=tmp_path,
            mode="parallel",
        )
        assert executor.mode == "parallel"

    def test_pipeline_mode(self, tmp_path):
        """pipeline モードで初期化できる."""
        executor = PipelineExecutor(
            processors=[],
            output_dir=tmp_path,
            mode="pipeline",
        )
        assert executor.mode == "pipeline"

    def test_default_values(self, tmp_path):
        """デフォルト値が正しく設定される."""
        executor = PipelineExecutor(
            processors=[],
            output_dir=tmp_path,
        )
        assert executor.mode == "parallel"
        assert executor.camera_index == 0
        assert executor.id_interval == 1
        assert executor.config_fps == 30.0


class TestPipelineExecutorFromConfig:
    """PipelineExecutor.from_config のテスト."""

    def test_from_config_creates_instance(self, tmp_path):
        """設定辞書からインスタンスを生成できる."""
        config = _minimal_config()
        executor = PipelineExecutor.from_config(
            config=config,
            output_dir=tmp_path,
            camera_index=0,
            profile_name="0",
        )
        assert executor.mode == "pipeline"
        assert len(executor.processors) == 1
        assert executor.processors[0].name == "resize"

    def test_from_config_missing_profile_raises(self, tmp_path):
        """存在しないプロファイルで例外が発生する."""
        config = _minimal_config()
        with pytest.raises(Exception):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="nonexistent",
            )

    def test_from_config_unregistered_processor_raises(self, tmp_path):
        """未登録プロセッサで ValueError が発生する."""
        config = _minimal_config()
        config["cameras"]["0"]["processors"] = ["nonexistent_processor"]
        with pytest.raises(ValueError, match="not registered"):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="0",
            )


class TestPipelineExecutorFromConfigErrors:
    """PipelineExecutor.from_config のエラーケーステスト."""

    def test_no_cameras_key_raises(self, tmp_path):
        """cameras キーがない設定で CameraConfigError が発生する."""
        config = {"id_interval": 1}
        with pytest.raises(CameraConfigError, match="No camera configurations"):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="0",
            )

    def test_empty_processors_raises(self, tmp_path):
        """プロセッサリストが空で CameraConfigError が発生する."""
        config = _minimal_config()
        config["cameras"]["0"]["processors"] = []
        with pytest.raises(CameraConfigError, match="Empty processors list"):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="0",
            )

    def test_no_processors_key_raises(self, tmp_path):
        """processors キーがない設定で CameraConfigError が発生する."""
        config = _minimal_config()
        del config["cameras"]["0"]["processors"]
        with pytest.raises(CameraConfigError, match="No processors defined"):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="0",
            )

    def test_invalid_mode_raises(self, tmp_path):
        """無効な mode で ValueError が発生する."""
        config = _minimal_config()
        config["cameras"]["0"]["mode"] = "invalid_mode"
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            PipelineExecutor.from_config(
                config=config,
                output_dir=tmp_path,
                profile_name="0",
            )

    def test_from_config_uses_default_mode(self, tmp_path):
        """mode 未指定時にデフォルトの parallel が使用される."""
        config = _minimal_config()
        del config["cameras"]["0"]["mode"]
        executor = PipelineExecutor.from_config(
            config=config,
            output_dir=tmp_path,
            profile_name="0",
        )
        assert executor.mode == "parallel"


class TestPipelineExecutorRun:
    """PipelineExecutor.run のテスト."""

    def test_parallel_mode_saves_images(self, tmp_path):
        """parallel モードで画像が保存される."""
        processor = get_processor(
            "resize", {"width": 50, "preserve_aspect_ratio": False}
        )
        executor = PipelineExecutor(
            processors=[processor],
            output_dir=tmp_path,
            mode="parallel",
        )

        image = _create_test_image()
        executor.run(image)

        assert (tmp_path / "original").exists()
        assert (tmp_path / "resize").exists()
        assert len(list((tmp_path / "original").glob("*"))) == 1
        assert len(list((tmp_path / "resize").glob("*"))) == 1

    def test_pipeline_mode_saves_images(self, tmp_path):
        """pipeline モードで画像が保存される."""
        processor = get_processor(
            "resize", {"width": 50, "preserve_aspect_ratio": False}
        )
        executor = PipelineExecutor(
            processors=[processor],
            output_dir=tmp_path,
            mode="pipeline",
        )

        image = _create_test_image()
        executor.run(image)

        assert (tmp_path / "original").exists()
        assert (tmp_path / "pipeline").exists()
        assert len(list((tmp_path / "original").glob("*"))) == 1
        assert len(list((tmp_path / "pipeline").glob("*"))) == 1

    def test_pipeline_output_is_resized(self, tmp_path):
        """pipeline モードで処理結果が実際にリサイズされている."""
        processor = get_processor(
            "resize", {"width": 50, "preserve_aspect_ratio": False}
        )
        executor = PipelineExecutor(
            processors=[processor],
            output_dir=tmp_path,
            mode="pipeline",
        )

        image = _create_test_image(width=100, height=80)
        executor.run(image)

        pipeline_files = list((tmp_path / "pipeline").glob("*"))
        assert len(pipeline_files) == 1
        result = cv2.imread(str(pipeline_files[0]))
        assert result.shape[1] == 50


class TestPipelineExecutorRobustness:
    """パイプライン堅牢化 (#371, #372) のテスト."""

    def test_pipeline_mode_aborts_on_failure(self, tmp_path):
        """pipeline モードでプロセッサ失敗時に後続が呼ばれない."""
        failing = _FailingProcessor(name="failing", config={})
        after = _CountingProcessor(name="after", config={})
        executor = PipelineExecutor(
            processors=[failing, after],
            output_dir=tmp_path,
            mode="pipeline",
        )

        image = _create_test_image()
        executor.run(image)

        # 失敗後のプロセッサは呼ばれない (パイプライン中断).
        assert after.call_count == 0
        # pipeline ディレクトリは作られない, もしくは空である.
        pipeline_dir = tmp_path / "pipeline"
        if pipeline_dir.exists():
            assert len(list(pipeline_dir.glob("*"))) == 0

    def test_pipeline_mode_failure_does_not_save_stale_result(self, tmp_path):
        """pipeline モードで失敗時に古い result が pipeline に保存されない."""
        resize = get_processor("resize", {"width": 50, "preserve_aspect_ratio": False})
        failing = _FailingProcessor(name="failing", config={})
        executor = PipelineExecutor(
            processors=[resize, failing],
            output_dir=tmp_path,
            mode="pipeline",
        )

        image = _create_test_image()
        executor.run(image)

        pipeline_dir = tmp_path / "pipeline"
        # 直前 resize の結果が "pipeline" として保存されていないこと.
        if pipeline_dir.exists():
            assert len(list(pipeline_dir.glob("*"))) == 0

    def test_parallel_mode_failure_does_not_stop_others(self, tmp_path):
        """parallel モードで一つ失敗しても他は実行される."""
        failing = _FailingProcessor(name="failing", config={})
        after = _CountingProcessor(name="after", config={})
        executor = PipelineExecutor(
            processors=[failing, after],
            output_dir=tmp_path,
            mode="parallel",
        )

        image = _create_test_image()
        executor.run(image)

        assert after.call_count == 1

    def test_parallel_mode_passes_independent_copies(self, tmp_path):
        """parallel モードで各プロセッサに独立したコピーが渡される."""
        rec1 = _RecordingProcessor(name="rec1", config={})
        rec2 = _RecordingProcessor(name="rec2", config={})
        executor = PipelineExecutor(
            processors=[rec1, rec2],
            output_dir=tmp_path,
            mode="parallel",
        )

        image = _create_test_image()
        original_sum = int(image.sum())
        executor.run(image)

        # 入力画像は外部から変更されていない.
        assert int(image.sum()) == original_sum
        # 2 つのプロセッサが受け取った配列は異なる id (別オブジェクト).
        assert rec1.received_ids[0] != rec2.received_ids[0]
        # 1 つ目が in-place で 0 埋めしても, 2 つ目は元の画素を受け取る.
        assert rec1.received_sums[0] == original_sum
        assert rec2.received_sums[0] == original_sum
