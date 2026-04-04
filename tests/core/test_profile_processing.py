"""ProfileProcessor のテスト."""

import json

import cv2
import numpy as np

from pochivision.core.profile_processing import ProfileProcessor
from pochivision.workspace import OutputManager


def _create_test_image(path, width=100, height=80):
    """テスト用画像を作成して保存する."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _create_config(tmp_path, processors=None, mode="pipeline", **extra_profile):
    """バリデーション通過する config.json を生成する."""
    if processors is None:
        processors = ["resize"]

    profile = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "backend": "any",
        "processors": processors,
        "mode": mode,
        "resize": {"width": 50, "preserve_aspect_ratio": False},
    }
    profile.update(extra_profile)

    config = {
        "cameras": {"test_profile": profile},
        "selected_camera_index": 0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


class TestProfileProcessor:
    """ProfileProcessor のテストクラス."""

    def test_process_directory_pipeline_mode(self, tmp_path):
        """pipeline モードでディレクトリ内の画像を処理する."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")
        _create_test_image(input_dir / "img2.png")

        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir))

        # processed ディレクトリに処理済み画像が存在する
        processed_files = list(
            (output_manager.create_output_dir("processed") / "processed").glob("*")
        )
        # OutputManager が別の output_dir を作るので processor の出力先を確認
        # process_directory 内で create_output_dir が呼ばれるため
        # tmp_path 以下に processed ディレクトリが存在するか確認
        output_dirs = list(tmp_path.rglob("processed"))
        assert len(output_dirs) > 0

        # 処理済み画像が存在する
        processed_images = list(tmp_path.rglob("processed/*.png"))
        assert len(processed_images) == 2

        # リサイズされている
        result = cv2.imread(str(processed_images[0]))
        assert result.shape[1] == 50

    def test_process_directory_saves_original(self, tmp_path):
        """save_original=True で元画像がコピーされる."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir), save_original=True)

        original_images = list(tmp_path.rglob("original/*.png"))
        assert len(original_images) == 1

    def test_process_directory_no_original(self, tmp_path):
        """save_original=False で元画像がコピーされない."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir), save_original=False)

        original_dirs = list(tmp_path.rglob("original"))
        assert len(original_dirs) == 0

    def test_profile_info_saved(self, tmp_path):
        """profile_info.json が出力ディレクトリに保存される."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir))

        info_files = list(tmp_path.rglob("profile_info.json"))
        assert len(info_files) == 1

        with open(info_files[0], encoding="utf-8") as f:
            info = json.load(f)
        assert info["profile_name"] == "test_profile"

    def test_invalid_profile_raises(self, tmp_path):
        """存在しないプロファイル名で ValueError が発生する."""
        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))

        try:
            ProfileProcessor(config_path, "nonexistent", output_manager)
            assert False, "ValueError が発生すべき"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_empty_input_directory(self, tmp_path):
        """空の入力ディレクトリでは処理済み画像が生成されない."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        config_path = _create_config(tmp_path)
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir))

        processed_images = list(tmp_path.rglob("processed/*.png"))
        assert len(processed_images) == 0

    def test_parallel_mode(self, tmp_path):
        """parallel モードで画像が処理される."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(tmp_path, mode="parallel")
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.process_directory(str(input_dir))

        processed_images = list(tmp_path.rglob("processed/*.png"))
        assert len(processed_images) == 1


class TestProcessImageEdgeCases:
    """ProfileProcessor._process_image のエッジケーステスト."""

    def test_unknown_mode_returns_none(self, tmp_path):
        """不明な mode 値で _process_image が None を返す."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img_path = input_dir / "img1.png"
        _create_test_image(img_path)

        config_path = _create_config(tmp_path, mode="pipeline")
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        # mode を強制的に不正な値に変更
        processor.profile_config["mode"] = "unknown_mode"

        result = processor._process_image(img_path)
        assert result is None

    def test_empty_processors_pipeline_returns_original(self, tmp_path):
        """プロセッサリストが空の pipeline モードで元画像が返る."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img_path = input_dir / "img1.png"
        _create_test_image(img_path, width=100, height=80)

        config_path = _create_config(tmp_path, processors=["resize"], mode="pipeline")
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        # プロセッサリストを空にする
        processor.processors = []

        result = processor._process_image(img_path)
        assert result is not None
        assert result.shape[:2] == (80, 100)

    def test_empty_processors_parallel_returns_none(self, tmp_path):
        """プロセッサリストが空の parallel モードで None が返る."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img_path = input_dir / "img1.png"
        _create_test_image(img_path)

        config_path = _create_config(tmp_path, processors=["resize"], mode="parallel")
        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(config_path, "test_profile", output_manager)
        processor.processors = []

        result = processor._process_image(img_path)
        assert result is None

    def test_parallel_mode_multiple_processors(self, tmp_path):
        """parallel モードで複数プロセッサの最後の結果が返る."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img_path = input_dir / "img1.png"
        _create_test_image(img_path, width=100, height=80)

        profile = {
            "width": 640,
            "height": 480,
            "fps": 30,
            "backend": "any",
            "processors": ["resize", "equalize"],
            "mode": "parallel",
            "resize": {"width": 50, "preserve_aspect_ratio": False},
            "equalize": {"color_mode": "gray"},
        }
        config = {
            "cameras": {"test_profile": profile},
            "selected_camera_index": 0,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        output_manager = OutputManager(str(tmp_path))
        processor = ProfileProcessor(str(config_path), "test_profile", output_manager)

        result = processor._process_image(img_path)
        assert result is not None
