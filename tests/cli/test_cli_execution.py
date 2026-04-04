"""CLI サブコマンドの実行テスト."""

import json

import cv2
import numpy as np
from click.testing import CliRunner

from pochivision.cli.main import main


def _create_test_image(path, width: int = 10, height: int = 10) -> None:
    """テスト用画像ファイルを作成する."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _create_extractor_config(tmp_path, input_dir: str) -> str:
    """テスト用 extractor_config.json を作成する."""
    config = {
        "input_directory": input_dir,
        "output_format": "csv",
        "extractors": ["brightness"],
        "feature_extractors": {
            "brightness": {"color_mode": "gray", "exclude_zero_pixels": False}
        },
        "file_filters": {"extensions": [".jpg", ".png"], "case_sensitive": False},
        "output_settings": {
            "include_timestamp": False,
            "csv_separator": ",",
            "output_filename": "features.csv",
            "enable_long_format": False,
        },
    }
    config_path = tmp_path / "extractor_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _create_process_config(tmp_path) -> str:
    """テスト用 config.json を作成する."""
    config = {
        "cameras": {
            "test_profile": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "backend": "DSHOW",
                "label": "Test_Cam",
                "processors": ["resize"],
                "mode": "pipeline",
                "resize": {"width": 50, "preserve_aspect_ratio": False},
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
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


class TestExtractCommand:
    """extract コマンドの実行テスト."""

    def test_nonexistent_config_file(self, tmp_path):
        """存在しない設定ファイルでエラーになる."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "extract",
                "-c",
                "nonexistent.json",
            ],
        )
        assert result.exit_code != 0

    def test_extract_with_valid_config(self, tmp_path):
        """有効な設定で特徴量抽出が実行される."""
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        _create_test_image(input_dir / "test1.jpg")
        _create_test_image(input_dir / "test2.jpg")

        config_path = _create_extractor_config(tmp_path, str(input_dir))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--output-root", str(tmp_path / "output"), "extract", "-c", config_path],
        )
        assert result.exit_code == 0

        # 出力ディレクトリに CSV が生成される
        output_dir = tmp_path / "output" / "features"
        assert output_dir.exists()
        csv_files = list(output_dir.rglob("features.csv"))
        assert len(csv_files) == 1

    def test_extract_empty_input_directory(self, tmp_path):
        """空の入力ディレクトリでも正常終了する."""
        input_dir = tmp_path / "empty_data"
        input_dir.mkdir()

        config_path = _create_extractor_config(tmp_path, str(input_dir))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--output-root", str(tmp_path / "output"), "extract", "-c", config_path],
        )
        assert result.exit_code == 0

    def test_extract_nonexistent_input_directory(self, tmp_path):
        """存在しない入力ディレクトリでエラーになる."""
        config_path = _create_extractor_config(
            tmp_path, str(tmp_path / "nonexistent_dir")
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--output-root", str(tmp_path / "output"), "extract", "-c", config_path],
        )
        assert result.exit_code != 0


class TestProcessCommand:
    """process コマンドの実行テスト."""

    def test_missing_required_options(self):
        """必須オプション未指定でエラーになる."""
        runner = CliRunner()
        result = runner.invoke(main, ["process"])
        assert result.exit_code != 0

    def test_missing_input_option(self):
        """--input 未指定でエラーになる."""
        runner = CliRunner()
        result = runner.invoke(main, ["process", "-p", "0"])
        assert result.exit_code != 0

    def test_missing_profile_option(self, tmp_path):
        """--profile 未指定でエラーになる."""
        runner = CliRunner()
        result = runner.invoke(main, ["process", "-i", str(tmp_path)])
        assert result.exit_code != 0

    def test_list_profiles(self, tmp_path):
        """--list-profiles でプロファイル一覧が表示される."""
        config_path = _create_process_config(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "process",
                "-c",
                config_path,
                "-i",
                str(tmp_path),
                "-p",
                "0",
                "--list-profiles",
            ],
        )
        assert result.exit_code == 0

    def test_process_with_valid_config(self, tmp_path):
        """有効な設定で画像処理が実行される."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "test1.jpg", width=100, height=80)

        config_path = _create_process_config(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "process",
                "-c",
                config_path,
                "-i",
                str(input_dir),
                "-p",
                "test_profile",
            ],
        )
        assert result.exit_code == 0

        # 出力ディレクトリに処理結果がある
        output_dir = tmp_path / "output" / "processed"
        assert output_dir.exists()

    def test_process_nonexistent_profile(self, tmp_path):
        """存在しないプロファイルでエラーになる."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        config_path = _create_process_config(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "process",
                "-c",
                config_path,
                "-i",
                str(input_dir),
                "-p",
                "nonexistent_profile",
            ],
        )
        assert result.exit_code != 0


class TestAggregateCommand:
    """aggregate コマンドの実行テスト."""

    def test_aggregate_empty_directory(self, tmp_path):
        """空のディレクトリで正常終了する."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "aggregate",
                "-i",
                str(input_dir),
            ],
        )
        assert result.exit_code == 0
        assert "No images were processed" in result.output

    def test_aggregate_copy_mode(self, tmp_path):
        """copy モードで画像が集約される."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image(input_dir / "20260401_0" / "resize" / "img1.jpg")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "aggregate",
                "-i",
                str(input_dir),
                "-m",
                "copy",
            ],
        )
        assert result.exit_code == 0
        # 元ファイルが残っている (copy)
        assert (input_dir / "20260401_0" / "resize" / "img1.jpg").exists()

    def test_aggregate_move_mode(self, tmp_path):
        """move モードで画像が移動される."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image(input_dir / "20260401_0" / "resize" / "img1.jpg")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(tmp_path / "output"),
                "aggregate",
                "-i",
                str(input_dir),
                "-m",
                "move",
            ],
        )
        assert result.exit_code == 0
        # 元ファイルが消えている (move)
        assert not (input_dir / "20260401_0" / "resize" / "img1.jpg").exists()


class TestOutputRootOption:
    """--output-root オプションの伝播テスト."""

    def test_output_root_propagates_to_extract(self, tmp_path):
        """--output-root が extract に伝播される."""
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        _create_test_image(input_dir / "test.jpg")

        config_path = _create_extractor_config(tmp_path, str(input_dir))
        custom_output = tmp_path / "custom_output"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--output-root", str(custom_output), "extract", "-c", config_path],
        )
        assert result.exit_code == 0
        assert (custom_output / "features").exists()

    def test_output_root_propagates_to_aggregate(self, tmp_path):
        """--output-root が aggregate に伝播される."""
        input_dir = tmp_path / "input"
        (input_dir / "20260401_0" / "resize").mkdir(parents=True)
        _create_test_image(input_dir / "20260401_0" / "resize" / "img.jpg")

        custom_output = tmp_path / "custom_output"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--output-root",
                str(custom_output),
                "aggregate",
                "-i",
                str(input_dir),
            ],
        )
        assert result.exit_code == 0
        assert (custom_output / "aggregated").exists()
