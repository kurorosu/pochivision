"""FeatureExtractionRunner の統合テスト."""

import csv
import json

import cv2
import numpy as np

from pochivision.core.feature_extraction import FeatureExtractionRunner
from pochivision.workspace import OutputManager


def _create_test_image(path, width=100, height=80):
    """テスト用画像を作成して保存する."""
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _create_config(tmp_path, input_dir, **overrides):
    """テスト用の設定ファイルを生成する."""
    config = {
        "input_directory": str(input_dir),
        "output_format": "csv",
        "extractors": ["brightness", "rgb"],
        "feature_extractors": {
            "brightness": {"color_mode": "gray", "exclude_zero_pixels": False},
            "rgb": {"exclude_black_pixels": False},
        },
        "file_filters": {
            "extensions": [".png"],
            "case_sensitive": False,
        },
        "output_settings": {
            "output_filename": "features.csv",
            "csv_separator": ",",
        },
    }
    config.update(overrides)
    config_path = tmp_path / "extractor_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


class TestFeatureExtractionRunner:
    """FeatureExtractionRunner のテストクラス."""

    def test_run_produces_csv(self, tmp_path):
        """画像から特徴量を抽出し CSV を生成する."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")
        _create_test_image(input_dir / "img2.png")

        config_path = _create_config(tmp_path, input_dir)
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        csv_path = runner.output_dir / "features.csv"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert "filename" in reader.fieldnames
        # brightness と rgb の特徴量カラムが存在する
        feature_cols = [c for c in reader.fieldnames if c != "filename"]
        assert any("brightness" in c for c in feature_cols)
        assert any("rgb" in c for c in feature_cols)

    def test_run_with_class_extraction(self, tmp_path):
        """class_extraction 有効時に class カラムが CSV に含まれる."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "cat_001_sample.png")

        config_path = _create_config(
            tmp_path,
            input_dir,
            output_settings={
                "output_filename": "features.csv",
                "csv_separator": ",",
                "class_extraction": {
                    "enabled": True,
                    "delimiter": "_",
                    "position": 0,
                    "column_name": "label",
                },
            },
        )
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        with open(runner.output_dir / "features.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "label" in reader.fieldnames
        assert rows[0]["label"] == "cat"

    def test_run_with_long_format(self, tmp_path):
        """enable_long_format 有効時に縦持ち CSV が生成される."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(
            tmp_path,
            input_dir,
            output_settings={
                "output_filename": "features.csv",
                "csv_separator": ",",
                "enable_long_format": True,
                "long_format_filename": "features_long.csv",
            },
        )
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        long_csv_path = runner.output_dir / "features_long.csv"
        assert long_csv_path.exists()

        with open(long_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        assert "feature_name" in reader.fieldnames
        assert "feature_value" in reader.fieldnames

    def test_empty_input_directory(self, tmp_path):
        """空の入力ディレクトリでは CSV が生成されない."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        config_path = _create_config(tmp_path, input_dir)
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        assert not (runner.output_dir / "features.csv").exists()

    def test_config_copied_to_output(self, tmp_path):
        """設定ファイルが出力ディレクトリにコピーされる."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(tmp_path, input_dir)
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        copied_config = runner.output_dir / "extractor_config.json"
        assert copied_config.exists()

    def test_invalid_extractor_name_skipped(self, tmp_path):
        """無効なエクストラクタ名はスキップされ, 有効なもので実行される."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _create_test_image(input_dir / "img1.png")

        config_path = _create_config(
            tmp_path,
            input_dir,
            extractors=["brightness", "nonexistent_extractor"],
        )
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)
        runner.run()

        csv_path = runner.output_dir / "features.csv"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert any("brightness" in c for c in reader.fieldnames)

    def test_nonexistent_input_directory_raises(self, tmp_path):
        """存在しない入力ディレクトリで FileNotFoundError が発生する."""
        input_dir = tmp_path / "nonexistent"
        config_path = _create_config(tmp_path, input_dir)
        output_manager = OutputManager(str(tmp_path))
        runner = FeatureExtractionRunner(config_path, output_manager)

        try:
            runner.run()
            assert False, "FileNotFoundError が発生すべき"
        except FileNotFoundError:
            pass
