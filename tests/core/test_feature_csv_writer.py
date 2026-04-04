"""FeatureCSVWriter のテスト."""

import csv

from pochivision.core.feature_csv_writer import FeatureCSVWriter


class TestFeatureCSVWriter:
    """FeatureCSVWriter のテストクラス."""

    def _make_config(self, **overrides):
        """テスト用の設定辞書を生成する."""
        config = {"output_settings": {"output_filename": "features.csv"}}
        config["output_settings"].update(overrides)
        return config

    def test_save_wide_csv_basic(self, tmp_path):
        """基本的な横持ち CSV 出力."""
        config = self._make_config()
        writer = FeatureCSVWriter(tmp_path, config)

        results = [
            {"filename": "img1.png", "feat_a": 1.0, "feat_b": 2.0},
            {"filename": "img2.png", "feat_a": 3.0, "feat_b": 4.0},
        ]
        writer.save_wide_csv(results)

        csv_path = tmp_path / "features.csv"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["filename"] == "img1.png"
        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == "filename"

    def test_save_wide_csv_with_class_column(self, tmp_path):
        """class カラム付きの横持ち CSV 出力."""
        config = self._make_config(
            class_extraction={"enabled": True, "column_name": "label"}
        )
        writer = FeatureCSVWriter(tmp_path, config)

        results = [
            {"filename": "img1.png", "label": "cat", "feat_a": 1.0},
        ]
        writer.save_wide_csv(results)

        with open(tmp_path / "features.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == "filename"
        assert reader.fieldnames[1] == "label"
        assert rows[0]["label"] == "cat"

    def test_save_wide_csv_empty_results(self, tmp_path):
        """空の結果では CSV ファイルが作成されない."""
        config = self._make_config()
        writer = FeatureCSVWriter(tmp_path, config)
        writer.save_wide_csv([])

        assert not (tmp_path / "features.csv").exists()

    def test_save_long_csv_basic(self, tmp_path):
        """基本的な縦持ち CSV 出力."""
        config = self._make_config(long_format_filename="features_long.csv")
        writer = FeatureCSVWriter(tmp_path, config)

        results = [
            {"filename": "img1.png", "feat_a": 1.0, "feat_b": 2.0},
        ]
        writer.save_long_csv(results)

        csv_path = tmp_path / "features_long.csv"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        feature_names = {row["feature_name"] for row in rows}
        assert feature_names == {"feat_a", "feat_b"}

    def test_save_wide_csv_custom_separator(self, tmp_path):
        """カスタム区切り文字での CSV 出力."""
        config = self._make_config(csv_separator="\t")
        writer = FeatureCSVWriter(tmp_path, config)

        results = [{"filename": "img1.png", "feat_a": 1.0}]
        writer.save_wide_csv(results)

        content = (tmp_path / "features.csv").read_text(encoding="utf-8")
        assert "\t" in content

    def test_header_order_filename_first(self, tmp_path):
        """ヘッダー順序: filename が先頭."""
        config = self._make_config()
        writer = FeatureCSVWriter(tmp_path, config)

        results = [{"z_feat": 1.0, "a_feat": 2.0, "filename": "img.png"}]
        writer.save_wide_csv(results)

        with open(tmp_path / "features.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            list(reader)

        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == "filename"
        assert reader.fieldnames[1] == "a_feat"
        assert reader.fieldnames[2] == "z_feat"
