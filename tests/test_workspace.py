"""OutputManager のテスト."""

import re

from pochivision.workspace import OutputManager


class TestOutputManager:
    """OutputManager の動作テスト."""

    def test_create_output_dir_creates_directory(self, tmp_path):
        """create_output_dir がディレクトリを作成する."""
        om = OutputManager(root=str(tmp_path))
        output_dir = om.create_output_dir("capture")

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_create_output_dir_path_structure(self, tmp_path):
        """出力パスが {root}/{command_type}/YYYYMMDD_{suffix}/ 構造になる."""
        om = OutputManager(root=str(tmp_path))
        output_dir = om.create_output_dir("capture")

        assert output_dir.parent.name == "capture"
        assert output_dir.parent.parent == tmp_path
        assert re.match(r"\d{8}_\d+", output_dir.name)

    def test_create_output_dir_suffix_increments(self, tmp_path):
        """同日に複数回呼ぶとサフィックスがインクリメントされる."""
        om = OutputManager(root=str(tmp_path))
        dir1 = om.create_output_dir("features")
        dir2 = om.create_output_dir("features")

        assert dir1 != dir2
        suffix1 = int(dir1.name.split("_")[-1])
        suffix2 = int(dir2.name.split("_")[-1])
        assert suffix2 == suffix1 + 1

    def test_create_output_dir_different_command_types(self, tmp_path):
        """コマンド種別ごとに独立したディレクトリが作成される."""
        om = OutputManager(root=str(tmp_path))
        capture_dir = om.create_output_dir("capture")
        features_dir = om.create_output_dir("features")
        processed_dir = om.create_output_dir("processed")
        aggregated_dir = om.create_output_dir("aggregated")

        assert capture_dir.parent.name == "capture"
        assert features_dir.parent.name == "features"
        assert processed_dir.parent.name == "processed"
        assert aggregated_dir.parent.name == "aggregated"

    def test_default_root_is_outputs(self):
        """デフォルトのルートディレクトリが 'outputs' である."""
        om = OutputManager()
        assert str(om.root) == "outputs"

    def test_custom_root(self, tmp_path):
        """カスタムルートディレクトリが設定できる."""
        custom_root = tmp_path / "my_outputs"
        om = OutputManager(root=str(custom_root))
        output_dir = om.create_output_dir("capture")

        assert custom_root in output_dir.parents

    def test_get_next_suffix_empty_dir(self, tmp_path):
        """空ディレクトリではサフィックスが 0 を返す."""
        om = OutputManager(root=str(tmp_path))
        assert om._get_next_suffix(tmp_path, "20260402") == 0

    def test_get_next_suffix_with_existing(self, tmp_path):
        """既存ディレクトリがある場合, 次のサフィックスを返す."""
        om = OutputManager(root=str(tmp_path))
        (tmp_path / "20260402_0").mkdir()
        (tmp_path / "20260402_1").mkdir()
        (tmp_path / "20260402_2").mkdir()

        assert om._get_next_suffix(tmp_path, "20260402") == 3

    def test_get_next_suffix_nonexistent_dir(self, tmp_path):
        """存在しないディレクトリでは 0 を返す."""
        om = OutputManager(root=str(tmp_path))
        assert om._get_next_suffix(tmp_path / "nonexistent", "20260402") == 0

    def test_get_next_suffix_ignores_files(self, tmp_path):
        """ファイルは無視してディレクトリのみカウントする."""
        om = OutputManager(root=str(tmp_path))
        (tmp_path / "20260402_0").mkdir()
        (tmp_path / "20260402_1.txt").touch()

        assert om._get_next_suffix(tmp_path, "20260402") == 1

    def test_get_next_suffix_ignores_other_dates(self, tmp_path):
        """異なる日付のディレクトリは無視する."""
        om = OutputManager(root=str(tmp_path))
        (tmp_path / "20260401_0").mkdir()
        (tmp_path / "20260401_1").mkdir()

        assert om._get_next_suffix(tmp_path, "20260402") == 0
