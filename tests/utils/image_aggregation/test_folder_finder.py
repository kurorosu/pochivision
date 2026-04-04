"""ProcessorFolderFinder のテスト."""

from pochivision.utils.image_aggregation.folder_finder import ProcessorFolderFinder


class TestProcessorFolderFinder:
    """ProcessorFolderFinder のテスト."""

    def test_find_processor_types(self, tmp_path):
        """処理タイプごとのフォルダを検出できる."""
        # フォルダ構造: base_dir/date_dir/processor_type/
        (tmp_path / "20260401_0" / "original").mkdir(parents=True)
        (tmp_path / "20260401_0" / "resize").mkdir(parents=True)
        (tmp_path / "20260402_0" / "original").mkdir(parents=True)
        (tmp_path / "20260402_0" / "resize").mkdir(parents=True)

        finder = ProcessorFolderFinder(tmp_path)
        result = finder.find_processor_types()

        assert "original" in result
        assert "resize" in result
        assert len(result["original"]) == 2
        assert len(result["resize"]) == 2

    def test_find_empty_directory(self, tmp_path):
        """空のディレクトリの場合は空の辞書を返す."""
        finder = ProcessorFolderFinder(tmp_path)
        result = finder.find_processor_types()

        assert result == {}

    def test_find_no_processor_subdirs(self, tmp_path):
        """日付フォルダ内にサブディレクトリがない場合."""
        (tmp_path / "20260401_0").mkdir()
        # ファイルのみ, サブディレクトリなし
        (tmp_path / "20260401_0" / "file.txt").touch()

        finder = ProcessorFolderFinder(tmp_path)
        result = finder.find_processor_types()

        assert result == {}

    def test_find_single_processor_type(self, tmp_path):
        """1種類の処理タイプのみの場合."""
        (tmp_path / "20260401_0" / "pipeline").mkdir(parents=True)
        (tmp_path / "20260402_0" / "pipeline").mkdir(parents=True)

        finder = ProcessorFolderFinder(tmp_path)
        result = finder.find_processor_types()

        assert list(result.keys()) == ["pipeline"]
        assert len(result["pipeline"]) == 2
