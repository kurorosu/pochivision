"""extract_class_from_filename のテスト."""

from pochivision.utils.class_extraction import extract_class_from_filename


class TestExtractClassFromFilename:
    """extract_class_from_filename のテストクラス."""

    def test_default_delimiter_position_zero(self):
        """デフォルト設定 (区切り文字 '_', 位置 0) でクラス名を抽出する."""
        result = extract_class_from_filename("apple_001")
        assert result == "apple"

    def test_custom_delimiter(self):
        """カスタム区切り文字でクラス名を抽出する."""
        result = extract_class_from_filename("apple-001", delimiter="-")
        assert result == "apple"

    def test_position_one(self):
        """位置 1 でクラス名を抽出する."""
        result = extract_class_from_filename("prefix_classA_001", position=1)
        assert result == "classA"

    def test_negative_position(self):
        """負のインデックスでクラス名を抽出する."""
        result = extract_class_from_filename("apple_001_suffix", position=-1)
        assert result == "suffix"

    def test_out_of_bounds_returns_empty(self):
        """範囲外インデックスで空文字列を返す."""
        result = extract_class_from_filename("apple_001", position=5)
        assert result == ""

    def test_no_delimiter_in_filename(self):
        """区切り文字がファイル名にない場合, 位置 0 で全体を返す."""
        result = extract_class_from_filename("apple", delimiter="_", position=0)
        assert result == "apple"

    def test_no_delimiter_position_nonzero_returns_empty(self):
        """区切り文字がなく位置 1 で空文字列を返す."""
        result = extract_class_from_filename("apple", delimiter="_", position=1)
        assert result == ""
