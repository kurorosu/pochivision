"""CLI サブコマンドの振る舞いテスト."""

from click.testing import CliRunner

from pochivision.cli.main import main


class TestCLISubcommands:
    """pochi コマンドのサブコマンド構成テスト."""

    def setup_method(self):
        """テストメソッドごとに CliRunner を初期化."""
        self.runner = CliRunner()

    def test_no_subcommand_shows_help(self):
        """サブコマンドなしでヘルプが表示される."""
        result = self.runner.invoke(main)
        assert result.exit_code == 0
        assert "Commands:" in result.output

    def test_help_shows_all_subcommands(self):
        """--help で全サブコマンドが表示される."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "extract" in result.output
        assert "process" in result.output
        assert "aggregate" in result.output
        assert "fft" in result.output

    def test_run_help(self):
        """pochi run --help が動作する."""
        result = self.runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--camera" in result.output
        assert "--profile" in result.output

    def test_extract_help(self):
        """pochi extract --help が動作する."""
        result = self.runner.invoke(main, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_process_help(self):
        """pochi process --help が動作する."""
        result = self.runner.invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--input" in result.output

    def test_aggregate_help(self):
        """pochi aggregate --help が動作する."""
        result = self.runner.invoke(main, ["aggregate", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--mode" in result.output

    def test_fft_help(self):
        """pochi fft --help が動作する."""
        result = self.runner.invoke(main, ["fft", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output

    def test_unknown_subcommand(self):
        """存在しないサブコマンドでエラー."""
        result = self.runner.invoke(main, ["nonexistent"])
        assert result.exit_code != 0
