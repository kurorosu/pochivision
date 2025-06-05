"""Data overview commands for CSV Analytics."""

from analytics.core.data_processor import DataProcessor
from analytics.ui.display import show_data_overview
from rich.console import Console

console = Console()


def show_data_overview_command(data_processor: DataProcessor) -> bool:
    """データの概要を表示するコマンドを実行します."""
    if data_processor.data is None:
        console.print("[red]まずCSVファイルを読み込んでください。[/red]")
        return False

    if data_processor.file_path is None:
        console.print("[red]ファイルパスが設定されていません。[/red]")
        return False

    # データ概要を表示
    show_data_overview(data_processor.data, data_processor.file_path)
    return True
