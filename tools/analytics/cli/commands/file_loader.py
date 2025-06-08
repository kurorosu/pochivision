"""File loading commands for CSV Analytics."""

from typing import Optional

from analytics.core.data_processor import DataProcessor
from analytics.ui.prompts import (
    select_csv_file_from_folder,
    select_folder_from_extraction_results,
    select_folder_interactively,
    select_load_method,
)
from analytics.utils.file_utils import load_csv_file, validate_file_path
from rich.console import Console

console = Console()


def load_csv_file_command(data_processor: DataProcessor) -> bool:
    """CSVファイルを読み込むコマンドを実行します."""
    console.print("\n[bold]CSVファイルの読み込み[/bold]")

    # 読み込み方法を選択
    load_choice = select_load_method()

    if not load_choice:
        return False

    file_path = None

    if load_choice.startswith("1."):
        # 既定フォルダから選択
        file_path = _select_csv_from_extraction_results()
    else:
        # フォルダを選択してCSVファイルを探す
        file_path = _select_csv_from_interactive_folder()

    if not file_path:
        return False

    # ファイルを読み込み
    return _load_csv_from_path(data_processor, file_path)


def _select_csv_from_extraction_results() -> Optional[str]:
    """extraction_resultsフォルダからCSVファイルを選択します."""
    # フォルダを選択
    folder_path = select_folder_from_extraction_results()
    if not folder_path:
        return None

    # CSVファイルを選択
    return select_csv_file_from_folder(folder_path)


def _select_csv_from_interactive_folder() -> Optional[str]:
    """インタラクティブにフォルダを選択してCSVファイルを選択します."""
    # フォルダを選択
    folder_path = select_folder_interactively()
    if not folder_path:
        return None

    # CSVファイルを選択
    return select_csv_file_from_folder(folder_path)


def _load_csv_from_path(data_processor: DataProcessor, file_path: str) -> bool:
    """指定されたパスからCSVファイルを読み込みます."""
    # ファイルパスの妥当性を検証
    if not validate_file_path(file_path):
        console.print(f"[red]無効なファイルパスです: {file_path}[/red]")
        return False

    # CSVファイルを読み込み
    data = load_csv_file(file_path)
    if data is None:
        return False

    # データプロセッサにデータを設定
    data_processor.load_data(data, file_path)

    # CSV読み込み直後にLong formatデータを検出
    data_processor.check_and_suggest_data_transformation()

    return True
