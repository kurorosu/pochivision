"""File operation utilities for CSV Analytics."""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_extraction_results_path() -> Path:
    """extraction_resultsフォルダのパスを取得します."""
    # 現在のファイルから3階層上がってextraction_resultsフォルダを取得
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    return project_root / "extraction_results"


def get_root_directories() -> List[Path]:
    """ルートディレクトリ（ドライブ）の一覧を取得します."""
    if os.name == "nt":  # Windows
        drives = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive_path = Path(f"{letter}:\\")
            if drive_path.exists():
                drives.append(drive_path)
        return drives
    else:  # Unix系
        return [Path("/")]


def get_directories_in_path(directory_path: Path) -> List[Path]:
    """指定されたパス内のディレクトリを取得します（隠しフォルダを除く）."""
    if not directory_path.exists() or not directory_path.is_dir():
        return []

    try:
        directories = []
        for item in directory_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                directories.append(item)
        return sorted(directories)
    except PermissionError:
        return []


def find_csv_files_in_directory(directory_path: Path) -> List[Path]:
    """指定されたディレクトリ内のCSVファイルを検索します."""
    if not directory_path.exists() or not directory_path.is_dir():
        return []

    return list(directory_path.glob("*.csv"))


def get_subdirectories(directory_path: Path) -> List[Path]:
    """指定されたディレクトリ内のサブディレクトリを取得します."""
    if not directory_path.exists():
        return []

    return [d for d in directory_path.iterdir() if d.is_dir()]


def format_file_size(file_path: Path) -> str:
    """ファイルサイズを人間が読みやすい形式でフォーマットします."""
    try:
        file_size = file_path.stat().st_size
        if file_size < 1024:
            return f"{file_size}B"
        elif file_size < 1024 * 1024:
            return f"{file_size / 1024:.1f}KB"
        else:
            return f"{file_size / (1024 * 1024):.1f}MB"
    except OSError:
        return "不明"


def load_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """CSVファイルを読み込みます."""
    if not os.path.exists(file_path):
        console.print(f"[red]ファイルが見つかりません: {file_path}[/red]")
        return None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("ファイルを読み込み中...", total=None)
            data = pd.read_csv(file_path)
            progress.update(task, completed=True)

        console.print("[green]✓ ファイルを正常に読み込みました[/green]")
        console.print(f"  - ファイル: {file_path}")
        console.print(f"  - 行数: {len(data):,}")
        console.print(f"  - 列数: {len(data.columns):,}")

        return data

    except Exception as e:
        console.print(f"[red]ファイルの読み込みに失敗しました: {str(e)}[/red]")
        return None


def validate_file_path(file_path: str) -> bool:
    """ファイルパスの妥当性を検証します."""
    if not file_path:
        return False

    if not os.path.exists(file_path):
        return False

    if not file_path.lower().endswith(".csv"):
        return False

    return True
