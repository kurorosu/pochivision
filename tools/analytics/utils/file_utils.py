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


def export_topn_features_to_csv(
    feature_choices: List[str], csv_file_path: str, ranking_count: int
) -> Optional[str]:
    """JSD距離順上位N位の特徴量をCSVファイルに出力します.

    Args:
        feature_choices (List[str]): 特徴量の選択肢リスト（JSD距離順）
        csv_file_path (str): 元のCSVファイルのパス
        ranking_count (int): 出力する順位数（-1の場合は全特徴量）

    Returns:
        Optional[str]: 出力されたCSVファイルのパス、失敗時はNone
    """
    try:
        # 特徴量名のみを抽出
        feature_names = []
        if ranking_count == -1:
            # 全特徴量を出力
            target_choices = feature_choices
            rank_text = "all"
        else:
            # 指定された順位まで出力
            target_choices = feature_choices[:ranking_count]
            rank_text = f"top{ranking_count}"

        for choice in target_choices:
            feature_name = choice.split(" (")[0]
            feature_names.append(feature_name)

        # 出力先のパスを決定（元のCSVファイルと同じフォルダ）
        csv_path = Path(csv_file_path)
        output_dir = csv_path.parent
        output_filename = f"{csv_path.stem}_{rank_text}_features.csv"
        output_path = output_dir / output_filename

        # CSVファイルに出力
        df_features = pd.DataFrame({"feature_name": feature_names})
        df_features.to_csv(output_path, index=False, encoding="utf-8")

        if ranking_count == -1:
            console.print(
                "[green]✓ JSD距離順全特徴量をCSVファイルに出力しました[/green]"
            )
        else:
            console.print(
                f"[green]✓ JSD距離順上位{ranking_count}位の特徴量をCSVファイルに出力しました[/green]"
            )
        console.print(f"  - 出力先: {output_path}")
        console.print(f"  - 特徴量数: {len(feature_names)}")

        return str(output_path)

    except Exception as e:
        console.print(f"[red]CSV出力に失敗しました: {str(e)}[/red]")
        return None


def export_correlation_features_to_csv(
    feature_choices: List[str],
    csv_file_path: str,
    ranking_count: int,
    y_axis_feature: str,
) -> Optional[str]:
    """相関係数順上位N位の特徴量をCSVファイルに出力します.

    Args:
        feature_choices (List[str]): 特徴量の選択肢リスト（相関係数順）
        csv_file_path (str): 元のCSVファイルのパス
        ranking_count (int): 出力する順位数（-1の場合は全特徴量）
        y_axis_feature (str): Y軸特徴量名

    Returns:
        Optional[str]: 出力されたCSVファイルのパス、失敗時はNone
    """
    try:
        # 特徴量名のみを抽出
        feature_names = []
        if ranking_count == -1:
            # 全特徴量を出力
            target_choices = feature_choices
            rank_text = "all"
        else:
            # 指定された順位まで出力
            target_choices = feature_choices[:ranking_count]
            rank_text = f"top{ranking_count}"

        for choice in target_choices:
            feature_name = choice.split(" (")[0]
            feature_names.append(feature_name)

        # 出力先のパスを決定（元のCSVファイルと同じフォルダ）
        csv_path = Path(csv_file_path)
        output_dir = csv_path.parent
        output_filename = (
            f"{csv_path.stem}_{rank_text}_correlation_with_{y_axis_feature}.csv"
        )
        output_path = output_dir / output_filename

        # CSVファイルに出力
        df_features = pd.DataFrame(
            {
                "feature_name": feature_names,
                "y_axis_reference": [y_axis_feature] * len(feature_names),
            }
        )
        df_features.to_csv(output_path, index=False, encoding="utf-8")

        if ranking_count == -1:
            console.print(
                f"[green]✓ {y_axis_feature}との相関係数順全特徴量をCSVファイルに出力しました[/green]"
            )
        else:
            console.print(
                f"[green]✓ {y_axis_feature}との相関係数順上位{ranking_count}位の"
                f"特徴量をCSVファイルに出力しました[/green]"
            )
        console.print(f"  - 出力先: {output_path}")
        console.print(f"  - 特徴量数: {len(feature_names)}")

        return str(output_path)

    except Exception as e:
        console.print(f"[red]CSV出力に失敗しました: {str(e)}[/red]")
        return None
