"""User input prompts for CSV Analytics."""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import questionary
from analytics.utils.file_utils import (
    find_csv_files_in_directory,
    format_file_size,
    get_extraction_results_path,
    get_subdirectories,
)
from rich.console import Console
from rich.prompt import Prompt

# プロジェクトルートをパスに追加
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
tools_path = project_root / "tools"
sys.path.insert(0, str(tools_path))


console = Console()


def get_questionary_style() -> questionary.Style:
    """questionary用の共通スタイルを取得します."""
    return questionary.Style(
        [
            ("question", "bold"),
            ("answer", "fg:#ff9d00 bold"),
            ("pointer", "fg:#ff9d00 bold"),
            ("highlighted", "fg:#ff9d00 bold"),
            ("selected", "fg:#cc5454"),
            ("separator", "fg:#cc5454"),
            ("instruction", ""),
            ("text", ""),
            ("disabled", "fg:#858585 italic"),
        ]
    )


def select_main_menu_option() -> Optional[str]:
    """メインメニューの選択を取得します."""
    menu_choices = [
        "1. CSVファイルを読み込む",
        "2. データの概要を表示",
        "3. ヒストグラムを表示",
        "4. 終了",
    ]

    return questionary.select(
        "選択してください（矢印キーで移動、Enterで決定）:",
        choices=menu_choices,
        style=get_questionary_style(),
    ).ask()


def select_load_method() -> Optional[str]:
    """ファイル読み込み方法の選択を取得します."""
    load_choices = [
        "1. 既定フォルダ（extraction_results）から選択",
        "2. 任意のパスを入力",
    ]

    return questionary.select(
        "読み込み方法を選択してください:",
        choices=load_choices,
        style=get_questionary_style(),
    ).ask()


def select_folder_from_extraction_results() -> Optional[str]:
    """extraction_resultsフォルダからサブフォルダを選択します."""
    extraction_results_path = get_extraction_results_path()

    if not extraction_results_path.exists():
        console.print(
            f"[red]extraction_resultsフォルダが見つかりません: {extraction_results_path}[/red]"
        )
        return None

    subdirs = get_subdirectories(extraction_results_path)

    if not subdirs:
        console.print(
            "[red]extraction_resultsフォルダにサブフォルダが見つかりません。[/red]"
        )
        return None

    console.print(f"\n[bold]フォルダ選択（{len(subdirs)}個のフォルダ）[/bold]")

    # フォルダ選択肢を作成
    folder_choices = []
    for subdir in sorted(subdirs):
        csv_files = find_csv_files_in_directory(subdir)
        choice_text = f"{subdir.name} ({len(csv_files)}個のCSVファイル)"
        folder_choices.append(choice_text)

    selected_folder_text = questionary.select(
        "フォルダを選択してください（矢印キーで移動、Enterで決定）:",
        choices=folder_choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_folder_text:
        return None

    # 選択されたフォルダ名を抽出
    selected_folder_name = selected_folder_text.split(" (")[0]
    return str(extraction_results_path / selected_folder_name)


def select_csv_file_from_folder(folder_path: str) -> Optional[str]:
    """指定されたフォルダからCSVファイルを選択します."""
    folder_path_obj = Path(folder_path)
    csv_files = find_csv_files_in_directory(folder_path_obj)

    if not csv_files:
        console.print(
            f"[red]選択されたフォルダにCSVファイルが見つかりません: {folder_path}[/red]"
        )
        return None

    if len(csv_files) == 1:
        # CSVファイルが1つの場合は自動選択
        selected_csv = csv_files[0]
        console.print(f"[green]✓ CSVファイルを自動選択: {selected_csv.name}[/green]")
        return str(selected_csv)

    # 複数のCSVファイルがある場合は選択
    console.print(f"\n[bold]CSVファイル選択（{len(csv_files)}個のファイル）[/bold]")

    csv_choices = []
    for csv_file in sorted(csv_files):
        size_str = format_file_size(csv_file)
        choice_text = f"{csv_file.name} ({size_str})"
        csv_choices.append(choice_text)

    selected_csv_text = questionary.select(
        "CSVファイルを選択してください（矢印キーで移動、Enterで決定）:",
        choices=csv_choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_csv_text:
        return None

    # 選択されたCSVファイル名を抽出
    selected_csv_name = selected_csv_text.split(" (")[0]
    return str(folder_path_obj / selected_csv_name)


def input_csv_path() -> Optional[str]:
    """CSVファイルのパスを手動入力します."""
    file_path = Prompt.ask("CSVファイルのパスを入力してください")
    return file_path if file_path else None


def select_feature_for_histogram(feature_choices: List[str]) -> Optional[str]:
    """ヒストグラム表示用の特徴量を選択します."""
    selected_feature_text = questionary.select(
        "ヒストグラムを表示する特徴量を選択してください（矢印キーで移動、Enterで決定）:",
        choices=feature_choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_feature_text:
        return None

    # 選択された特徴量名を抽出
    return selected_feature_text.split(" (")[0]


def confirm_data_transformation(message: str, default: bool = True) -> bool:
    """データ変換の確認を取得します."""
    return questionary.confirm(message, default=default).ask()


def select_display_mode() -> Optional[str]:
    """表示モードを選択します."""
    display_choices = [
        "1. 単純なヒストグラム",
        "2. クラス別色分けヒストグラム",
    ]

    return questionary.select(
        "表示モードを選択してください:",
        choices=display_choices,
        style=get_questionary_style(),
    ).ask()


def select_class_column(
    data: pd.DataFrame, categorical_columns: List[str]
) -> Optional[str]:
    """クラス列を選択します."""
    class_choices = []
    for col in categorical_columns:
        unique_count = data[col].nunique()
        dtype_str = "文字列型" if data[col].dtype == "object" else str(data[col].dtype)
        choice_text = f"{col}({unique_count}種類.{dtype_str})"
        class_choices.append(choice_text)

    selected_class_text = questionary.select(
        "クラス別表示に使用する列を選択してください:",
        choices=class_choices,
        style=get_questionary_style(),
    ).ask()

    # 列名のみを抽出して返す
    return selected_class_text.split("(")[0] if selected_class_text else None


def select_post_histogram_action() -> Optional[str]:
    """ヒストグラム表示後のアクションを選択します."""
    action_choices = [
        "1. 別の特徴量のヒストグラムを表示",
        "2. 表示設定を変更",
        "3. メインメニューに戻る",
    ]

    return questionary.select(
        "次のアクションを選択してください:",
        choices=action_choices,
        style=get_questionary_style(),
    ).ask()


def select_index_column(id_columns: List[str]) -> Optional[str]:
    """インデックス列を選択します."""
    index_choices = []
    for col in id_columns:
        choice_text = f"{col}"
        index_choices.append(choice_text)

    selected_index_text = questionary.select(
        "メインインデックス列を選択してください:",
        choices=index_choices,
        style=get_questionary_style(),
    ).ask()

    return selected_index_text.split(" (")[0] if selected_index_text else None
