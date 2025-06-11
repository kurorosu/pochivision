"""User input prompts for CSV Analytics."""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import questionary
from analytics.utils.file_utils import (
    find_csv_files_in_directory,
    format_file_size,
    get_directories_in_path,
    get_extraction_results_path,
    get_subdirectories,
)
from rich.console import Console

# プロジェクトルートをパスに追加
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
tools_path = project_root / "tools"
sys.path.insert(0, str(tools_path))


console = Console()


def extract_feature_name_from_choice(choice_text: str) -> str:
    """
    選択肢テキストから特徴量名を抽出します.

    例: "sepal length (cm) (float64, 欠損値: 0, 相関: 0.123)" -> "sepal length (cm)"
    例: "feature_name (int64, 欠損値: 5)" -> "feature_name"
    """
    if not choice_text:
        return ""

    # 最後の括弧群（データ型情報など）を特定
    parts = choice_text.split(" (")
    if len(parts) < 2:
        return choice_text

    # データ型や統計情報の括弧を探す
    # "float64", "int64", "object" などが含まれる括弧以降を除去
    data_type_found_index = -1
    for i in range(len(parts)):
        part = parts[i]
        # データ型情報を示すキーワードをチェック
        if any(
            keyword in part
            for keyword in ["float64", "int64", "object", "欠損値", "相関", "分離度"]
        ):
            data_type_found_index = i
            break

    if data_type_found_index > 0:
        # データ型情報より前の部分を特徴量名として結合
        return " (".join(parts[:data_type_found_index])
    else:
        # データ型情報が見つからない場合は最初の部分を返す
        return parts[0]


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
        "4. 散布図を表示",
        "5. 終了",
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
        "2. フォルダを選択してCSVファイルを探す",
    ]

    return questionary.select(
        "読み込み方法を選択してください:",
        choices=load_choices,
        style=get_questionary_style(),
    ).ask()


def select_folder_interactively() -> Optional[str]:
    """インタラクティブにフォルダを選択します."""
    # プロジェクトルートから開始
    project_root = Path(__file__).parent.parent.parent.parent
    current_path = project_root

    while True:
        # 現在のパスのサブディレクトリを表示
        directories = get_directories_in_path(current_path)
        console.print(f"\n[bold]フォルダ選択: {current_path}[/bold]")
        prompt_text = "フォルダを選択してください:"

        # 選択肢を作成
        choices = []

        # メインメニューに戻るオプション
        choices.append("🏠 メインメニューに戻る")

        # 親ディレクトリに戻るオプション（プロジェクトルート以外）
        if current_path != project_root:
            choices.append(".. (親フォルダに戻る)")

        # 現在のフォルダを選択するオプション（CSVファイルがある場合）
        csv_files = find_csv_files_in_directory(current_path)
        if csv_files:
            choices.append(f"✓ このフォルダを選択 ({len(csv_files)}個のCSVファイル)")

        # サブディレクトリ
        for directory in directories:
            try:
                csv_count = len(find_csv_files_in_directory(directory))
                csv_info = f" ({csv_count}個のCSV)" if csv_count > 0 else ""

                display_name = directory.name

                choices.append(f"📁 {display_name}{csv_info}")
            except PermissionError:
                display_name = directory.name if directory.name else str(directory)
                choices.append(f"📁 {display_name} (アクセス不可)")

        selected = questionary.select(
            prompt_text,
            choices=choices,
            style=get_questionary_style(),
        ).ask()

        if not selected:
            return None

        if selected == "🏠 メインメニューに戻る":
            return None
        elif selected == ".. (親フォルダに戻る)":
            # プロジェクトルート以上には戻らない
            if current_path.parent != current_path and current_path != project_root:
                current_path = current_path.parent
        elif selected.startswith("✓ このフォルダを選択"):
            return str(current_path)
        else:
            # フォルダ名を抽出
            folder_name = selected.replace("📁 ", "").split(" (")[0]
            current_path = current_path / folder_name


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

    # メインメニューに戻るオプションを追加
    csv_choices.append("🏠 メインメニューに戻る")

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

    # メインメニューに戻る場合
    if selected_csv_text == "🏠 メインメニューに戻る":
        return None

    # 選択されたCSVファイル名を抽出
    selected_csv_name = selected_csv_text.split(" (")[0]
    return str(folder_path_obj / selected_csv_name)


def select_feature_for_histogram(
    feature_choices: List[str], show_export_option: bool = False
) -> Optional[str]:
    """ヒストグラム表示用の特徴量を選択します.

    Args:
        feature_choices (List[str]): 特徴量の選択肢リスト
        show_export_option (bool): CSV出力オプションを表示するかどうか
    """
    choices = feature_choices.copy()

    # クラス分けヒストグラム時のみCSV出力オプションを追加
    if show_export_option:
        choices = ["📊 JSD距離順上位の特徴量をCSV出力"] + choices

    selected_feature_text = questionary.select(
        "ヒストグラムを表示する特徴量を選択してください（矢印キーで移動、Enterで決定）:",
        choices=choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_feature_text:
        return None

    # CSV出力が選択された場合は特別な値を返す
    if selected_feature_text == "📊 JSD距離順上位の特徴量をCSV出力":
        return "EXPORT_TOPN_FEATURES"

    # 選択された特徴量名を抽出
    return extract_feature_name_from_choice(selected_feature_text)


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


def select_ranking_count() -> Optional[int]:
    """出力する特徴量の順位数を選択します."""
    ranking_choices = [
        "🏠 メインメニューに戻る",
        "5位まで",
        "10位まで",
        "15位まで",
        "20位まで",
        "25位まで",
        "30位まで",
        "すべて",
    ]

    selected_text = questionary.select(
        "何位まで出力しますか？:",
        choices=ranking_choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_text:
        return None

    if selected_text == "🏠 メインメニューに戻る":
        return None
    elif selected_text == "すべて":
        return -1  # -1は全特徴量を意味する
    else:
        # "10位まで" -> 10
        return int(selected_text.replace("位まで", ""))


# ========== 散布図関連のプロンプト機能 ==========


def select_y_axis_feature(numeric_columns: List[str]) -> Optional[str]:
    """Y軸の特徴量を選択します."""
    if not numeric_columns:
        return None

    console.print(
        f"\n[bold]Y軸特徴量選択（{len(numeric_columns)}個の数値列から選択）[/bold]"
    )

    y_axis_choices = []
    for col in numeric_columns:
        choice_text = f"{col}"
        y_axis_choices.append(choice_text)

    selected_y_axis = questionary.select(
        "Y軸に使用する特徴量を選択してください（矢印キーで移動、Enterで決定）:",
        choices=y_axis_choices,
        style=get_questionary_style(),
    ).ask()

    return selected_y_axis


def select_x_axis_feature(
    feature_choices: List[str], show_export_option: bool = False
) -> Optional[str]:
    """散布図のX軸特徴量を選択します.

    Args:
        feature_choices (List[str]): 特徴量の選択肢リスト（相関係数順）
        show_export_option (bool): CSV出力オプションを表示するかどうか
    """
    choices = feature_choices.copy()

    # クラス分け散布図時のみCSV出力オプションを追加
    if show_export_option:
        choices = ["📊 相関係数順上位の特徴量をCSV出力"] + choices

    selected_feature_text = questionary.select(
        "X軸の特徴量を選択してください（矢印キーで移動、Enterで決定）:",
        choices=choices,
        style=get_questionary_style(),
    ).ask()

    if not selected_feature_text:
        return None

    # CSV出力が選択された場合は特別な値を返す
    if selected_feature_text == "📊 相関係数順上位の特徴量をCSV出力":
        return "EXPORT_CORRELATION_FEATURES"

    # 選択された特徴量名を抽出
    return extract_feature_name_from_choice(selected_feature_text)


def select_post_scatter_plot_action() -> Optional[str]:
    """散布図表示後のアクションを選択します."""
    action_choices = [
        "1. 別のX軸特徴量で散布図を表示",
        "2. Y軸特徴量を変更",
        "3. メインメニューに戻る",
    ]

    return questionary.select(
        "次のアクションを選択してください:",
        choices=action_choices,
        style=get_questionary_style(),
    ).ask()


def select_scatter_display_mode() -> Optional[str]:
    """散布図の表示モードを選択します."""
    display_choices = [
        "1. 単純な散布図",
        "2. クラス別色分け散布図",
    ]

    return questionary.select(
        "散布図の表示モードを選択してください:",
        choices=display_choices,
        style=get_questionary_style(),
    ).ask()
