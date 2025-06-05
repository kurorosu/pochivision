"""Data display components for CSV Analytics."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotext as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def show_welcome_message() -> None:
    """ウェルカムメッセージを表示します."""
    welcome_text = """
[bold blue]データ概要表示CLI アプリケーション[/bold blue]

CSVファイルの基本情報を素早く確認できるシンプルなツールです。
ヒストグラム表示機能も搭載しています。
    """
    console.print(Panel(welcome_text, title="ようこそ", border_style="blue"))


def show_main_menu_header(
    data: Optional[pd.DataFrame],
    file_path: Optional[str],
    display_mode: Optional[str],
    selected_class_column: Optional[str],
) -> None:
    """メインメニューのヘッダー情報を表示します."""
    console.print("\n" + "=" * 40)
    console.print("[bold cyan]メインメニュー[/bold cyan]")
    console.print("=" * 40)

    # 現在の設定状況を表示
    if data is not None and file_path is not None:
        console.print(f"[dim]読み込み済みファイル: {os.path.basename(file_path)}[/dim]")
        if display_mode == "simple":
            console.print("[dim]表示設定: 単純なヒストグラム[/dim]")
        elif display_mode == "class":
            console.print(
                f"[dim]表示設定: クラス別色分け（{selected_class_column}）[/dim]"
            )
        else:
            console.print("[dim]表示設定: 未設定[/dim]")


def show_data_overview(data: pd.DataFrame, file_path: str) -> None:
    """データの概要を表示します."""
    console.print(f"\n[bold]データ概要: {os.path.basename(file_path)}[/bold]")

    # 基本情報テーブル
    basic_info_table = Table(show_header=True, header_style="bold magenta")
    basic_info_table.add_column("項目", style="cyan", width=20)
    basic_info_table.add_column("値", style="white", width=30)

    basic_info_table.add_row("ファイルパス", file_path)
    basic_info_table.add_row("行数", f"{len(data):,}")
    basic_info_table.add_row("列数", f"{len(data.columns):,}")
    basic_info_table.add_row(
        "メモリ使用量", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    )

    console.print(basic_info_table)

    # データ型情報
    console.print("\n[bold]データ型情報[/bold]")
    dtype_table = Table(show_header=True, header_style="bold magenta")
    dtype_table.add_column("データ型", style="cyan", width=15)
    dtype_table.add_column("列数", style="white", width=10)
    dtype_table.add_column("割合", style="white", width=10)

    dtype_counts = data.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        percentage = count / len(data.columns) * 100
        dtype_table.add_row(str(dtype), str(count), f"{percentage:.1f}%")

    console.print(dtype_table)

    # 欠損値情報
    console.print("\n[bold]欠損値情報[/bold]")
    null_counts = data.isnull().sum()
    null_info = null_counts[null_counts > 0]

    if len(null_info) > 0:
        null_table = Table(show_header=True, header_style="bold magenta")
        null_table.add_column("列名", style="cyan", width=25)
        null_table.add_column("欠損値数", style="white", width=12)
        null_table.add_column("欠損率", style="white", width=10)

        for col, null_count in null_info.items():
            null_rate = null_count / len(data) * 100
            null_table.add_row(col, f"{null_count:,}", f"{null_rate:.1f}%")

        console.print(null_table)
    else:
        console.print("[green]✓ 欠損値はありません[/green]")

    # 数値列の統計情報
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        console.print(f"\n[bold]数値列の統計情報（{len(numeric_columns)}列）[/bold]")
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("列名", style="cyan", width=20)
        stats_table.add_column("平均", style="white", width=12)
        stats_table.add_column("標準偏差", style="white", width=12)
        stats_table.add_column("最小値", style="white", width=12)
        stats_table.add_column("最大値", style="white", width=12)

        for col in numeric_columns[:10]:  # 最初の10列のみ表示
            col_data = data[col].dropna()
            if len(col_data) > 0:
                stats_table.add_row(
                    col,
                    f"{col_data.mean():.4f}",
                    f"{col_data.std():.4f}",
                    f"{col_data.min():.4f}",
                    f"{col_data.max():.4f}",
                )

        console.print(stats_table)

        if len(numeric_columns) > 10:
            console.print(f"[dim]... 他 {len(numeric_columns) - 10} 列[/dim]")

    # データサンプル表示
    console.print("\n[bold]データサンプル（最初の5行）[/bold]")
    sample_table = Table(show_header=True, header_style="bold magenta")

    # 列数が多い場合は最初の5列のみ表示
    display_columns = data.columns[:5] if len(data.columns) > 5 else data.columns

    for col in display_columns:
        sample_table.add_column(col, style="white", width=15, overflow="ellipsis")

    for idx in range(min(5, len(data))):
        row_data = []
        for col in display_columns:
            value = str(data.iloc[idx][col])
            if len(value) > 12:
                value = value[:12] + "..."
            row_data.append(value)
        sample_table.add_row(*row_data)

    console.print(sample_table)

    if len(data.columns) > 5:
        console.print(f"[dim]... 他 {len(data.columns) - 5} 列[/dim]")


def show_histogram_header(
    display_mode: str, selected_class_column: Optional[str]
) -> None:
    """ヒストグラム表示のヘッダーを表示します."""
    console.print("\n[bold]ヒストグラム表示[/bold]")
    console.print("\n" + "=" * 40)
    if display_mode == "simple":
        console.print("[bold cyan]表示設定: 単純なヒストグラム[/bold cyan]")
    elif display_mode == "class":
        console.print(
            f"[bold cyan]表示設定: クラス別色分け（{selected_class_column}）[/bold cyan]"
        )
    console.print("=" * 40)


def create_feature_choices(data: pd.DataFrame) -> List[str]:
    """特徴量選択用の選択肢を作成します."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_choices = []

    for col in numeric_columns:
        null_count = data[col].isnull().sum()
        dtype_str = str(data[col].dtype)
        choice_text = f"{col} ({dtype_str}, 欠損値: {null_count})"
        feature_choices.append(choice_text)

    return feature_choices


def show_simple_histogram(data: pd.DataFrame, column_name: str) -> None:
    """単純なヒストグラムを表示します."""
    column_data = data[column_name].dropna()

    if len(column_data) == 0:
        console.print(f"[red]{column_name} にデータがありません。[/red]")
        return

    console.print(f"\n[bold]{column_name} のヒストグラム[/bold]")

    # plotextでヒストグラムを表示
    plt.clear_data()
    plt.hist(column_data.values, bins=20)
    plt.title(f"{column_name} の分布")
    plt.xlabel(column_name)
    plt.ylabel("頻度")
    plt.show()

    # 統計情報を表示
    console.print("\n[bold]統計情報[/bold]")
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("統計量", style="cyan", width=15)
    stats_table.add_column("値", style="white", width=20)

    stats_table.add_row("データ数", f"{len(column_data):,}")
    stats_table.add_row("平均", f"{column_data.mean():.6f}")
    stats_table.add_row("標準偏差", f"{column_data.std():.6f}")
    stats_table.add_row("最小値", f"{column_data.min():.6f}")
    stats_table.add_row("25%点", f"{column_data.quantile(0.25):.6f}")
    stats_table.add_row("中央値", f"{column_data.median():.6f}")
    stats_table.add_row("75%点", f"{column_data.quantile(0.75):.6f}")
    stats_table.add_row("最大値", f"{column_data.max():.6f}")

    console.print(stats_table)


def show_class_histogram(
    data: pd.DataFrame, column_name: str, class_column: str
) -> None:
    """クラス別に色分けしたヒストグラムを表示します."""
    console.print(
        f"\n[bold]{column_name} のクラス別ヒストグラム（{class_column}で色分け）[/bold]"
    )

    # クラスごとのデータを取得
    classes = data[class_column].unique()
    classes = [c for c in classes if pd.notna(c)]  # NaNを除外

    if len(classes) == 0:
        console.print("[red]有効なクラスデータがありません。[/red]")
        return

    if len(classes) > 10:
        console.print(
            f"[yellow]警告: クラス数が多すぎます（{len(classes)}個）。最初の10個のみ表示します。[/yellow]"
        )
        classes = classes[:10]

    # plotextでクラス別ヒストグラムを表示
    plt.clear_data()

    for class_name in classes:
        class_data = data[data[class_column] == class_name][column_name].dropna()
        if len(class_data) > 0:
            plt.hist(class_data.values, bins=15, label=str(class_name))

    plt.title(f"{column_name} のクラス別ヒストグラム")
    plt.xlabel(column_name)
    plt.ylabel("頻度")
    plt.show()

    # クラス別統計量を表示
    console.print("\n[bold]クラス別統計量[/bold]")
    class_stats_table = Table(show_header=True, header_style="bold magenta")
    class_stats_table.add_column("クラス", style="cyan", width=15)
    class_stats_table.add_column("データ数", style="white", width=10)
    class_stats_table.add_column("平均", style="white", width=12)
    class_stats_table.add_column("標準偏差", style="white", width=12)
    class_stats_table.add_column("最小値", style="white", width=12)
    class_stats_table.add_column("最大値", style="white", width=12)

    for class_name in classes:
        class_data = data[data[class_column] == class_name][column_name].dropna()
        if len(class_data) > 0:
            class_stats_table.add_row(
                str(class_name),
                f"{len(class_data):,}",
                f"{class_data.mean():.4f}",
                f"{class_data.std():.4f}",
                f"{class_data.min():.4f}",
                f"{class_data.max():.4f}",
            )

    console.print(class_stats_table)


def show_post_histogram_menu() -> None:
    """ヒストグラム表示後のメニューヘッダーを表示します."""
    console.print("\n" + "=" * 40)
    console.print("[bold cyan]次のアクション[/bold cyan]")
    console.print("=" * 40)


def show_data_transformation_suggestion(reasons: List[str]) -> None:
    """データ変換の提案を表示します."""
    console.print("\n" + "=" * 50)
    console.print("[bold yellow]縦持ちデータの可能性があります[/bold yellow]")
    console.print("=" * 50)

    console.print("[dim]検出された特徴：[/dim]")
    for reason in reasons:
        console.print(f"  • {reason}")

    console.print("\n[dim]縦持ちデータの場合、行と列を入れ替える（転置）ことで[/dim]")
    console.print("[dim]ヒストグラム表示などの機能が正常に動作します。[/dim]")


def show_long_format_detection() -> None:
    """Long形式データ検出の表示を行います."""
    console.print("\n" + "=" * 50)
    console.print(
        "[bold yellow]Long format（縦持ち）データが検出されました[/bold yellow]"
    )
    console.print("=" * 50)


def show_pivot_structure(
    feature_name_col: str,
    feature_value_col: str,
    id_columns: List[str],
    unique_features: int,
) -> None:
    """ピボット変換の構造を表示します."""
    console.print("[dim]検出された構造：[/dim]")
    console.print(f"  • 特徴量名列: {feature_name_col}")
    console.print(f"  • 特徴量値列: {feature_value_col}")
    console.print(f"  • 識別列: {', '.join(id_columns)}")
    console.print(f"  • ユニークな特徴量数: {unique_features:,}")

    console.print(
        "\n[dim]Long formatデータを横持ち（Wide format）に変換することで[/dim]"
    )
    console.print("[dim]ヒストグラム表示などの機能が正常に動作します。[/dim]")


def show_transformation_success(data: pd.DataFrame, transformation_type: str) -> None:
    """データ変換成功メッセージを表示します."""
    console.print(f"[green]✓ {transformation_type}が完了しました[/green]")
    console.print(f"  - 新しい行数: {len(data):,}")
    console.print(f"  - 新しい列数: {len(data.columns):,}")

    # 変換後のデータ概要を表示
    console.print(f"\n[bold]{transformation_type}後のデータ概要[/bold]")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    console.print(f"  - 数値列: {len(numeric_columns):,}個")
    console.print(f"  - 数値列の割合: {len(numeric_columns) / len(data.columns):.1%}")
