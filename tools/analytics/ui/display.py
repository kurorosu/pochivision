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


def create_feature_choices(
    data: pd.DataFrame, class_column: Optional[str] = None
) -> List[str]:
    """特徴量選択用の選択肢を作成します."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_choices = []

    # クラス別表示モードの場合、Jensen-Shannon Divergence順でソート
    if class_column is not None:
        from analytics.core.statistical_analyzer import (
            rank_features_by_class_separation,
        )

        ranked_features = rank_features_by_class_separation(
            data, numeric_columns, class_column
        )

        for feature, js_score in ranked_features:
            null_count = data[feature].isnull().sum()
            dtype_str = str(data[feature].dtype)
            choice_text = (
                f"{feature} ({dtype_str}, 欠損値: {null_count}, 分離度: {js_score:.3f})"
            )
            feature_choices.append(choice_text)
    else:
        # 単純表示モードの場合、元の順序
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


# ========== 散布図関連の表示機能 ==========


def show_scatter_plot_header(
    display_mode: str, selected_class_column: Optional[str], y_axis_feature: str
) -> None:
    """散布図表示のヘッダーを表示します."""
    console.print("\n[bold]散布図表示[/bold]")
    console.print("\n" + "=" * 40)
    console.print(f"[bold cyan]Y軸: {y_axis_feature}[/bold cyan]")
    if display_mode == "simple":
        console.print("[bold cyan]表示設定: 単純な散布図[/bold cyan]")
    elif display_mode == "class":
        console.print(
            f"[bold cyan]表示設定: クラス別色分け（{selected_class_column}）[/bold cyan]"
        )
    console.print("=" * 40)


def create_scatter_feature_choices(
    data: pd.DataFrame, y_axis_feature: str, class_column: Optional[str] = None
) -> List[str]:
    """散布図用のX軸特徴量選択肢を相関係数順で作成します."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Y軸特徴量を除外
    if y_axis_feature in numeric_columns:
        numeric_columns.remove(y_axis_feature)

    if not numeric_columns:
        return []

    # Y軸特徴量との相関係数を計算
    y_data = data[y_axis_feature].dropna()
    correlations = []

    for col in numeric_columns:
        col_data = data[col].dropna()
        # 両方のデータが存在する行のみを使用
        common_indices = y_data.index.intersection(col_data.index)
        if len(common_indices) > 1:  # 最低2点は必要
            y_common = y_data.loc[common_indices]
            col_common = col_data.loc[common_indices]
            correlation = y_common.corr(col_common)
            if not pd.isna(correlation):
                correlations.append((col, abs(correlation)))

    # 相関係数の絶対値で降順ソート
    correlations.sort(key=lambda x: x[1], reverse=True)

    feature_choices = []
    for feature, correlation in correlations:
        null_count = data[feature].isnull().sum()
        dtype_str = str(data[feature].dtype)
        choice_text = (
            f"{feature} ({dtype_str}, 欠損値: {null_count}, 相関: {correlation:.3f})"
        )
        feature_choices.append(choice_text)

    return feature_choices


def show_simple_scatter_plot(data: pd.DataFrame, x_column: str, y_column: str) -> None:
    """単純な散布図を表示します."""
    # 両方の列にデータがある行のみを使用
    clean_data = data[[x_column, y_column]].dropna()

    if len(clean_data) == 0:
        console.print(f"[red]{x_column} と {y_column} に共通データがありません。[/red]")
        return

    console.print(f"\n[bold]{x_column} vs {y_column} の散布図[/bold]")

    # plotextで散布図を表示
    plt.clear_data()
    plt.scatter(clean_data[x_column].values, clean_data[y_column].values)
    plt.title(f"{x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

    # 相関情報を表示
    correlation = clean_data[x_column].corr(clean_data[y_column])
    console.print("\n[bold]相関情報[/bold]")
    corr_table = Table(show_header=True, header_style="bold magenta")
    corr_table.add_column("統計量", style="cyan", width=20)
    corr_table.add_column("値", style="white", width=20)

    corr_table.add_row("データ点数", f"{len(clean_data):,}")
    corr_table.add_row("相関係数", f"{correlation:.6f}")

    if abs(correlation) >= 0.7:
        strength = "強い"
    elif abs(correlation) >= 0.3:
        strength = "中程度"
    else:
        strength = "弱い"

    direction = "正の" if correlation > 0 else "負の"
    corr_table.add_row("相関の強さ", f"{direction}{strength}相関")

    console.print(corr_table)


def show_class_scatter_plot(
    data: pd.DataFrame, x_column: str, y_column: str, class_column: str
) -> None:
    """クラス別に色分けした散布図を表示します."""
    console.print(
        f"\n[bold]{x_column} vs {y_column} のクラス別散布図（{class_column}で色分け）[/bold]"
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

    # plotextでクラス別散布図を表示
    plt.clear_data()

    total_points = 0
    for class_name in classes:
        class_mask = data[class_column] == class_name
        class_data = data[class_mask][[x_column, y_column]].dropna()

        if len(class_data) > 0:
            plt.scatter(
                class_data[x_column].values,
                class_data[y_column].values,
                label=str(class_name),
            )
            total_points += len(class_data)

    plt.title(f"{x_column} vs {y_column} のクラス別散布図")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

    # クラス別統計を表示
    console.print("\n[bold]クラス別統計[/bold]")
    class_stats_table = Table(show_header=True, header_style="bold magenta")
    class_stats_table.add_column("クラス", style="cyan", width=15)
    class_stats_table.add_column("データ点数", style="white", width=12)
    class_stats_table.add_column("X軸平均", style="white", width=12)
    class_stats_table.add_column("Y軸平均", style="white", width=12)
    class_stats_table.add_column("相関係数", style="white", width=12)

    for class_name in classes:
        class_mask = data[class_column] == class_name
        class_data = data[class_mask][[x_column, y_column]].dropna()

        if len(class_data) > 1:  # 相関計算には最低2点必要
            correlation = class_data[x_column].corr(class_data[y_column])
            x_mean = class_data[x_column].mean()
            y_mean = class_data[y_column].mean()

            class_stats_table.add_row(
                str(class_name),
                f"{len(class_data):,}",
                f"{x_mean:.4f}",
                f"{y_mean:.4f}",
                f"{correlation:.4f}" if not pd.isna(correlation) else "N/A",
            )

    console.print(class_stats_table)
    console.print(f"\n[dim]全体データ点数: {total_points:,}[/dim]")


def show_post_scatter_plot_menu() -> None:
    """散布図表示後のメニューヘッダーを表示します."""
    console.print("\n" + "=" * 40)
    console.print("[bold cyan]次のアクション[/bold cyan]")
    console.print("=" * 40)
