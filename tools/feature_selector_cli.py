#!/usr/bin/env python3
"""
データ概要表示CLI アプリケーション.

このモジュールは、CSVファイルを読み込んで基本的な概要を表示する
シンプルなコマンドラインインターフェースを提供します。
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import plotext as plt
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

console = Console()


class DataOverviewCLI:
    """
    データ概要表示のためのCLIアプリケーション.

    CSVファイルを読み込んで、基本的なデータ概要を表示します。
    """

    def __init__(self):
        """初期化処理."""
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.display_mode: Optional[str] = None  # 'simple' or 'class'
        self.selected_class_column: Optional[str] = None  # クラス別表示時のクラス列
        self.is_transposed: bool = False  # データが転置されているかどうか

    def run(self) -> None:
        """メインアプリケーションを実行します."""
        self._show_welcome()

        while True:
            choice = self._show_main_menu()

            if choice == "1. CSVファイルを読み込む":
                self._load_csv_file()
            elif choice == "2. データの概要を表示":
                self._show_data_overview()
            elif choice == "3. ヒストグラムを表示":
                self._show_histogram()
            elif choice == "4. 終了":
                console.print("\n[green]ありがとうございました！[/green]")
                break

    def _show_welcome(self) -> None:
        """ウェルカムメッセージを表示します."""
        welcome_text = """
[bold blue]データ概要表示CLI アプリケーション[/bold blue]

CSVファイルの基本情報を素早く確認できるシンプルなツールです。
ヒストグラム表示機能も搭載しています。
        """
        console.print(Panel(welcome_text, title="ようこそ", border_style="blue"))

    def _show_main_menu(self) -> str:
        """メインメニューを表示し、ユーザーの選択を取得します."""
        console.print("\n" + "=" * 40)
        console.print("[bold cyan]メインメニュー[/bold cyan]")
        console.print("=" * 40)

        # 現在の設定状況を表示
        if self.data is not None and self.file_path is not None:
            console.print(
                f"[dim]読み込み済みファイル: {os.path.basename(self.file_path)}[/dim]"
            )
            if self.display_mode == "simple":
                console.print("[dim]表示設定: 単純なヒストグラム[/dim]")
            elif self.display_mode == "class":
                console.print(
                    f"[dim]表示設定: クラス別色分け（{self.selected_class_column}）[/dim]"
                )
            else:
                console.print("[dim]表示設定: 未設定[/dim]")

        menu_choices = [
            "1. CSVファイルを読み込む",
            "2. データの概要を表示",
            "3. ヒストグラムを表示",
            "4. 終了",
        ]

        return questionary.select(
            "選択してください（矢印キーで移動、Enterで決定）:",
            choices=menu_choices,
            style=questionary.Style(
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
            ),
        ).ask()

    def _load_csv_file(self) -> None:
        """CSVファイルを読み込みます."""
        console.print("\n[bold]CSVファイルの読み込み[/bold]")

        # 読み込み方法を選択
        load_choices = [
            "1. 既定フォルダ（extraction_results）から選択",
            "2. 任意のパスを入力",
        ]

        load_choice = questionary.select(
            "読み込み方法を選択してください:",
            choices=load_choices,
            style=questionary.Style(
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
            ),
        ).ask()

        if not load_choice:
            return

        if load_choice.startswith("1."):
            file_path = self._select_csv_from_extraction_results()
        else:
            file_path = self._input_csv_path()

        if not file_path:
            return

        self._load_csv_from_path(file_path)

    def _select_csv_from_extraction_results(self) -> Optional[str]:
        """extraction_resultsフォルダからCSVファイルを選択します."""
        extraction_results_path = Path(__file__).parent.parent / "extraction_results"

        if not extraction_results_path.exists():
            console.print(
                f"[red]extraction_resultsフォルダが見つかりません: {extraction_results_path}[/red]"
            )
            return None

        # サブフォルダを取得
        subdirs = [d for d in extraction_results_path.iterdir() if d.is_dir()]

        if not subdirs:
            console.print(
                "[red]extraction_resultsフォルダにサブフォルダが見つかりません。[/red]"
            )
            return None

        console.print(f"\n[bold]フォルダ選択（{len(subdirs)}個のフォルダ）[/bold]")

        # フォルダ選択肢を作成
        folder_choices = []
        for subdir in sorted(subdirs):
            # フォルダ内のCSVファイル数を確認
            csv_files = list(subdir.glob("*.csv"))
            choice_text = f"{subdir.name} ({len(csv_files)}個のCSVファイル)"
            folder_choices.append(choice_text)

        selected_folder_text = questionary.select(
            "フォルダを選択してください（矢印キーで移動、Enterで決定）:",
            choices=folder_choices,
            style=questionary.Style(
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
            ),
        ).ask()

        if not selected_folder_text:
            return None

        # 選択されたフォルダ名を抽出
        selected_folder_name = selected_folder_text.split(" (")[0]
        selected_folder_path = extraction_results_path / selected_folder_name

        # フォルダ内のCSVファイルを取得
        csv_files = list(selected_folder_path.glob("*.csv"))

        if not csv_files:
            console.print(
                f"[red]選択されたフォルダにCSVファイルが見つかりません: {selected_folder_path}[/red]"
            )
            return None

        if len(csv_files) == 1:
            # CSVファイルが1つの場合は自動選択
            selected_csv = csv_files[0]
            console.print(
                f"[green]✓ CSVファイルを自動選択: {selected_csv.name}[/green]"
            )
        else:
            # 複数のCSVファイルがある場合は選択
            console.print(
                f"\n[bold]CSVファイル選択（{len(csv_files)}個のファイル）[/bold]"
            )

            csv_choices = []
            for csv_file in sorted(csv_files):
                # ファイルサイズを取得
                file_size = csv_file.stat().st_size
                if file_size < 1024:
                    size_str = f"{file_size}B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f}KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f}MB"

                choice_text = f"{csv_file.name} ({size_str})"
                csv_choices.append(choice_text)

            selected_csv_text = questionary.select(
                "CSVファイルを選択してください（矢印キーで移動、Enterで決定）:",
                choices=csv_choices,
                style=questionary.Style(
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
                ),
            ).ask()

            if not selected_csv_text:
                return None

            # 選択されたCSVファイル名を抽出
            selected_csv_name = selected_csv_text.split(" (")[0]
            selected_csv = selected_folder_path / selected_csv_name

        return str(selected_csv)

    def _input_csv_path(self) -> Optional[str]:
        """CSVファイルのパスを手動入力します."""
        file_path = Prompt.ask("CSVファイルのパスを入力してください")

        if not file_path:
            return None

        return file_path

    def _load_csv_from_path(self, file_path: str) -> None:
        """指定されたパスからCSVファイルを読み込みます."""
        if not os.path.exists(file_path):
            console.print(f"[red]ファイルが見つかりません: {file_path}[/red]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("ファイルを読み込み中...", total=None)
                self.data = pd.read_csv(file_path)
                self.file_path = file_path
                # 新しいファイルを読み込んだ場合は表示設定をリセット
                self.display_mode = None
                self.selected_class_column = None
                self.is_transposed = False  # 転置状態もリセット
                progress.update(task, completed=True)

            console.print("[green]✓ ファイルを正常に読み込みました[/green]")
            console.print(f"  - ファイル: {file_path}")
            console.print(f"  - 行数: {len(self.data):,}")
            console.print(f"  - 列数: {len(self.data.columns):,}")

            # CSV読み込み直後にLong formatデータを検出
            self._check_and_suggest_data_transformation()

        except Exception as e:
            console.print(f"[red]ファイルの読み込みに失敗しました: {str(e)}[/red]")

    def _check_and_suggest_data_transformation(self) -> None:
        """データの形式をチェックし、必要に応じて変換を提案します."""
        if self.data is None:
            return

        # Long format（縦持ち）データの検出
        long_format_detected = self._detect_long_format()

        if long_format_detected:
            self._suggest_pivot_transformation()
            return

        # 従来の転置判定
        rows, cols = self.data.shape
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        suggest_transpose = False
        reasons = []

        if rows > cols * 2:  # 行数が列数の2倍以上
            suggest_transpose = True
            reasons.append(f"行数（{rows:,}）が列数（{cols:,}）より大幅に多い")

        numeric_ratio = len(numeric_columns) / cols
        if numeric_ratio < 0.3:  # 数値列が30%未満
            suggest_transpose = True
            reasons.append(f"数値列の割合が低い（{numeric_ratio:.1%}）")

        if suggest_transpose and not self.is_transposed:
            console.print("\n" + "=" * 50)
            console.print("[bold yellow]縦持ちデータの可能性があります[/bold yellow]")
            console.print("=" * 50)

            console.print("[dim]検出された特徴：[/dim]")
            for reason in reasons:
                console.print(f"  • {reason}")

            console.print(
                "\n[dim]縦持ちデータの場合、行と列を入れ替える（転置）ことで[/dim]"
            )
            console.print("[dim]ヒストグラム表示などの機能が正常に動作します。[/dim]")

            transpose_choice = questionary.confirm(
                "データを転置しますか？", default=False
            ).ask()

            if transpose_choice:
                self._transpose_data()

    def _show_data_overview(self) -> None:
        """データの概要を表示します."""
        if self.data is None:
            console.print("[red]まずCSVファイルを読み込んでください。[/red]")
            return

        if self.file_path is None:
            console.print("[red]ファイルパスが設定されていません。[/red]")
            return

        console.print(f"\n[bold]データ概要: {os.path.basename(self.file_path)}[/bold]")

        # 転置状態を表示
        if self.is_transposed:
            console.print("[dim]（データは変換済みです）[/dim]")

        # 基本情報テーブル
        basic_info_table = Table(
            title="基本情報", show_header=True, header_style="bold magenta"
        )
        basic_info_table.add_column("項目", style="cyan", width=20)
        basic_info_table.add_column("値", style="white", width=15)

        basic_info_table.add_row("ファイルパス", self.file_path)
        basic_info_table.add_row("行数", f"{len(self.data):,}")
        basic_info_table.add_row("列数", f"{len(self.data.columns):,}")
        basic_info_table.add_row(
            "総セル数", f"{len(self.data) * len(self.data.columns):,}"
        )
        basic_info_table.add_row("欠損値総数", f"{self.data.isnull().sum().sum():,}")
        basic_info_table.add_row("重複行数", f"{self.data.duplicated().sum():,}")

        console.print(basic_info_table)

        # データ型の集計
        dtype_counts = self.data.dtypes.value_counts()
        console.print("\n[bold]データ型の分布[/bold]")
        dtype_table = Table(show_header=True, header_style="bold magenta")
        dtype_table.add_column("データ型", style="yellow", width=15)
        dtype_table.add_column("列数", style="green", width=10)
        dtype_table.add_column("割合", style="blue", width=10)

        for dtype, count in dtype_counts.items():
            percentage = (count / len(self.data.columns)) * 100
            dtype_table.add_row(str(dtype), str(count), f"{percentage:.1f}%")

        console.print(dtype_table)

        # 各列の詳細情報
        console.print("\n[bold]列の詳細情報[/bold]")
        columns_table = Table(show_header=True, header_style="bold magenta")
        columns_table.add_column("No.", style="dim", width=5)
        columns_table.add_column("列名", style="cyan", width=25)
        columns_table.add_column("データ型", style="yellow", width=12)
        columns_table.add_column("欠損値", style="red", width=8)
        columns_table.add_column("ユニーク値", style="green", width=10)
        columns_table.add_column("欠損率", style="magenta", width=8)

        for i, col in enumerate(self.data.columns, 1):
            null_count = self.data[col].isnull().sum()
            null_percentage = (null_count / len(self.data)) * 100
            unique_count = self.data[col].nunique()

            # 列名が長い場合は省略
            display_name = col if len(col) <= 25 else col[:22] + "..."

            columns_table.add_row(
                str(i),
                display_name,
                str(self.data[col].dtype),
                str(null_count),
                str(unique_count),
                f"{null_percentage:.1f}%",
            )

        console.print(columns_table)

        # 数値列の基本統計量（全列表示）
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            console.print(
                f"\n[bold]数値列の基本統計量（{len(numeric_columns)}列）[/bold]"
            )
            stats_df = self.data[numeric_columns].describe()

            stats_table = Table(show_header=True, header_style="bold magenta")
            stats_table.add_column("特徴量", style="cyan", width=20)
            stats_table.add_column("平均", style="white", width=12)
            stats_table.add_column("標準偏差", style="white", width=12)
            stats_table.add_column("最小値", style="white", width=12)
            stats_table.add_column("25%", style="white", width=12)
            stats_table.add_column("50%", style="white", width=12)
            stats_table.add_column("75%", style="white", width=12)
            stats_table.add_column("最大値", style="white", width=12)

            for col in numeric_columns:
                # 列名が長い場合は省略
                display_name = col if len(col) <= 20 else col[:17] + "..."

                col_stats = stats_df[col]
                stats_table.add_row(
                    display_name,
                    f"{col_stats['mean']:.2f}",
                    f"{col_stats['std']:.2f}",
                    f"{col_stats['min']:.2f}",
                    f"{col_stats['25%']:.2f}",
                    f"{col_stats['50%']:.2f}",
                    f"{col_stats['75%']:.2f}",
                    f"{col_stats['max']:.2f}",
                )

            console.print(stats_table)

    def _show_histogram(self) -> None:
        """ヒストグラムを表示します."""
        if self.data is None:
            console.print("[red]まずCSVファイルを読み込んでください。[/red]")
            return

        console.print("\n[bold]ヒストグラム表示[/bold]")

        # 数値列を取得
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            console.print("[red]数値列が見つかりません。[/red]")
            return

        # 初回のみ表示設定を選択
        if self.display_mode is None:
            self._configure_display_settings()

        # 現在の表示設定を表示（特徴量選択の前に表示）
        console.print("\n" + "=" * 40)
        if self.display_mode == "simple":
            console.print("[bold cyan]表示設定: 単純なヒストグラム[/bold cyan]")
        elif self.display_mode == "class":
            console.print(
                f"[bold cyan]表示設定: クラス別色分け（{self.selected_class_column}）[/bold cyan]"
            )
        console.print("=" * 40)

        # 特徴量を選択（矢印キーで選択）
        console.print(
            f"\n[bold]特徴量選択（{len(numeric_columns)}個の数値列から選択）[/bold]"
        )

        # 特徴量の選択肢を作成（データ型と欠損値情報付き）
        feature_choices = []
        for col in numeric_columns:
            null_count = self.data[col].isnull().sum()
            dtype_str = str(self.data[col].dtype)
            choice_text = f"{col} ({dtype_str}, 欠損値: {null_count})"
            feature_choices.append(choice_text)

        selected_feature_text = questionary.select(
            "ヒストグラムを表示する特徴量を選択してください（矢印キーで移動、Enterで決定）:",
            choices=feature_choices,
            style=questionary.Style(
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
            ),
        ).ask()

        if not selected_feature_text:
            return

        # 選択された特徴量名を抽出
        selected_feature = selected_feature_text.split(" (")[0]
        console.print(f"[green]✓ 選択された特徴量: {selected_feature}[/green]")

        # 設定に基づいてヒストグラムを表示
        if self.display_mode == "simple":
            self._show_simple_histogram(selected_feature)
        elif self.display_mode == "class" and self.selected_class_column is not None:
            self._show_class_histogram(selected_feature, self.selected_class_column)

    def _change_display_settings(self) -> None:
        """表示設定を変更します."""
        if self.data is None:
            console.print("[red]まずCSVファイルを読み込んでください。[/red]")
            return

        console.print("\n[bold]表示設定の変更[/bold]")

        # 現在の設定を表示
        if self.display_mode == "simple":
            console.print("[dim]現在の設定: 単純なヒストグラム[/dim]")
        elif self.display_mode == "class":
            console.print(
                f"[dim]現在の設定: クラス別色分け（{self.selected_class_column}）[/dim]"
            )
        else:
            console.print("[dim]現在の設定: 未設定[/dim]")

        # 設定を強制的にリセットして再設定
        self.display_mode = None
        self.selected_class_column = None

        console.print("\n[yellow]新しい表示設定を選択してください[/yellow]")
        self._configure_display_settings()

    def _configure_display_settings(self) -> None:
        """表示設定を構成します."""
        if self.data is None:
            return

        # カテゴリ列があるかチェック
        categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # カテゴリ列がある場合のみ色分けオプションを表示
        if categorical_columns:
            console.print("\n[bold]表示設定[/bold]")

            display_choices = [
                "1. 単純なヒストグラム（推奨）",
                "2. クラス別色分けヒストグラム",
            ]

            display_choice = questionary.select(
                "表示方法を選択してください:",
                choices=display_choices,
                style=questionary.Style(
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
                ),
            ).ask()

            if not display_choice:
                return

            if display_choice.startswith("1."):
                self.display_mode = "simple"
                console.print("[green]✓ 単純なヒストグラム表示に設定しました[/green]")
            else:
                # クラス列を選択
                console.print(
                    f"\n[bold]クラス列選択（{len(categorical_columns)}個のカテゴリ列）[/bold]"
                )

                class_choices = []
                for col in categorical_columns:
                    unique_count = self.data[col].nunique()
                    choice_text = f"{col} ({unique_count}種類)"
                    class_choices.append(choice_text)

                selected_class_text = questionary.select(
                    "色分けに使用するクラス列を選択してください:",
                    choices=class_choices,
                    style=questionary.Style(
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
                    ),
                ).ask()

                if not selected_class_text:
                    return

                self.selected_class_column = selected_class_text.split(" (")[0]
                self.display_mode = "class"
                console.print(
                    f"[green]✓ {self.selected_class_column} でクラス別色分け表示に設定しました[/green]"
                )
        else:
            console.print(
                "[yellow]カテゴリ列が見つかりません。単純なヒストグラム表示に設定します。[/yellow]"
            )
            self.display_mode = "simple"

    def _show_simple_histogram(self, column_name: str) -> None:
        """単純なヒストグラムを表示します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        data = self.data[column_name].dropna()

        if len(data) == 0:
            console.print("[red]有効なデータがありません。[/red]")
            return

        console.print(f"\n[bold]{column_name} のヒストグラム[/bold]")

        # plotextでヒストグラムを表示
        plt.clear_data()
        plt.hist(data.values, bins=20)
        plt.title(f"{column_name} のヒストグラム")
        plt.xlabel(column_name)
        plt.ylabel("頻度")
        plt.show()

        # 基本統計量も表示
        console.print("\n[bold]基本統計量[/bold]")
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("統計量", style="cyan", width=15)
        stats_table.add_column("値", style="white", width=15)

        stats_table.add_row("データ数", f"{len(data):,}")
        stats_table.add_row("平均", f"{data.mean():.4f}")
        stats_table.add_row("標準偏差", f"{data.std():.4f}")
        stats_table.add_row("最小値", f"{data.min():.4f}")
        stats_table.add_row("最大値", f"{data.max():.4f}")

        console.print(stats_table)

        # ヒストグラム表示後のメニューを表示
        self._show_post_histogram_menu()

    def _show_class_histogram(self, column_name: str, class_column: str) -> None:
        """クラス別に色分けしたヒストグラムを表示します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        console.print(
            f"\n[bold]{column_name} のクラス別ヒストグラム（{class_column}で色分け）[/bold]"
        )

        # クラスごとのデータを取得
        classes = self.data[class_column].unique()
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
            class_data = self.data[self.data[class_column] == class_name][
                column_name
            ].dropna()
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
            class_data = self.data[self.data[class_column] == class_name][
                column_name
            ].dropna()
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

        # ヒストグラム表示後のメニューを表示
        self._show_post_histogram_menu()

    def _show_post_histogram_menu(self) -> None:
        """ヒストグラム表示後のメニューを表示します."""
        console.print("\n" + "=" * 40)
        console.print("[bold cyan]次のアクション[/bold cyan]")
        console.print("=" * 40)

        menu_choices = [
            "1. 別の特徴量のヒストグラムを表示",
            "2. 表示設定を変更",
            "3. メインメニューに戻る",
        ]

        choice = questionary.select(
            "次に何をしますか？（矢印キーで移動、Enterで決定）:",
            choices=menu_choices,
            style=questionary.Style(
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
            ),
        ).ask()

        if choice == "1. 別の特徴量のヒストグラムを表示":
            self._show_histogram()
        elif choice == "2. 表示設定を変更":
            self._change_display_settings()
        elif choice == "3. メインメニューに戻る":
            return  # メインループに戻る

    def _detect_long_format(self) -> bool:
        """データがロング形式（縦型）かどうかを検出します."""
        if self.data is None:
            return False

        columns = self.data.columns.tolist()

        # Long formatの典型的なパターンを検出
        feature_name_candidates = []
        feature_value_candidates = []

        # 特徴量名を表す列を検出
        for col in columns:
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in ["feature", "variable", "metric", "measure"]
            ):
                if "name" in col_lower or "type" in col_lower:
                    feature_name_candidates.append(col)
                elif "value" in col_lower or "val" in col_lower:
                    feature_value_candidates.append(col)

        # より一般的なパターンも検出
        for col in columns:
            col_lower = col.lower()
            if col_lower in [
                "feature_name",
                "variable_name",
                "metric_name",
                "measure_name",
            ]:
                feature_name_candidates.append(col)
            elif col_lower in [
                "feature_value",
                "variable_value",
                "metric_value",
                "measure_value",
                "value",
            ]:
                feature_value_candidates.append(col)

        return len(feature_name_candidates) > 0 and len(feature_value_candidates) > 0

    def _suggest_pivot_transformation(self) -> None:
        """ピボット変換を提案し、実行します."""
        if self.data is None:
            return

        console.print("\n" + "=" * 50)
        console.print(
            "[bold yellow]Long format（縦持ち）データが検出されました[/bold yellow]"
        )
        console.print("=" * 50)

        columns = self.data.columns.tolist()

        # 特徴量名列と値列を特定
        feature_name_col: Optional[str] = None
        feature_value_col: Optional[str] = None

        for col in columns:
            col_lower = col.lower()
            if "feature" in col_lower and "name" in col_lower:
                feature_name_col = col
            elif "feature" in col_lower and "value" in col_lower:
                feature_value_col = col

        if not feature_name_col or not feature_value_col:
            # より一般的な検出
            for col in columns:
                col_lower = col.lower()
                if col_lower in ["feature_name", "variable_name", "metric_name"]:
                    feature_name_col = col
                elif col_lower in ["feature_value", "value", "val"]:
                    feature_value_col = col

        if not feature_name_col or not feature_value_col:
            console.print("[red]適切な特徴量名列または値列が見つかりません。[/red]")
            return

        console.print("[dim]検出された構造：[/dim]")
        console.print(f"  • 特徴量名列: {feature_name_col}")
        console.print(f"  • 特徴量値列: {feature_value_col}")

        # 識別列（ID列）を特定
        id_columns = [
            col for col in columns if col not in [feature_name_col, feature_value_col]
        ]
        console.print(f"  • 識別列: {', '.join(id_columns)}")

        # ユニークな特徴量数を表示
        unique_features = self.data[feature_name_col].nunique()
        console.print(f"  • ユニークな特徴量数: {unique_features:,}")

        console.print(
            "\n[dim]Long formatデータを横持ち（Wide format）に変換することで[/dim]"
        )
        console.print("[dim]ヒストグラム表示などの機能が正常に動作します。[/dim]")

        pivot_choice = questionary.confirm(
            "データをWide format（横持ち）に変換しますか？", default=True
        ).ask()

        if pivot_choice:
            self._pivot_long_to_wide(feature_name_col, feature_value_col, id_columns)

    def _pivot_long_to_wide(
        self, feature_name_col: str, feature_value_col: str, id_columns: list
    ) -> None:
        """ロング形式のデータをワイド形式に変換します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        try:
            console.print("\n[yellow]データをWide formatに変換しています...[/yellow]")

            # 複数のID列がある場合の処理
            if len(id_columns) > 1:
                # どの列をインデックスにするか選択
                console.print("\n[bold]インデックス列の選択[/bold]")
                console.print(
                    "[dim]複数の識別列があります。メインのインデックスとして使用する列を選択してください。[/dim]"
                )

                index_choices = []
                for col in id_columns:
                    unique_count = self.data[col].nunique()
                    choice_text = f"{col} ({unique_count:,}個のユニーク値)"
                    index_choices.append(choice_text)

                selected_index_text = questionary.select(
                    "メインインデックス列を選択してください:",
                    choices=index_choices,
                    style=questionary.Style(
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
                    ),
                ).ask()

                if not selected_index_text:
                    return

                main_index_col = selected_index_text.split(" (")[0]
                other_cols = [col for col in id_columns if col != main_index_col]

                # 他の列は最初の値を使用するか、削除するかを選択
                if other_cols:
                    keep_other_choice = questionary.confirm(
                        f"他の識別列（{', '.join(other_cols)}）も保持しますか？",
                        default=False,
                    ).ask()

                    if keep_other_choice:
                        # 他の列の最初の値を使用
                        pivot_data = (
                            self.data.groupby([main_index_col, feature_name_col])[
                                feature_value_col
                            ]
                            .first()
                            .reset_index()
                        )
                        wide_data = pivot_data.pivot(
                            index=main_index_col,
                            columns=feature_name_col,
                            values=feature_value_col,
                        )

                        # 他の列の情報を追加
                        for col in other_cols:
                            col_values = self.data.groupby(main_index_col)[col].first()
                            wide_data[col] = col_values
                    else:
                        # メインインデックスのみ使用
                        pivot_data = (
                            self.data.groupby([main_index_col, feature_name_col])[
                                feature_value_col
                            ]
                            .first()
                            .reset_index()
                        )
                        wide_data = pivot_data.pivot(
                            index=main_index_col,
                            columns=feature_name_col,
                            values=feature_value_col,
                        )
                else:
                    pivot_data = (
                        self.data.groupby([main_index_col, feature_name_col])[
                            feature_value_col
                        ]
                        .first()
                        .reset_index()
                    )
                    wide_data = pivot_data.pivot(
                        index=main_index_col,
                        columns=feature_name_col,
                        values=feature_value_col,
                    )
            else:
                # 単一のID列の場合
                main_index_col = id_columns[0]
                wide_data = self.data.pivot(
                    index=main_index_col,
                    columns=feature_name_col,
                    values=feature_value_col,
                )

            # インデックスをリセットして通常の列にする
            wide_data = wide_data.reset_index()

            # 列名をクリーンアップ
            wide_data.columns.name = None

            # データを更新
            self.data = wide_data
            self.is_transposed = True  # 変換済みフラグ

            # 表示設定をリセット
            self.display_mode = None
            self.selected_class_column = None

            console.print("[green]✓ Wide formatへの変換が完了しました[/green]")
            console.print(f"  - 新しい行数: {len(self.data):,}")
            console.print(f"  - 新しい列数: {len(self.data.columns):,}")

            # 変換後のデータ概要を表示
            console.print("\n[bold]変換後のデータ概要[/bold]")
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            console.print(f"  - 数値列: {len(numeric_columns):,}個")
            console.print(
                f"  - 数値列の割合: {len(numeric_columns) / len(self.data.columns):.1%}"
            )

        except Exception as e:
            console.print(f"[red]Wide formatへの変換に失敗しました: {str(e)}[/red]")
            console.print(f"[red]詳細: {type(e).__name__}[/red]")

    def _transpose_data(self) -> None:
        """データを転置します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        try:
            console.print("\n[yellow]データを転置しています...[/yellow]")

            # データを転置
            self.data = self.data.T
            self.is_transposed = not self.is_transposed

            # 表示設定をリセット（新しいデータ構造に対応）
            self.display_mode = None
            self.selected_class_column = None

            console.print("[green]✓ データの転置が完了しました[/green]")
            console.print(f"  - 新しい行数: {len(self.data):,}")
            console.print(f"  - 新しい列数: {len(self.data.columns):,}")

            # 転置後のデータ概要を表示
            console.print("\n[bold]転置後のデータ概要[/bold]")
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            console.print(f"  - 数値列: {len(numeric_columns):,}個")
            console.print(
                f"  - 数値列の割合: {len(numeric_columns) / len(self.data.columns):.1%}"
            )

        except Exception as e:
            console.print(f"[red]データの転置に失敗しました: {str(e)}[/red]")


@click.command()
def main():
    """データ概要表示CLIアプリケーションのメインエントリーポイント."""
    app = DataOverviewCLI()
    app.run()


if __name__ == "__main__":
    main()
