"""Histogram display commands for CSV Analytics."""

from typing import Optional

from analytics.core.data_processor import DataProcessor
from analytics.ui.display import (
    create_feature_choices,
    show_class_histogram,
    show_histogram_header,
    show_post_histogram_menu,
    show_simple_histogram,
)
from analytics.ui.prompts import (
    select_class_column,
    select_display_mode,
    select_feature_for_histogram,
    select_post_histogram_action,
)
from rich.console import Console

console = Console()


class HistogramManager:
    """ヒストグラム表示を管理するクラス."""

    def __init__(self):
        """初期化処理."""
        self.display_mode: Optional[str] = None  # 'simple' or 'class'
        self.selected_class_column: Optional[str] = None  # クラス別表示時のクラス列

    def show_histogram_command(self, data_processor: DataProcessor) -> bool:
        """ヒストグラムを表示するコマンドを実行します."""
        if data_processor.data is None:
            console.print("[red]まずCSVファイルを読み込んでください。[/red]")
            return False

        # 数値列を取得
        numeric_columns = data_processor.get_numeric_columns()
        if not numeric_columns:
            console.print("[red]数値列が見つかりません。[/red]")
            return False

        # 初回のみ表示設定を選択
        if self.display_mode is None:
            self._configure_display_settings(data_processor)

        # ヒストグラム表示のヘッダーを表示
        show_histogram_header(self.display_mode or "simple", self.selected_class_column)

        # 特徴量を選択
        feature_choices = create_feature_choices(data_processor.data)
        console.print(
            f"\n[bold]特徴量選択（{len(numeric_columns)}個の数値列から選択）[/bold]"
        )

        selected_feature = select_feature_for_histogram(feature_choices)
        if not selected_feature:
            return False

        console.print(f"[green]✓ 選択された特徴量: {selected_feature}[/green]")

        # 設定に基づいてヒストグラムを表示
        if self.display_mode == "simple":
            show_simple_histogram(data_processor.data, selected_feature)
        elif self.display_mode == "class" and self.selected_class_column is not None:
            show_class_histogram(
                data_processor.data, selected_feature, self.selected_class_column
            )

        # ヒストグラム表示後のメニューを表示
        self._show_post_histogram_menu(data_processor)
        return True

    def _configure_display_settings(self, data_processor: DataProcessor) -> None:
        """表示設定を構成します."""
        if data_processor.data is None:
            return

        # カテゴリ列があるかチェック
        categorical_columns = data_processor.get_categorical_columns()

        if not categorical_columns:
            # カテゴリ列がない場合は単純なヒストグラムのみ
            console.print(
                "[dim]カテゴリ列が見つからないため、単純なヒストグラムを使用します。[/dim]"
            )
            self.display_mode = "simple"
            return

        # 表示モードを選択
        display_choice = select_display_mode()
        if not display_choice:
            self.display_mode = "simple"
            return

        if display_choice.startswith("1."):
            self.display_mode = "simple"
        else:
            self.display_mode = "class"
            # クラス列を選択
            self.selected_class_column = select_class_column(
                data_processor.data, categorical_columns
            )
            if not self.selected_class_column:
                self.display_mode = "simple"

    def _show_post_histogram_menu(self, data_processor: DataProcessor) -> None:
        """ヒストグラム表示後のメニューを表示します."""
        show_post_histogram_menu()

        while True:
            action_choice = select_post_histogram_action()
            if not action_choice:
                break

            if action_choice.startswith("1."):
                # 別の特徴量のヒストグラムを表示
                self.show_histogram_command(data_processor)
                break
            elif action_choice.startswith("2."):
                # 表示設定を変更
                self._change_display_settings(data_processor)
                break
            elif action_choice.startswith("3."):
                # メインメニューに戻る
                break

    def _change_display_settings(self, data_processor: DataProcessor) -> None:
        """表示設定を変更します."""
        if data_processor.data is None:
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

        # 新しい設定を選択
        self._configure_display_settings(data_processor)

        console.print("[green]✓ 表示設定を更新しました[/green]")

    def reset_settings(self) -> None:
        """設定をリセットします."""
        self.display_mode = None
        self.selected_class_column = None
