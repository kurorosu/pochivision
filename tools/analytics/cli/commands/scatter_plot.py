"""Scatter plot display commands for CSV Analytics."""

from typing import List, Optional

from analytics.core.data_processor import DataProcessor
from analytics.ui.display import (
    create_scatter_feature_choices,
    show_class_scatter_plot,
    show_post_scatter_plot_menu,
    show_scatter_plot_header,
    show_simple_scatter_plot,
)
from analytics.ui.prompts import (
    select_class_column,
    select_post_scatter_plot_action,
    select_scatter_display_mode,
    select_x_axis_feature,
    select_y_axis_feature,
)
from analytics.utils.file_utils import export_correlation_features_to_csv
from rich.console import Console

console = Console()


class ScatterPlotManager:
    """散布図表示を管理するクラス."""

    def __init__(self):
        """初期化処理."""
        self.display_mode: Optional[str] = None  # 'simple' or 'class'
        self.selected_class_column: Optional[str] = None  # クラス別表示時のクラス列
        self.selected_y_axis: Optional[str] = None  # 固定するy軸の特徴量

    def show_scatter_plot_command(self, data_processor: DataProcessor) -> bool:
        """散布図を表示するコマンドを実行します."""
        if data_processor.data is None:
            console.print("[red]まずCSVファイルを読み込んでください。[/red]")
            return False

        # 数値列を取得
        numeric_columns = data_processor.get_numeric_columns()
        if len(numeric_columns) < 2:
            console.print("[red]散布図には少なくとも2つの数値列が必要です。[/red]")
            return False

        # 初回のみ表示設定を選択
        if self.display_mode is None:
            self._configure_display_settings(data_processor)

        # y軸の特徴量が未選択の場合は選択
        if self.selected_y_axis is None:
            self._select_y_axis_feature(numeric_columns)
            if self.selected_y_axis is None:
                return False

        # 散布図表示のヘッダーを表示
        show_scatter_plot_header(
            self.display_mode or "simple",
            self.selected_class_column,
            self.selected_y_axis,
        )

        # x軸特徴量を選択（y軸との相関係数順でソート）
        class_column = (
            self.selected_class_column if self.display_mode == "class" else None
        )
        x_axis_choices = create_scatter_feature_choices(
            data_processor.data, self.selected_y_axis, class_column
        )

        console.print(f"\n[bold]X軸特徴量選択（y軸: {self.selected_y_axis}）[/bold]")
        console.print(f"[dim]{len(x_axis_choices)}個の特徴量を相関係数順で表示[/dim]")

        # クラス分け散布図時のみCSV出力オプションを表示
        show_export_option = self.display_mode == "class"
        selected_x_feature = select_x_axis_feature(x_axis_choices, show_export_option)
        if not selected_x_feature:
            return False

        # CSV出力が選択された場合
        if selected_x_feature == "EXPORT_CORRELATION_FEATURES":
            self._export_correlation_features(data_processor, x_axis_choices)
            return True

        console.print(f"[green]✓ 選択されたX軸特徴量: {selected_x_feature}[/green]")

        # 設定に基づいて散布図を表示
        if self.display_mode == "simple":
            show_simple_scatter_plot(
                data_processor.data, selected_x_feature, self.selected_y_axis
            )
        elif self.display_mode == "class" and self.selected_class_column is not None:
            show_class_scatter_plot(
                data_processor.data,
                selected_x_feature,
                self.selected_y_axis,
                self.selected_class_column,
            )

        # 散布図表示後のメニューを表示
        self._show_post_scatter_plot_menu(data_processor)
        return True

    def _configure_display_settings(self, data_processor: DataProcessor) -> None:
        """表示設定を構成します."""
        if data_processor.data is None:
            return

        # カテゴリ列があるかチェック
        categorical_columns = data_processor.get_categorical_columns()

        if not categorical_columns:
            # カテゴリ列がない場合は単純な散布図のみ
            console.print(
                "[dim]カテゴリ列が見つからないため、単純な散布図を使用します。[/dim]"
            )
            self.display_mode = "simple"
            return

        # 表示モードを選択（散布図用）
        display_choice = select_scatter_display_mode()
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

    def _select_y_axis_feature(self, numeric_columns: List[str]) -> None:
        """y軸の特徴量を選択します."""
        self.selected_y_axis = select_y_axis_feature(numeric_columns)

    def _show_post_scatter_plot_menu(self, data_processor: DataProcessor) -> None:
        """散布図表示後のメニューを表示します."""
        show_post_scatter_plot_menu()

        while True:
            action_choice = select_post_scatter_plot_action()
            if not action_choice:
                break

            if action_choice.startswith("1."):
                # 別のX軸特徴量で散布図を表示
                self.show_scatter_plot_command(data_processor)
                break
            elif action_choice.startswith("2."):
                # Y軸特徴量を変更
                numeric_columns = data_processor.get_numeric_columns()
                self._select_y_axis_feature(numeric_columns)
                if self.selected_y_axis:
                    self.show_scatter_plot_command(data_processor)
                break
            elif action_choice.startswith("3."):
                # 表示設定を変更
                self._change_display_settings(data_processor)
                break
            elif action_choice.startswith("4."):
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
            console.print("[dim]現在の設定: 単純な散布図[/dim]")
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
        self.selected_y_axis = None

    def _export_correlation_features(
        self, data_processor: DataProcessor, feature_choices: List[str]
    ) -> None:
        """相関係数順上位N位の特徴量をCSVファイルに出力します."""
        if data_processor.file_path is None:
            console.print("[red]CSVファイルのパスが取得できません。[/red]")
            return

        console.print("\n[bold]相関係数順上位の特徴量をCSV出力[/bold]")

        # 順位数を選択
        from analytics.ui.prompts import select_ranking_count

        ranking_count = select_ranking_count()
        if ranking_count is None:
            return

        # models{index}フォルダに出力
        from pathlib import Path

        from analytics.core.classification_modeler import ClassificationModeler

        # models{index}フォルダを作成
        data_dir = Path(data_processor.file_path).parent
        modeler = ClassificationModeler()
        models_dir = modeler._create_models_directory(data_dir)

        # 相関特徴量CSVを出力
        output_path = export_correlation_features_to_csv(
            feature_choices,
            data_processor.file_path,
            ranking_count,
            self.selected_y_axis,
            models_dir=models_dir,
        )

        if output_path:
            console.print(
                f"\n[green]✓ models{models_dir.name}フォルダに保存しました[/green]"
            )
            console.print(
                "\n[dim]このCSVファイルをPythonで読み込むには以下のコードを使用してください:[/dim]"
            )
            console.print("[dim]import pandas as pd[/dim]")
            console.print(f"[dim]df = pd.read_csv('{output_path}')[/dim]")
            console.print("[dim]features = df['feature_name'].tolist()[/dim]")
            console.print("[dim]print(features)[/dim]")
        else:
            console.print("[red]CSV出力に失敗しました。[/red]")
