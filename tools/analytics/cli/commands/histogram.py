"""Histogram display commands for CSV Analytics."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from analytics.core.classification_modeler import ClassificationModeler
from analytics.core.data_processor import DataProcessor
from analytics.ui.display import (
    create_feature_choices,
    show_class_histogram,
    show_histogram_header,
    show_post_histogram_menu,
    show_simple_histogram,
)
from analytics.ui.prompts import (
    confirm_classification_modeling,
    extract_feature_name_from_choice,
    select_class_column,
    select_display_mode,
    select_feature_for_histogram,
    select_post_histogram_action,
)
from analytics.utils.file_utils import export_topn_features_to_csv
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

        # 特徴量を選択（クラス別表示の場合は分離度順でソート）
        class_column = (
            self.selected_class_column if self.display_mode == "class" else None
        )
        feature_choices = create_feature_choices(data_processor.data, class_column)

        if class_column:
            console.print(
                f"\n[bold]特徴量選択（{len(numeric_columns)}個の数値列を分離度順で表示）[/bold]"
            )
        else:
            console.print(
                f"\n[bold]特徴量選択（{len(numeric_columns)}個の数値列から選択）[/bold]"
            )

        # クラス分けヒストグラム時のみCSV出力オプションを表示
        show_export_option = self.display_mode == "class"
        selected_feature = select_feature_for_histogram(
            feature_choices, show_export_option
        )
        if not selected_feature:
            return False

        # CSV出力が選択された場合
        if selected_feature == "EXPORT_TOPN_FEATURES":
            self._export_topn_features(data_processor, feature_choices)
            return True

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

    def _export_topn_features(
        self, data_processor: DataProcessor, feature_choices: List[str]
    ) -> None:
        """JSD距離順上位N位の特徴量をCSVファイルに出力します."""
        if data_processor.file_path is None:
            console.print("[red]CSVファイルのパスが取得できません。[/red]")
            return

        console.print("\n[bold]JSD距離順上位の特徴量をCSV出力[/bold]")

        # 順位数を選択
        from analytics.ui.prompts import select_ranking_count

        ranking_count = select_ranking_count()
        if ranking_count is None:
            return

        # モデリング確認プロンプトを表示（色分けヒストグラム時のみ）
        will_execute_modeling = False
        if self.display_mode == "class" and self.selected_class_column is not None:
            console.print("\n[bold cyan]分類モデリング機能[/bold cyan]")
            will_execute_modeling = confirm_classification_modeling()

        if will_execute_modeling:
            # モデリングを実行する場合は、models{index}フォルダのみに出力
            console.print(
                "[dim]特徴量CSVはモデリング結果と共にmodels{index}フォルダに出力されます[/dim]"
            )
            self._execute_classification_modeling(
                data_processor, feature_choices, ranking_count
            )
        else:
            # モデリングを実行しない場合は、従来通りの場所に出力
            output_path = export_topn_features_to_csv(
                feature_choices,
                data_processor.file_path,
                ranking_count,
                models_dir=None,
            )

            if output_path:
                console.print(
                    "\n[dim]このCSVファイルをPythonで読み込むには以下のコードを使用してください:[/dim]"
                )
                console.print("[dim]import pandas as pd[/dim]")
                console.print(f"[dim]df = pd.read_csv('{output_path}')[/dim]")
                console.print("[dim]features = df['feature_name'].tolist()[/dim]")
                console.print("[dim]print(features)[/dim]")
            else:
                console.print("[red]CSV出力に失敗しました。[/red]")

    def _execute_classification_modeling(
        self,
        data_processor: DataProcessor,
        feature_choices: List[str],
        ranking_count: int,
    ) -> None:
        """分類モデリングを実行します."""
        if data_processor.data is None or data_processor.file_path is None:
            console.print("[red]データまたはファイルパスが取得できません。[/red]")
            return

        try:
            console.print("\n[bold]分類モデリングを実行中...[/bold]")

            # 上位N位の特徴量名を抽出
            selected_features = []
            max_features = (
                len(feature_choices) if ranking_count == -1 else ranking_count
            )

            for i, choice in enumerate(feature_choices[:max_features]):
                feature_name = extract_feature_name_from_choice(choice)
                selected_features.append(feature_name)

            # モデリングを実行
            modeler = ClassificationModeler()
            accuracy_scores: Dict[str, Union[float, int]] = modeler.train_model(
                data_processor.data,
                selected_features,
                self.selected_class_column,
            )

            # 特徴量重要度を取得
            feature_importance = modeler.get_feature_importance()

            # 結果をCSVに保存
            result_path = modeler.save_model_results(
                data_processor.file_path,
                accuracy_scores,
                feature_importance,
            )

            # models{index}フォルダにJSD特徴量CSVを出力
            models_dir = Path(result_path).parent
            jsd_csv_path = export_topn_features_to_csv(
                feature_choices, data_processor.file_path, ranking_count, models_dir
            )

            # 結果を表示
            console.print("\n[green]✓ モデリングが完了しました！[/green]")
            console.print(f"[bold]結果ファイル:[/bold] {result_path}")
            console.print(f"[bold]目的変数:[/bold] {self.selected_class_column}")
            console.print(f"[bold]特徴量数:[/bold] {len(selected_features)}")

            train_acc = accuracy_scores["train_accuracy"]
            train_pct = train_acc * 100
            console.print(f"[bold]訓練精度:[/bold] {train_acc:.4f} ({train_pct:.2f}%)")

            test_acc = accuracy_scores["test_accuracy"]
            test_pct = test_acc * 100
            console.print(f"[bold]テスト精度:[/bold] {test_acc:.4f} ({test_pct:.2f}%)")

            n_samples = accuracy_scores["n_samples"]
            n_train = accuracy_scores["n_train"]
            n_test = accuracy_scores["n_test"]
            console.print(
                f"[bold]データ数:[/bold] {n_samples} "
                f"(訓練: {n_train}, テスト: {n_test})"
            )

            # JSD特徴量CSVの出力結果を表示
            if jsd_csv_path:
                console.print(f"\n[bold cyan]JSD特徴量CSV:[/bold cyan] {jsd_csv_path}")

            # PCA散布図の生成結果を表示
            if len(selected_features) >= 3:
                models_dir = Path(result_path).parent
                pca_plot_path = models_dir / "pca_scatter_plot.png"
                if pca_plot_path.exists():
                    console.print(
                        f"\n[bold cyan]PCA散布図:[/bold cyan] {pca_plot_path}"
                    )
                    if modeler.pca is not None:
                        total_variance = sum(modeler.pca.explained_variance_ratio_)
                        pc1_var = modeler.pca.explained_variance_ratio_[0]
                        pc2_var = modeler.pca.explained_variance_ratio_[1]
                        console.print(
                            f"[dim]寄与率 - 全体: {total_variance:.1%}, "
                            f"PC1: {pc1_var:.1%}, PC2: {pc2_var:.1%}[/dim]"
                        )
                else:
                    console.print("[yellow]PCA散布図の生成に失敗しました[/yellow]")
            else:
                console.print(
                    f"[dim]PCA散布図: 特徴量数が{len(selected_features)}"
                    f"のため生成されませんでした（3以上必要）[/dim]"
                )

            # 特徴量重要度上位3位を表示
            if feature_importance:
                console.print("\n[bold]特徴量重要度 Top 3:[/bold]")
                sorted_importance = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
                for i, (feature, importance) in enumerate(sorted_importance[:3], 1):
                    console.print(f"  {i}位: {feature} ({importance:.4f})")

        except Exception as e:
            console.print(f"[red]モデリングでエラーが発生しました: {str(e)}[/red]")
