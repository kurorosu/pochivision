"""Data processing logic for CSV Analytics."""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from analytics.core.format_detector import (
    analyze_categorical_columns,
    analyze_transpose_necessity,
    detect_long_format,
    find_feature_columns,
    get_id_columns,
    get_unique_feature_count,
)
from analytics.core.pivot_transformer import (
    pivot_long_to_wide,
    transpose_data,
    validate_pivot_parameters,
    validate_transformation_result,
)
from analytics.ui.display import (
    show_data_transformation_suggestion,
    show_long_format_detection,
    show_pivot_structure,
    show_transformation_success,
)
from analytics.ui.prompts import confirm_data_transformation, select_index_column
from rich.console import Console

# プロジェクトルートをパスに追加
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
tools_path = project_root / "tools"
sys.path.insert(0, str(tools_path))


console = Console()


class DataProcessor:
    """データ処理を統合管理するクラス."""

    def __init__(self):
        """初期化処理."""
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.is_transposed: bool = False

    def load_data(self, data: pd.DataFrame, file_path: str) -> None:
        """データを読み込みます."""
        self.data = data
        self.file_path = file_path
        self.is_transposed = False

    def check_and_suggest_data_transformation(self) -> None:
        """データの形式をチェックし、必要に応じて変換を提案します."""
        if self.data is None:
            return

        # Long format（縦持ち）データの検出
        long_format_detected = detect_long_format(self.data)

        if long_format_detected:
            self._suggest_pivot_transformation()
            return

        # 従来の転置判定
        suggest_transpose, reasons = analyze_transpose_necessity(self.data)

        if suggest_transpose and not self.is_transposed:
            show_data_transformation_suggestion(reasons)

            transpose_choice = confirm_data_transformation(
                "データを転置しますか？", default=False
            )

            if transpose_choice:
                self._transpose_data()

    def _suggest_pivot_transformation(self) -> None:
        """ピボット変換を提案し、実行します."""
        if self.data is None:
            return

        show_long_format_detection()

        feature_name_col, feature_value_col = find_feature_columns(self.data)

        if not feature_name_col or not feature_value_col:
            console.print("[red]適切な特徴量名列または値列が見つかりません。[/red]")
            return

        # 識別列（ID列）を特定
        id_columns = get_id_columns(self.data, feature_name_col, feature_value_col)
        unique_features = get_unique_feature_count(self.data, feature_name_col)

        show_pivot_structure(
            feature_name_col, feature_value_col, id_columns, unique_features
        )

        pivot_choice = confirm_data_transformation(
            "データをWide format（横持ち）に変換しますか？", default=True
        )

        if pivot_choice:
            self._pivot_long_to_wide(feature_name_col, feature_value_col, id_columns)

    def _pivot_long_to_wide(
        self, feature_name_col: str, feature_value_col: str, id_columns: List[str]
    ) -> None:
        """ロング形式のデータをワイド形式に変換します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        # パラメータの検証
        if not validate_pivot_parameters(
            self.data, feature_name_col, feature_value_col, id_columns
        ):
            return

        main_index_col = None
        keep_other_columns = False

        # 複数のID列がある場合の処理
        if len(id_columns) > 1:
            console.print("\n[bold]インデックス列の選択[/bold]")
            console.print(
                "[dim]複数の識別列があります。メインのインデックスとして使用する列を選択してください。[/dim]"
            )

            index_choices = []
            for col in id_columns:
                unique_count = self.data[col].nunique()
                choice_text = f"{col} ({unique_count:,}個のユニーク値)"
                index_choices.append(choice_text)

            selected_index_text = select_index_column(index_choices)

            if not selected_index_text:
                return

            main_index_col = selected_index_text.split(" (")[0]
            other_cols = [col for col in id_columns if col != main_index_col]

            # 他の列は最初の値を使用するか、削除するかを選択
            if other_cols:
                keep_other_choice = confirm_data_transformation(
                    f"他の識別列（{', '.join(other_cols)}）も保持しますか？",
                    default=False,
                )
                keep_other_columns = keep_other_choice

        # ピボット変換を実行
        wide_data = pivot_long_to_wide(
            self.data,
            feature_name_col,
            feature_value_col,
            id_columns,
            main_index_col,
            keep_other_columns,
        )

        if wide_data is not None and validate_transformation_result(
            self.data, wide_data, "Wide formatへの変換"
        ):
            self.data = wide_data
            self.is_transposed = True  # 変換済みフラグ
            show_transformation_success(self.data, "Wide formatへの変換")

    def _transpose_data(self) -> None:
        """データを転置します."""
        if self.data is None:
            console.print("[red]データが読み込まれていません。[/red]")
            return

        transposed_data = transpose_data(self.data)

        if transposed_data is not None and validate_transformation_result(
            self.data, transposed_data, "データの転置"
        ):
            self.data = transposed_data
            self.is_transposed = not self.is_transposed
            show_transformation_success(self.data, "データの転置")

    def get_numeric_columns(self) -> List[str]:
        """数値列を取得します."""
        if self.data is None:
            return []
        return self.data.select_dtypes(include=["number"]).columns.tolist()

    def get_categorical_columns(self) -> List[str]:
        """カテゴリ列を取得します."""
        if self.data is None:
            return []
        return analyze_categorical_columns(self.data)

    def reset_transformation_state(self) -> None:
        """変換状態をリセットします."""
        self.is_transposed = False

    def get_data_info(self) -> dict:
        """データの基本情報を取得します."""
        if self.data is None:
            return {}

        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "numeric_columns": len(self.get_numeric_columns()),
            "categorical_columns": len(self.get_categorical_columns()),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            "is_transposed": self.is_transposed,
        }
