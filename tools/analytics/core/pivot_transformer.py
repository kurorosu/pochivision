"""Pivot transformation logic for CSV Analytics."""

from typing import List, Optional

import pandas as pd
from rich.console import Console

console = Console()


def pivot_long_to_wide(
    data: pd.DataFrame,
    feature_name_col: str,
    feature_value_col: str,
    id_columns: List[str],
    main_index_col: Optional[str] = None,
    keep_other_columns: bool = False,
) -> Optional[pd.DataFrame]:
    """ロング形式のデータをワイド形式に変換します."""
    try:
        console.print("\n[yellow]データをWide formatに変換しています...[/yellow]")

        if len(id_columns) > 1 and main_index_col:
            # 複数のID列がある場合の処理
            other_cols = [col for col in id_columns if col != main_index_col]

            if keep_other_columns and other_cols:
                # 他の列の最初の値を使用
                pivot_data = (
                    data.groupby([main_index_col, feature_name_col])[feature_value_col]
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
                    col_values = data.groupby(main_index_col)[col].first()
                    wide_data[col] = col_values
            else:
                # メインインデックスのみ使用
                pivot_data = (
                    data.groupby([main_index_col, feature_name_col])[feature_value_col]
                    .first()
                    .reset_index()
                )
                wide_data = pivot_data.pivot(
                    index=main_index_col,
                    columns=feature_name_col,
                    values=feature_value_col,
                )
        else:
            # 単一のID列の場合、または複数でもmain_index_colが指定されていない場合
            index_col = main_index_col if main_index_col else id_columns[0]
            wide_data = data.pivot(
                index=index_col,
                columns=feature_name_col,
                values=feature_value_col,
            )

        # インデックスをリセットして通常の列にする
        wide_data = wide_data.reset_index()

        # 列名をクリーンアップ
        wide_data.columns.name = None

        return wide_data

    except Exception as e:
        console.print(f"[red]Wide formatへの変換に失敗しました: {str(e)}[/red]")
        console.print(f"[red]詳細: {type(e).__name__}[/red]")
        return None


def transpose_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """データを転置します."""
    try:
        console.print("\n[yellow]データを転置しています...[/yellow]")
        return data.T
    except Exception as e:
        console.print(f"[red]データの転置に失敗しました: {str(e)}[/red]")
        return None


def validate_pivot_parameters(
    data: pd.DataFrame,
    feature_name_col: str,
    feature_value_col: str,
    id_columns: List[str],
) -> bool:
    """ピボット変換のパラメータを検証します."""
    # 必要な列が存在するかチェック
    if feature_name_col not in data.columns:
        console.print(f"[red]特徴量名列が見つかりません: {feature_name_col}[/red]")
        return False

    if feature_value_col not in data.columns:
        console.print(f"[red]特徴量値列が見つかりません: {feature_value_col}[/red]")
        return False

    for col in id_columns:
        if col not in data.columns:
            console.print(f"[red]ID列が見つかりません: {col}[/red]")
            return False

    # データが存在するかチェック
    if data[feature_name_col].isnull().all():
        console.print(f"[red]特徴量名列にデータがありません: {feature_name_col}[/red]")
        return False

    if data[feature_value_col].isnull().all():
        console.print(f"[red]特徴量値列にデータがありません: {feature_value_col}[/red]")
        return False

    return True


def analyze_pivot_complexity(
    data: pd.DataFrame,
    feature_name_col: str,
    feature_value_col: str,
    id_columns: List[str],
) -> dict:
    """ピボット変換の複雑さを分析します."""
    unique_features = data[feature_name_col].nunique()
    unique_ids = data[id_columns[0]].nunique() if id_columns else 0

    # 重複の確認
    duplicates = data.duplicated(subset=id_columns + [feature_name_col]).sum()

    # 予想される結果のサイズ
    expected_rows = unique_ids
    expected_cols = unique_features + len(id_columns)

    return {
        "unique_features": unique_features,
        "unique_ids": unique_ids,
        "duplicates": duplicates,
        "expected_rows": expected_rows,
        "expected_cols": expected_cols,
        "size_reduction": len(data) / expected_rows if expected_rows > 0 else 0,
    }


def create_index_column_choices(data: pd.DataFrame, id_columns: List[str]) -> List[str]:
    """インデックス列選択用の選択肢を作成します."""
    choices = []
    for col in id_columns:
        unique_count = data[col].nunique()
        choice_text = f"{col} ({unique_count:,}個のユニーク値)"
        choices.append(choice_text)
    return choices


def prepare_pivot_data(
    data: pd.DataFrame,
    feature_name_col: str,
    feature_value_col: str,
    id_columns: List[str],
) -> pd.DataFrame:
    """ピボット変換前のデータ準備を行います."""
    # 必要な列のみを抽出
    required_columns = id_columns + [feature_name_col, feature_value_col]
    prepared_data = data[required_columns].copy()

    # 欠損値の処理
    prepared_data = prepared_data.dropna(subset=[feature_name_col, feature_value_col])

    return prepared_data


def validate_transformation_result(
    original_data: pd.DataFrame,
    transformed_data: pd.DataFrame,
    transformation_type: str,
) -> bool:
    """変換結果の妥当性を検証します."""
    if transformed_data is None:
        console.print(f"[red]{transformation_type}の結果がNoneです[/red]")
        return False

    if len(transformed_data) == 0:
        console.print(f"[red]{transformation_type}の結果が空です[/red]")
        return False

    if len(transformed_data.columns) == 0:
        console.print(f"[red]{transformation_type}の結果に列がありません[/red]")
        return False

    # メモリ使用量の確認（極端に大きくなっていないか）
    original_memory = original_data.memory_usage(deep=True).sum()
    transformed_memory = transformed_data.memory_usage(deep=True).sum()

    if transformed_memory > original_memory * 10:  # 10倍以上になった場合は警告
        console.print(
            f"[yellow]警告: {transformation_type}後のメモリ使用量が大幅に増加しました "
            f"({original_memory / 1024 / 1024:.1f}MB → "
            f"{transformed_memory / 1024 / 1024:.1f}MB)[/yellow]"
        )

    return True
