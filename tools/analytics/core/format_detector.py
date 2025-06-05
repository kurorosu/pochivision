"""Data format detection logic for CSV Analytics."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_long_format(data: pd.DataFrame) -> bool:
    """データがロング形式（縦型）かどうかを検出します."""
    columns = data.columns.tolist()

    # Long formatの典型的なパターンを検出
    has_feature_name_col = False
    has_feature_value_col = False

    for col in columns:
        col_lower = col.lower()
        if "feature" in col_lower and "name" in col_lower:
            has_feature_name_col = True
        elif "feature" in col_lower and "value" in col_lower:
            has_feature_value_col = True

    if has_feature_name_col and has_feature_value_col:
        return True

    # より一般的なパターンを検出
    for col in columns:
        col_lower = col.lower()
        if col_lower in ["feature_name", "variable_name", "metric_name"]:
            has_feature_name_col = True
        elif col_lower in ["feature_value", "value", "val"]:
            has_feature_value_col = True

    return has_feature_name_col and has_feature_value_col


def find_feature_columns(data: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """特徴量名列と値列を特定します."""
    columns = data.columns.tolist()
    feature_name_col: Optional[str] = None
    feature_value_col: Optional[str] = None

    # 明示的なパターンを検索
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

    return feature_name_col, feature_value_col


def get_id_columns(
    data: pd.DataFrame, feature_name_col: str, feature_value_col: str
) -> List[str]:
    """識別列（ID列）を取得します."""
    return [
        col for col in data.columns if col not in [feature_name_col, feature_value_col]
    ]


def analyze_transpose_necessity(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """転置の必要性を分析します."""
    rows, cols = data.shape
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    suggest_transpose = False
    reasons = []

    if rows > cols * 2:  # 行数が列数の2倍以上
        suggest_transpose = True
        reasons.append(f"行数（{rows:,}）が列数（{cols:,}）より大幅に多い")

    numeric_ratio = len(numeric_columns) / cols
    if numeric_ratio < 0.3:  # 数値列が30%未満
        suggest_transpose = True
        reasons.append(f"数値列の割合が低い（{numeric_ratio:.1%}）")

    return suggest_transpose, reasons


def validate_pivot_columns(
    data: pd.DataFrame, feature_name_col: str, feature_value_col: str
) -> bool:
    """ピボット変換に必要な列が存在するかを検証します."""
    if feature_name_col not in data.columns:
        return False

    if feature_value_col not in data.columns:
        return False

    # 特徴量名列にデータがあるかチェック
    if data[feature_name_col].isnull().all():
        return False

    # 特徴量値列にデータがあるかチェック
    if data[feature_value_col].isnull().all():
        return False

    return True


def get_unique_feature_count(data: pd.DataFrame, feature_name_col: str) -> int:
    """ユニークな特徴量数を取得します."""
    return data[feature_name_col].nunique()


def analyze_categorical_columns(data: pd.DataFrame) -> List[str]:
    """カテゴリ列を分析して取得します."""
    return data.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(data: pd.DataFrame) -> List[str]:
    """数値列を取得します."""
    return data.select_dtypes(include=[np.number]).columns.tolist()


def analyze_data_quality(data: pd.DataFrame) -> dict:
    """データ品質を分析します."""
    total_cells = len(data) * len(data.columns)
    null_cells = data.isnull().sum().sum()

    return {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "total_cells": total_cells,
        "null_cells": null_cells,
        "null_percentage": (null_cells / total_cells) * 100 if total_cells > 0 else 0,
        "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
        "numeric_columns": len(get_numeric_columns(data)),
        "categorical_columns": len(analyze_categorical_columns(data)),
    }
