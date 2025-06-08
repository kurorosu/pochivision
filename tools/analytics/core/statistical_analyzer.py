"""Statistical analysis utilities for feature ranking."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def calculate_jensen_shannon_divergence(
    data: pd.DataFrame, feature_col: str, class_col: str
) -> float:
    """
    指定された特徴量とクラス列間のJensen-Shannon Divergenceを計算します.

    Args:
        data (pd.DataFrame): データフレーム
        feature_col (str): 特徴量列名
        class_col (str): クラス列名

    Returns:
        float: 最大Jensen-Shannon Divergence値（0-1の範囲）
    """
    try:
        # 欠損値を除去
        clean_data = data[[feature_col, class_col]].dropna()

        if len(clean_data) == 0:
            return 0.0

        # クラスごとのデータを取得
        classes = clean_data[class_col].unique()

        if len(classes) < 2:
            return 0.0

        # 各クラスのヒストグラムを作成
        feature_data = clean_data[feature_col]
        min_val, max_val = feature_data.min(), feature_data.max()

        # ビン数を決定（データ数に応じて調整）
        n_bins = min(50, max(10, int(np.sqrt(len(feature_data)))))
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # クラスごとのヒストグラムを計算
        histograms = {}
        for class_name in classes:
            class_data = clean_data[clean_data[class_col] == class_name][feature_col]
            if len(class_data) > 0:
                hist, _ = np.histogram(class_data, bins=bins, density=True)
                # 正規化（確率分布にする）
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                histograms[class_name] = hist

        # 全ペアのJensen-Shannon Divergenceを計算し、最大値を返す
        max_js_divergence = 0.0
        class_list = list(histograms.keys())

        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                class1, class2 = class_list[i], class_list[j]

                # 両方のヒストグラムが有効な場合のみ計算
                if class1 in histograms and class2 in histograms:
                    hist1, hist2 = histograms[class1], histograms[class2]

                    # ゼロ除算を避けるため、小さな値を追加
                    hist1 = hist1 + 1e-10
                    hist2 = hist2 + 1e-10

                    # 再正規化
                    hist1 = hist1 / np.sum(hist1)
                    hist2 = hist2 / np.sum(hist2)

                    # Jensen-Shannon Divergenceを計算
                    js_div = jensenshannon(hist1, hist2)
                    max_js_divergence = max(max_js_divergence, js_div)

        return max_js_divergence

    except Exception:
        # 計算エラーが発生した場合は0を返す
        return 0.0


def rank_features_by_class_separation(
    data: pd.DataFrame, numeric_columns: List[str], class_col: str
) -> List[Tuple[str, float]]:
    """
    クラス分離度に基づいて特徴量をランキングします.

    Args:
        data (pd.DataFrame): データフレーム
        numeric_columns (List[str]): 数値列のリスト
        class_col (str): クラス列名

    Returns:
        List[Tuple[str, float]]: (特徴量名, JS divergence)のタプルのリスト（降順）
    """
    feature_scores = []

    for feature in numeric_columns:
        js_score = calculate_jensen_shannon_divergence(data, feature, class_col)
        feature_scores.append((feature, js_score))

    # JS divergence値で降順ソート
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    return feature_scores


def get_separation_score_description(js_score: float) -> str:
    """
    Jensen-Shannon Divergence値を分かりやすい説明に変換します.

    Args:
        js_score (float): Jensen-Shannon Divergence値

    Returns:
        str: 分離度の説明
    """
    if js_score >= 0.8:
        return "非常に高い"
    elif js_score >= 0.6:
        return "高い"
    elif js_score >= 0.4:
        return "中程度"
    elif js_score >= 0.2:
        return "低い"
    else:
        return "非常に低い"
