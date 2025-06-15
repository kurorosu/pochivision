"""Classification modeling functionality for CSV Analytics."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ClassificationModeler:
    """分類モデリング機能を提供するクラス."""

    def __init__(self):
        """初期化処理."""
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.clean_data: Optional[pd.DataFrame] = None

    def train_model(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Union[float, int]]:
        """
        XGBoostを使用して分類モデルを訓練します.

        Args:
            data (pd.DataFrame): 訓練データ
            feature_names (List[str]): 特徴量名のリスト
            target_column (str): 目的変数列名
            test_size (float, optional): テストデータの割合. Defaults to 0.2.
            random_state (int, optional): 乱数シード. Defaults to 42.

        Returns:
            Dict[str, float]: 精度スコア（train_accuracy, test_accuracy）

        Raises:
            ValueError: 特徴量または目的変数が見つからない場合
        """
        # 特徴量が存在するかチェック
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(
                f"以下の特徴量がデータに見つかりません: {missing_features}"
            )

        if target_column not in data.columns:
            raise ValueError(f"目的変数 '{target_column}' がデータに見つかりません")

        # 欠損値を含む行を除去
        clean_data = data[feature_names + [target_column]].dropna()
        if len(clean_data) == 0:
            raise ValueError("欠損値を除去した結果、データが空になりました")

        # クリーンデータを保存（PCA用）
        self.clean_data = clean_data.copy()

        # 特徴量と目的変数を分離
        X = clean_data[feature_names]
        y = clean_data[target_column]

        # ラベルエンコーディング（文字列の目的変数の場合）
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # クラス数を確認してobjectiveを決定
        n_classes = len(self.label_encoder.classes_)
        if n_classes < 2:
            raise ValueError(
                f"分類には最低2つのクラスが必要です。現在のクラス数: {n_classes}"
            )

        # クラス数に応じてobjectiveとeval_metricを設定
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        # 訓練・テストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        # XGBoostモデルを訓練
        self.model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )

        self.model.fit(X_train, y_train)

        # 予測と精度計算
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # インスタンス変数に保存
        self.feature_names = feature_names.copy()
        self.target_name = target_column

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_samples": len(clean_data),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(self.label_encoder.classes_),
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        特徴量重要度を取得します.

        Returns:
            Optional[Dict[str, float]]: 特徴量名と重要度のマッピング
        """
        if self.model is None or not self.feature_names:
            return None

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))

    def save_model_results(
        self,
        data_file_path: str,
        accuracy_scores: Dict[str, Union[float, int]],
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        モデリング結果をCSVファイルに保存します.

        Args:
            data_file_path (str): 元のCSVファイルのパス
            accuracy_scores (Dict[str, float]): 精度スコア
            feature_importance (Optional[Dict[str, float]], optional): 特徴量重要度

        Returns:
            str: 保存されたCSVファイルのパス
        """
        # 保存先ディレクトリを作成
        data_dir = Path(data_file_path).parent
        models_dir = self._create_models_directory(data_dir)

        # PCA散布図を生成（特徴量が3つ以上の場合のみ）
        pca_plot_path = None
        if len(self.feature_names) >= 3:
            pca_plot_path = self.generate_pca_scatter_plot(models_dir)

        # 結果をDataFrameに整理
        results_data: List[Dict[str, Union[str, int, float]]] = []

        # 基本情報
        results_data.append(
            {
                "項目": "目的変数",
                "値": self.target_name,
                "詳細": "",
            }
        )

        results_data.append(
            {
                "項目": "特徴量数",
                "値": len(self.feature_names),
                "詳細": ", ".join(self.feature_names),
            }
        )

        results_data.append(
            {
                "項目": "訓練精度",
                "値": f"{accuracy_scores['train_accuracy']:.4f}",
                "詳細": f"({accuracy_scores['train_accuracy']*100:.2f}%)",
            }
        )

        results_data.append(
            {
                "項目": "テスト精度",
                "値": f"{accuracy_scores['test_accuracy']:.4f}",
                "詳細": f"({accuracy_scores['test_accuracy']*100:.2f}%)",
            }
        )

        train_count = accuracy_scores["n_train"]
        test_count = accuracy_scores["n_test"]
        results_data.append(
            {
                "項目": "データ数",
                "値": accuracy_scores["n_samples"],
                "詳細": f"訓練: {train_count}, テスト: {test_count}",
            }
        )

        results_data.append(
            {
                "項目": "クラス数",
                "値": accuracy_scores["n_classes"],
                "詳細": "",
            }
        )

        # PCA情報を追加
        if pca_plot_path:
            results_data.append(
                {
                    "項目": "PCA散布図",
                    "値": "生成済み",
                    "詳細": f"ファイル: {Path(pca_plot_path).name}",
                }
            )

            if self.pca is not None:
                total_variance = np.sum(self.pca.explained_variance_ratio_)
                results_data.append(
                    {
                        "項目": "PCA寄与率",
                        "値": f"{total_variance:.3f}",
                        "詳細": f"PC1: {self.pca.explained_variance_ratio_[0]:.3f}, "
                        f"PC2: {self.pca.explained_variance_ratio_[1]:.3f}",
                    }
                )
        else:
            if len(self.feature_names) < 3:
                results_data.append(
                    {
                        "項目": "PCA散布図",
                        "値": "未生成",
                        "詳細": "特徴量数が3未満のため実行されませんでした",
                    }
                )

        # 特徴量重要度を追加（上位5位まで）
        if feature_importance:
            sorted_importance = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
                results_data.append(
                    {
                        "項目": f"重要度{i}位",
                        "値": f"{importance:.4f}",
                        "詳細": feature,
                    }
                )

        # DataFrameに変換
        results_df = pd.DataFrame(results_data)

        # CSVファイルに保存
        output_file = models_dir / "modeling_results.csv"
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        return str(output_file)

    def _create_models_directory(self, data_dir: Path) -> Path:
        """
        modelsディレクトリを作成します.

        Args:
            data_dir (Path): データファイルがあるディレクトリ

        Returns:
            Path: 作成されたmodelsディレクトリのパス
        """
        # models{index}の形式でディレクトリを作成
        index = 1
        while True:
            models_dir = data_dir / f"models{index}"
            if not models_dir.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
                return models_dir
            index += 1

    def get_class_names(self) -> List[str]:
        """
        クラス名のリストを取得します.

        Returns:
            List[str]: クラス名のリスト
        """
        if self.label_encoder is None:
            return []
        return self.label_encoder.classes_.tolist()

    def generate_pca_scatter_plot(self, models_dir: Path) -> Optional[str]:
        """
        PCAを用いた2D散布図を生成し、models{index}フォルダに保存します.

        Args:
            models_dir (Path): 保存先のmodelsディレクトリ

        Returns:
            Optional[str]: 保存されたPNG画像ファイルのパス、失敗時はNone
        """
        if (
            self.clean_data is None
            or self.label_encoder is None
            or len(self.feature_names) < 3
        ):
            return None

        try:
            # 特徴量データを取得
            X = self.clean_data[self.feature_names]
            y = self.clean_data[self.target_name]

            # データの標準化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # PCAを実行（2次元に削減）
            self.pca = PCA(n_components=2, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)

            # クラスラベルをエンコード
            y_encoded = self.label_encoder.transform(y)
            class_names = self.label_encoder.classes_

            # 散布図を作成
            plt.figure(figsize=(12, 10))
            plt.style.use("default")

            # クラス数に応じたカラーパレットを取得
            n_classes = len(class_names)
            if n_classes <= 10:
                colors = sns.color_palette("tab10", n_classes)
            else:
                colors = sns.color_palette("hsl", n_classes)

            # 各クラスの散布図を描画
            for i, class_name in enumerate(class_names):
                mask = y_encoded == i
                class_data = X_pca[mask]
                plt.scatter(
                    class_data[:, 0],
                    class_data[:, 1],
                    alpha=0.7,
                    label=f"{class_name} (n={np.sum(mask)})",
                    color=colors[i],
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )

            # グラフの装飾
            plt.title(
                f"PCA Scatter Plot: {self.target_name} Classification",
                fontsize=16,
                fontweight="bold",
            )
            plt.xlabel(
                f"PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)",
                fontsize=12,
            )
            plt.ylabel(
                f"PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)",
                fontsize=12,
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)

            # 統計情報をテキストで追加
            total_variance = np.sum(self.pca.explained_variance_ratio_)
            stats_text = f"Features used: {len(self.feature_names)}\n"
            stats_text += f"Total samples: {len(X_pca)}\n"
            stats_text += f"Total variance explained: {total_variance:.1%}\n"
            stats_text += f"PC1 variance: {self.pca.explained_variance_ratio_[0]:.1%}\n"
            stats_text += f"PC2 variance: {self.pca.explained_variance_ratio_[1]:.1%}"

            plt.figtext(
                0.98,
                0.02,
                stats_text,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="bottom",
            )

            # レイアウトを調整
            plt.tight_layout()

            # ファイルを保存
            output_file = models_dir / "pca_scatter_plot.png"
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

            return str(output_file)

        except Exception as e:
            print(f"PCA散布図の生成中にエラーが発生しました: {e}")
            return None
