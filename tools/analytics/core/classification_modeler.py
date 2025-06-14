"""Classification modeling functionality for CSV Analytics."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ClassificationModeler:
    """分類モデリング機能を提供するクラス."""

    def __init__(self):
        """初期化処理."""
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""

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
