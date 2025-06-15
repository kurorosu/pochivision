"""Classification modeling functionality for CSV Analytics."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from analytics.utils.param_manager import ParameterManager
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ClassificationModeler:
    """分類モデリング機能を提供するクラス."""

    def __init__(self, param_config_file: Optional[str] = None):
        """
        初期化処理.

        Args:
            param_config_file (Optional[str]): パラメータ設定ファイルのパス
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.holdout_model: Optional[xgb.XGBClassifier] = None  # ホールドアウトモデル
        self.cv_model: Optional[xgb.XGBClassifier] = None  # CVモデル
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Union[int, float]]] = None
        self.study: Optional[optuna.Study] = None
        self.param_manager = ParameterManager(param_config_file)

    def train_model(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        use_optuna: bool = False,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        XGBoostを使用して分類モデルを訓練します.

        Args:
            data (pd.DataFrame): 訓練データ
            feature_names (List[str]): 特徴量名のリスト
            target_column (str): 目的変数列名
            test_size (float, optional): テストデータの割合. Defaults to 0.2.
            random_state (int, optional): 乱数シード. Defaults to 42.
            use_optuna (bool, optional): Optunaを使用するかどうか. Defaults to False.
            n_trials (Optional[int], optional): Optunaの試行回数. Noneの場合は設定ファイルのデフォルト値を使用.

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

        # ホールドアウト用のデータ分割を最初に実行（独立したランダムシード）
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        # Optunaを使用する場合はハイパーパラメータチューニングを実行
        # （全データを使用し、ホールドアウト分割とは独立）
        if use_optuna:
            # 試行回数が指定されていない場合は設定ファイルのデフォルト値を使用
            if n_trials is None:
                n_trials = self.param_manager.get_default_trials()

            print(
                f"Optunaを使用してハイパーパラメータチューニングを開始します（試行回数: {n_trials}）..."
            )
            # Optuna用に独立したランダムシードを使用
            optuna_random_state = random_state + 1000  # 独立したシード
            best_params = self._optimize_hyperparameters(
                X, y_encoded, objective, eval_metric, n_trials, optuna_random_state
            )
            self.best_params = best_params
            print(f"最適なパラメータ: {best_params}")
        else:
            # 設定ファイルからデフォルトパラメータを取得
            best_params = self.param_manager.get_default_params()

        # ホールドアウトモデルは常にデフォルトパラメータで訓練
        default_params = self.param_manager.get_default_params()
        self.model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            **default_params,
        )

        self.model.fit(X_train, y_train)

        # 予測と精度計算
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # CVモデルも作成して保存用に準備（Optuna使用時のみ）
        cv_model = None
        cv_accuracy = None
        if use_optuna and self.study is not None:
            # CV用の最適パラメータでモデルを作成（全データで訓練）
            cv_model = xgb.XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                random_state=optuna_random_state,  # Optuna用のシードを使用
                **best_params,
            )
            cv_model.fit(X, y_encoded)  # エンコードされたラベルを使用

            # 全データでの5-fold CV精度を計算
            cv_folds = self.param_manager.get_cv_folds()
            cv_splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=optuna_random_state
            )
            cv_scores = cross_val_score(
                cv_model, X, y_encoded, cv=cv_splitter, scoring="accuracy", n_jobs=-1
            )
            cv_accuracy = cv_scores.mean()

        # インスタンス変数に保存（ホールドアウトモデルとCVモデル）
        self.holdout_model = self.model  # ホールドアウトモデル
        self.cv_model = cv_model  # CVモデル（Optuna使用時のみ）
        self.feature_names = feature_names.copy()
        self.target_name = target_column

        result = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_samples": len(clean_data),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": len(self.label_encoder.classes_),
        }

        # Optunaを使用した場合は追加情報を含める
        if use_optuna and self.study is not None:
            result["optuna_trials"] = len(self.study.trials)
            result["best_cv_score"] = cv_accuracy

        return result

    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        objective: str,
        eval_metric: str,
        n_trials: int,
        random_state: int,
    ) -> Dict[str, Union[int, float]]:
        """
        Optunaを使用してハイパーパラメータを最適化します.

        Args:
            X (pd.DataFrame): 全特徴量データ
            y (np.ndarray): 全目的変数データ
            objective (str): XGBoostのobjective
            eval_metric (str): XGBoostのeval_metric
            n_trials (int): 最適化の試行回数
            random_state (int): 乱数シード

        Returns:
            Dict[str, Union[int, float]]: 最適化されたハイパーパラメータ
        """

        def objective_function(trial):
            """Optunaの目的関数."""
            # ParameterManagerを使用してパラメータを提案
            params = self.param_manager.get_all_parameters_for_trial(trial)

            # XGBoostモデルを作成
            model = xgb.XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                random_state=random_state,
                **params,
            )

            # 設定ファイルからCVフォールド数を取得
            cv_folds = self.param_manager.get_cv_folds()

            # クロスバリデーションで性能を評価（ランダムシード固定）
            cv_splitter = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
            cv_scores = cross_val_score(
                model, X, y, cv=cv_splitter, scoring="accuracy", n_jobs=-1
            )

            return cv_scores.mean()

        # Optunaスタディを作成して最適化を実行
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        self.study.optimize(
            objective_function, n_trials=n_trials, show_progress_bar=True
        )

        return self.study.best_params

    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """
        Optunaの最適化履歴を取得します.

        Returns:
            Optional[pd.DataFrame]: 最適化履歴のDataFrame
        """
        if self.study is None:
            return None

        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            # パラメータを追加
            if trial.params:
                trial_data.update(trial.params)
            trials_data.append(trial_data)

        return pd.DataFrame(trials_data)

    def get_feature_importance(
        self, model_type: str = "both"
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        特徴量重要度を取得します.

        Args:
            model_type (str): 取得するモデルタイプ ("holdout", "cv", "both")

        Returns:
            Optional[Dict[str, Dict[str, float]]]: モデルタイプ別の特徴量重要度
        """
        if not self.feature_names:
            return None

        result = {}

        # ホールドアウトモデルの重要度
        if model_type in ["holdout", "both"] and self.holdout_model is not None:
            importance_scores = self.holdout_model.feature_importances_
            result["holdout"] = dict(zip(self.feature_names, importance_scores))

        # CVモデルの重要度
        if model_type in ["cv", "both"] and self.cv_model is not None:
            importance_scores = self.cv_model.feature_importances_
            result["cv"] = dict(zip(self.feature_names, importance_scores))

        return result if result else None

    def save_model_results(
        self,
        data_file_path: str,
        accuracy_scores: Dict[str, Union[float, int]],
        feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """
        モデリング結果をCSVファイルに保存します.

        Args:
            data_file_path (str): 元のCSVファイルのパス
            accuracy_scores (Dict[str, float]): 精度スコア
            feature_importance (Optional[Dict[str, Dict[str, float]]], optional): 特徴量重要度

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
                "項目": "ホールドアウト訓練精度",
                "値": f"{accuracy_scores['train_accuracy']:.4f}",
                "詳細": f"({accuracy_scores['train_accuracy']*100:.2f}%) - 訓練データ80%での精度",
            }
        )

        results_data.append(
            {
                "項目": "ホールドアウトテスト精度",
                "値": f"{accuracy_scores['test_accuracy']:.4f}",
                "詳細": f"({accuracy_scores['test_accuracy']*100:.2f}%) - テストデータ20%での精度",
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

        # Optuna情報を追加
        if "optuna_trials" in accuracy_scores:
            results_data.append(
                {
                    "項目": "Optuna使用",
                    "値": "有効",
                    "詳細": f"試行回数: {accuracy_scores['optuna_trials']}",
                }
            )

            if "best_cv_score" in accuracy_scores:
                results_data.append(
                    {
                        "項目": "CV法最適平均精度",
                        "値": f"{accuracy_scores['best_cv_score']:.4f}",
                        "詳細": f"({accuracy_scores['best_cv_score']*100:.2f}%) "
                        f"- 全データでの5-fold CV平均精度",
                    }
                )

            # 最適パラメータを追加
            if self.best_params:
                for param_name, param_value in self.best_params.items():
                    if isinstance(param_value, float):
                        value_str = f"{param_value:.4f}"
                    else:
                        value_str = str(param_value)
                    results_data.append(
                        {
                            "項目": f"最適_{param_name}",
                            "値": value_str,
                            "詳細": "Optunaによる最適化結果",
                        }
                    )
        else:
            results_data.append(
                {
                    "項目": "Optuna使用",
                    "値": "無効",
                    "詳細": "デフォルトパラメータを使用",
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
            # ホールドアウトモデルの重要度
            if "holdout" in feature_importance:
                holdout_importance = feature_importance["holdout"]
                sorted_importance = sorted(
                    holdout_importance.items(), key=lambda x: x[1], reverse=True
                )
                for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
                    results_data.append(
                        {
                            "項目": f"ホールドアウト重要度{i}位",
                            "値": f"{importance:.4f}",
                            "詳細": feature,
                        }
                    )

            # CVモデルの重要度
            if "cv" in feature_importance:
                cv_importance = feature_importance["cv"]
                sorted_importance = sorted(
                    cv_importance.items(), key=lambda x: x[1], reverse=True
                )
                for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
                    results_data.append(
                        {
                            "項目": f"CV重要度{i}位",
                            "値": f"{importance:.4f}",
                            "詳細": feature,
                        }
                    )

        # モデル保存情報を追加
        model_info = "ホールドアウトモデル、LabelEncoder"
        if self.cv_model is not None:
            model_info += "、CVモデル"

        results_data.append(
            {
                "項目": "モデル保存",
                "値": "完了",
                "詳細": model_info + "を保存",
            }
        )

        # DataFrameに変換
        results_df = pd.DataFrame(results_data)

        # CSVファイルに保存
        output_file = models_dir / "modeling_results.csv"
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        # XGBoostモデルを保存
        model_files = self._save_model_files(models_dir)
        print("モデルファイルを保存しました:")
        for model_file in model_files:
            print(f"  - {Path(model_file).name}")

        # Optuna最適化履歴を保存
        if self.study is not None:
            optimization_history = self.get_optimization_history()
            if optimization_history is not None:
                history_file = models_dir / "optuna_optimization_history.csv"
                optimization_history.to_csv(
                    history_file, index=False, encoding="utf-8-sig"
                )
                print(f"Optuna最適化履歴を保存しました: {history_file}")

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

    def _save_model_files(self, models_dir: Path) -> List[str]:
        """
        XGBoostモデルを保存します.

        Args:
            models_dir (Path): 保存先のmodelsディレクトリ

        Returns:
            List[str]: 保存されたモデルファイルのパスのリスト
        """
        model_files = []

        # ホールドアウトモデルを保存
        holdout_model_file = models_dir / "xgboost_holdout_model.pkl"
        with open(holdout_model_file, "wb") as f:
            pickle.dump(self.holdout_model, f)
        model_files.append(str(holdout_model_file))

        # CVモデルを保存（存在する場合）
        if self.cv_model is not None:
            cv_model_file = models_dir / "xgboost_cv_model.pkl"
            with open(cv_model_file, "wb") as f:
                pickle.dump(self.cv_model, f)
            model_files.append(str(cv_model_file))

        # ラベルエンコーダを保存
        label_encoder_file = models_dir / "label_encoder.pkl"
        with open(label_encoder_file, "wb") as f:
            pickle.dump(self.label_encoder, f)
        model_files.append(str(label_encoder_file))

        return model_files

    def load_model_files(self, models_dir: Path, model_type: str = "holdout") -> bool:
        """
        保存されたモデルファイルを読み込みます.

        Args:
            models_dir (Path): モデルファイルがあるディレクトリ
            model_type (str): 読み込むモデルタイプ ("holdout" または "cv")

        Returns:
            bool: 読み込みが成功した場合True

        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
            Exception: 読み込みに失敗した場合
        """
        try:
            # モデルファイルを選択
            if model_type == "cv":
                model_file = models_dir / "xgboost_cv_model.pkl"
                if not model_file.exists():
                    raise FileNotFoundError(
                        f"CVモデルファイルが見つかりません: {model_file}"
                    )
            else:
                model_file = models_dir / "xgboost_holdout_model.pkl"
                if not model_file.exists():
                    raise FileNotFoundError(
                        f"ホールドアウトモデルファイルが見つかりません: {model_file}"
                    )

            with open(model_file, "rb") as f:
                self.model = pickle.load(f)

            # ラベルエンコーダを読み込み
            label_encoder_file = models_dir / "label_encoder.pkl"
            if not label_encoder_file.exists():
                raise FileNotFoundError(
                    f"LabelEncoderファイルが見つかりません: {label_encoder_file}"
                )

            with open(label_encoder_file, "rb") as f:
                self.label_encoder = pickle.load(f)

            print("モデルファイルを読み込みました:")
            print(f"  - {model_file.name} ({model_type}モデル)")
            print(f"  - {label_encoder_file.name}")

            return True

        except Exception as e:
            print(f"モデルファイルの読み込みに失敗しました: {e}")
            return False

    def predict_new_data(self, new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        新しいデータに対して予測を行います.

        Args:
            new_data (pd.DataFrame): 予測対象のデータ

        Returns:
            Optional[pd.DataFrame]: 予測結果（クラス名と確率）

        Raises:
            ValueError: モデルが読み込まれていない場合
        """
        if self.model is None or self.label_encoder is None:
            raise ValueError(
                "モデルが読み込まれていません。load_model_files()を先に実行してください。"
            )

        if not self.feature_names:
            raise ValueError("特徴量名が設定されていません。")

        try:
            # 特徴量を抽出
            X_new = new_data[self.feature_names]

            # 予測実行
            predictions = self.model.predict(X_new)
            probabilities = self.model.predict_proba(X_new)

            # クラス名に変換
            predicted_classes = self.label_encoder.inverse_transform(predictions)

            # 結果をDataFrameに整理
            result_df = pd.DataFrame(
                {
                    "predicted_class": predicted_classes,
                }
            )

            # 各クラスの確率を追加
            class_names = self.label_encoder.classes_
            for i, class_name in enumerate(class_names):
                result_df[f"probability_{class_name}"] = probabilities[:, i]

            return result_df

        except Exception as e:
            print(f"予測中にエラーが発生しました: {e}")
            return None
