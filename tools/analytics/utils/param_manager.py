"""Parameter management utilities for model configuration."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import optuna


class ParameterManager:
    """モデルパラメータの管理を行うクラス."""

    def __init__(self, config_file: Optional[str] = None):
        """
        初期化処理.

        Args:
            config_file (Optional[str]): 設定ファイルのパス。Noneの場合はデフォルトパスを使用
        """
        if config_file is None:
            # デフォルトの設定ファイルパス
            current_dir = Path(__file__).parent.parent
            config_file_path = current_dir / "model_param.json"
        else:
            config_file_path = Path(config_file)

        self.config_file = config_file_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """設定ファイルを読み込みます."""
        try:
            if not self.config_file.exists():
                raise FileNotFoundError(
                    f"設定ファイルが見つかりません: {self.config_file}"
                )

            with open(self.config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        except Exception as e:
            raise RuntimeError(f"設定ファイルの読み込みに失敗しました: {e}")

    def get_default_params(
        self, model_type: str = "xgboost"
    ) -> Dict[str, Union[int, float]]:
        """
        デフォルトパラメータを取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Dict[str, Union[int, float]]: デフォルトパラメータ

        Raises:
            KeyError: 指定されたモデルタイプが見つからない場合
        """
        if model_type not in self.config:
            raise KeyError(
                f"モデルタイプ '{model_type}' が設定ファイルに見つかりません"
            )

        return self.config[model_type]["default_params"].copy()

    def get_optuna_search_space(
        self, model_type: str = "xgboost"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optunaの探索空間設定を取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Dict[str, Dict[str, Any]]: 探索空間設定

        Raises:
            KeyError: 指定されたモデルタイプが見つからない場合
        """
        if model_type not in self.config:
            raise KeyError(
                f"モデルタイプ '{model_type}' が設定ファイルに見つかりません"
            )

        return self.config[model_type]["optuna_search_space"].copy()

    def get_optuna_config(self, model_type: str = "xgboost") -> Dict[str, Any]:
        """
        Optuna設定を取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Dict[str, Any]: Optuna設定

        Raises:
            KeyError: 指定されたモデルタイプが見つからない場合
        """
        if model_type not in self.config:
            raise KeyError(
                f"モデルタイプ '{model_type}' が設定ファイルに見つかりません"
            )

        return self.config[model_type]["optuna_config"].copy()

    def suggest_parameter(
        self, trial: optuna.Trial, param_name: str, model_type: str = "xgboost"
    ) -> Union[int, float]:
        """
        Optunaトライアルでパラメータを提案します.

        Args:
            trial (optuna.Trial): Optunaトライアル
            param_name (str): パラメータ名
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Union[int, float]: 提案されたパラメータ値

        Raises:
            KeyError: パラメータが見つからない場合
            ValueError: パラメータタイプが不正な場合
        """
        search_space = self.get_optuna_search_space(model_type)

        if param_name not in search_space:
            raise KeyError(f"パラメータ '{param_name}' が探索空間に見つかりません")

        param_config = search_space[param_name]
        param_type = param_config["type"]

        if param_type == "int":
            return trial.suggest_int(
                param_name, param_config["low"], param_config["high"]
            )
        elif param_type == "float":
            return trial.suggest_float(
                param_name, param_config["low"], param_config["high"]
            )
        else:
            raise ValueError(f"サポートされていないパラメータタイプ: {param_type}")

    def get_all_parameters_for_trial(
        self, trial: optuna.Trial, model_type: str = "xgboost"
    ) -> Dict[str, Union[int, float]]:
        """
        Optunaトライアルで全パラメータを提案します.

        Args:
            trial (optuna.Trial): Optunaトライアル
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Dict[str, Union[int, float]]: 提案された全パラメータ
        """
        search_space = self.get_optuna_search_space(model_type)
        params = {}

        for param_name in search_space.keys():
            params[param_name] = self.suggest_parameter(trial, param_name, model_type)

        return params

    def get_parameter_descriptions(self, model_type: str = "xgboost") -> Dict[str, str]:
        """
        パラメータの説明を取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            Dict[str, str]: パラメータ名と説明のマッピング
        """
        search_space = self.get_optuna_search_space(model_type)
        descriptions = {}

        for param_name, param_config in search_space.items():
            descriptions[param_name] = param_config.get("description", "説明なし")

        return descriptions

    def get_default_trials(self, model_type: str = "xgboost") -> int:
        """
        デフォルトの試行回数を取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            int: デフォルト試行回数
        """
        optuna_config = self.get_optuna_config(model_type)
        return optuna_config.get("default_trials", 100)

    def get_cv_folds(self, model_type: str = "xgboost") -> int:
        """
        クロスバリデーションのフォールド数を取得します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            int: CVフォールド数
        """
        optuna_config = self.get_optuna_config(model_type)
        return optuna_config.get("cv_folds", 5)

    def get_metadata(self) -> Dict[str, Any]:
        """
        設定ファイルのメタデータを取得します.

        Returns:
            Dict[str, Any]: メタデータ
        """
        return self.config.get("metadata", {})

    def reload_config(self) -> None:
        """設定ファイルを再読み込みします."""
        self._load_config()

    def validate_config(self, model_type: str = "xgboost") -> bool:
        """
        設定ファイルの妥当性を検証します.

        Args:
            model_type (str): モデルタイプ. Defaults to "xgboost".

        Returns:
            bool: 設定が妥当な場合True

        Raises:
            ValueError: 設定に問題がある場合
        """
        try:
            # 必要なセクションの存在確認
            if model_type not in self.config:
                raise ValueError(f"モデルタイプ '{model_type}' が見つかりません")

            model_config = self.config[model_type]
            required_sections = [
                "default_params",
                "optuna_search_space",
                "optuna_config",
            ]

            for section in required_sections:
                if section not in model_config:
                    raise ValueError(f"必要なセクション '{section}' が見つかりません")

            # 探索空間の妥当性確認
            search_space = model_config["optuna_search_space"]
            for param_name, param_config in search_space.items():
                if "type" not in param_config:
                    raise ValueError(
                        f"パラメータ '{param_name}' にtypeが指定されていません"
                    )

                param_type = param_config["type"]
                if param_type not in ["int", "float"]:
                    raise ValueError(
                        f"パラメータ '{param_name}' の型 '{param_type}' はサポートされていません"
                    )

                if "low" not in param_config or "high" not in param_config:
                    raise ValueError(
                        f"パラメータ '{param_name}' にlowまたはhighが指定されていません"
                    )

                if param_config["low"] >= param_config["high"]:
                    raise ValueError(
                        f"パラメータ '{param_name}' のlowがhigh以上になっています"
                    )

            return True

        except Exception as e:
            raise ValueError(f"設定ファイルの検証に失敗しました: {e}")
