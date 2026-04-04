"""画像特徴量抽出の基底クラスを定義するモジュール."""

import abc
from typing import Any

import numpy as np


class BaseFeatureExtractor(abc.ABC):
    """
    すべての画像特徴量抽出器の基底クラス.

    Attributes:
        name (str): 特徴量抽出器の識別名.
        config (dict): 特徴量抽出器固有の設定.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """
        BaseFeatureExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名.
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        self.name = name

        # デフォルト設定とユーザー設定をマージ
        default_config = self.get_default_config()
        user_config = config or {}
        self.config = {**default_config, **user_config}

    @abc.abstractmethod
    def extract(self, image: np.ndarray) -> dict[str, float | int]:
        """
        画像から特徴量を抽出する.各特徴量抽出器でオーバーライドして実装する.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            dict[str, float | int]: 抽出された特徴量の辞書.
                キーは特徴量名、値は特徴量の値.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_default_config() -> dict[str, Any]:
        """
        特徴量抽出器のデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す.

        Returns:
            list[str]: 特徴量名のリスト.
        """
        pass
