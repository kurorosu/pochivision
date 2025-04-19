import abc

import numpy as np


class BaseProcessor:
    """
    すべての画像処理プロセッサの基底クラス。

    Attributes:
        name (str): プロセッサの識別名。
        config (dict): プロセッサ固有の設定。
    """

    def __init__(self, name: str, config: dict = None) -> None:
        """
        BaseProcessor のコンストラクタ。

        Args:
            name (str): プロセッサ名。
            config (dict, optional): 設定パラメータ。デフォルトは空の辞書。
        """
        self.name = name
        self.config = config or {}

    @abc.abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像処理の本体。各プロセッサでオーバーライドして実装する。

        Args:
            image (np.ndarray): 入力画像。

        Returns:
            np.ndarray: 処理後の画像。
        """
        pass
