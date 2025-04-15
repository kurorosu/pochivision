import time
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import numpy as np


class BaseProcessor(ABC):
    """
    画像処理のベースクラス。すべての画像処理クラスはこのクラスを継承して実装する。
    """

    def __init__(self, name: str, save_dir: Path, config: dict) -> None:
        """
        コンストラクタ。処理名、保存先ディレクトリ、設定を初期化。

        Parameters:
            name (str): 処理名
            save_dir (Path): 画像保存先ディレクトリ
            config (dict): 処理に必要な設定情報
        """
        self.name = name
        self.save_dir = save_dir
        self.config = config

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        サブクラスで実装されるべき画像処理メソッド。

        Parameters:
            image (np.ndarray): 処理する画像

        Returns:
            np.ndarray: 処理された画像
        """
        pass

    def save(self, image: np.ndarray) -> Path:
        """
        処理した画像を保存する。

        Parameters:
            image (np.ndarray): 保存する画像

        Returns:
            Path: 保存した画像のファイルパス
        """
        filename = f"snapshot_{self.name}_{int(time.time())}.bmp"
        save_path = self.save_dir / filename
        cv2.imwrite(str(save_path), image)
        print(f"[{self.name}] 保存しました: {save_path}")
        return save_path
