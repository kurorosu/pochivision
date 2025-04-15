import time
from pathlib import Path

import cv2
import numpy as np


class CaptureManager:
    """
    キャプチャ処理の管理を行い、保存用のディレクトリを生成・管理します。
    """

    def __init__(self, base_dir: str = "capture") -> None:
        """
        キャプチャ用のベースディレクトリを指定して、セッションディレクトリを生成します。

        Parameters:
            base_dir (str): 保存先のベースディレクトリのパス。デフォルトは 'capture'
        """
        self.base_dir = Path(base_dir)
        self.session_dir = self._create_session_directory()

    def _create_session_directory(self) -> Path:
        """
        セッション用のディレクトリを作成します。日付ごとにディレクトリを作成します。
        もし同じ名前のディレクトリがすでに存在する場合、インデックスを付けてユニークにします。

        Returns:
            Path: 作成したセッション用のディレクトリのパス
        """
        timestamp = time.strftime("%Y%m%d")
        session_path = self.base_dir / timestamp

        # 同じ名前のディレクトリがあればインデックスを追加
        index = 1
        while session_path.exists():
            session_path = self.base_dir / f"{timestamp}_{index}"
            index += 1

        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def get_processing_dir(self, process_name: str) -> Path:
        """
        指定された処理名用のディレクトリを返し、なければ作成します。

        Parameters:
            process_name (str): 処理名（例: 'grayscale', 'blur' など）

        Returns:
            Path: 指定された処理名のディレクトリのパス
        """
        process_path = self.session_dir / process_name
        process_path.mkdir(parents=True, exist_ok=True)
        return process_path

    def save_image(self, image: np.ndarray, process_name: str) -> Path:
        """
        処理した画像を保存します。

        Parameters:
            image (np.ndarray): 保存する画像
            process_name (str): 画像保存先ディレクトリを決めるための処理名

        Returns:
            Path: 保存した画像のファイルパス
        """
        save_dir = self.get_processing_dir(process_name)
        filename = f"snapshot_{int(time.time())}.bmp"
        save_path = save_dir / filename
        cv2.imwrite(str(save_path), image)
        print(f"[{process_name}] 保存しました: {save_path}")
        return save_path
