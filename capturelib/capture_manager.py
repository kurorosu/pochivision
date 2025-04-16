import os
from pathlib import Path
from datetime import datetime


class CaptureManager(object):
    """
    保存用ディレクトリの管理を行うクラス。

    Attributes:
        base_dir (Path): 保存のベースディレクトリ。
        capture_dir (Path): 実際に画像を保存するセッション単位のサブディレクトリ。
    """

    def __init__(self, base_dir="capture"):
        """
        CaptureManager のコンストラクタ。

        Args:
            base_dir (str): ベースとなるディレクトリ名。デフォルトは 'capture'。
        """
        self.base_dir = Path(base_dir)
        self.capture_dir = self._create_capture_subdir()

    def _create_capture_subdir(self):
        """
        日付ごとにユニークなサブディレクトリを作成。

        Returns:
            Path: 作成されたディレクトリのパス。
        """
        date_str = datetime.now().strftime("%Y%m%d")
        base_path = self.base_dir / date_str
        index = 0
        final_path = base_path

        while final_path.exists():
            index += 1
            final_path = Path(f"{str(base_path)}_{index}")

        final_path.mkdir(parents=True, exist_ok=True)
        return final_path

    def get_processing_dir(self, process_name):
        """
        指定されたプロセッサ名に対応するサブディレクトリを取得または作成。

        Args:
            process_name (str): プロセッサの名前。

        Returns:
            Path: 対応する保存先ディレクトリのパス。
        """
        path = self.capture_dir / process_name
        path.mkdir(parents=True, exist_ok=True)
        return path
