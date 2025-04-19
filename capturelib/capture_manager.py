from datetime import datetime
from pathlib import Path


class CaptureManager:
    """
    出力ディレクトリの管理のみを担当。
    """

    def __init__(self, base_dir: str = "capture") -> None:
        self.base_dir = Path(base_dir)
        self.capture_dir = self._create_capture_subdir()

    def _create_capture_subdir(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        base_path = self.base_dir / date_str
        index = 0
        final_path = base_path

        while final_path.exists():
            index += 1
            final_path = Path(f"{str(base_path)}_{index}")

        final_path.mkdir(parents=True, exist_ok=True)
        return final_path

    def get_output_dir(self) -> Path:
        return self.capture_dir

    def get_processing_dir(self, process_name: str) -> Path:
        path = self.capture_dir / process_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_log_file_path(self) -> Path:
        """
        ログファイルのパスを取得する。

        Returns:
            Path: ログファイルのパス。
        """
        return self.capture_dir / "capture.log"
