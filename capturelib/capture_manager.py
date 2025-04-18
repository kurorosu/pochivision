import json
from pathlib import Path
from datetime import datetime


class CaptureManager:
    """
    保存用ディレクトリの管理を行うクラス。
    """

    def __init__(self, base_dir: str = "capture", config_file: str = "config.json") -> None:
        """
        CaptureManager のコンストラクタ。

        Args:
            base_dir (str): ベースとなるディレクトリ名。デフォルトは 'capture'。
            config_file (str): 設定ファイルのパス。デフォルトは 'config.json'。
        """
        self.base_dir = Path(base_dir)
        self.config_file = config_file
        self.capture_dir = self._create_capture_subdir()

    def _create_capture_subdir(self) -> Path:
        """
        日付ごとにユニークなサブディレクトリを作成し、設定ファイルをその中にコピーする。

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
        self._copy_config_to_output(final_path)
        return final_path

    def _copy_config_to_output(self, output_dir: Path) -> None:
        """
        config.json を指定された出力ディレクトリにコピーし、タイムスタンプを更新する。

        Args:
            output_dir (Path): 出力先ディレクトリ。
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        timestamped_config_file = output_dir / f"{timestamp}_config.json"
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            config_data["timestamp"] = timestamp
            with open(timestamped_config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"config.json を保存しました: {timestamped_config_file}")
        else:
            print(f"設定ファイル {self.config_file} が見つかりませんでした。")

    def get_processing_dir(self, process_name: str) -> Path:
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
