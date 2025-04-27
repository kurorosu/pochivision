"""
キャプチャ出力ディレクトリの管理を行うモジュール.

各カメラごとに日付・サフィックス付きのディレクトリを作成し、画像やログの保存先を管理します。
"""

from datetime import datetime
from pathlib import Path
from typing import Dict


class CaptureManager:
    """
    キャプチャ出力ディレクトリを管理するクラス.

    各カメラの出力ディレクトリは以下の構造で作成されます：
    capture/
        camera{camera_index}/
            YYYYMMDD_{suffix}/  # suffixは0から始まり、同じ日付のディレクトリが存在する場合にインクリメント
                image.bmp
                gaussian_blur/
                grayscale/

    Attributes:
        base_dir (Path): キャプチャ出力のベースディレクトリ
        camera_dirs (Dict[int, Path]): カメラインデックスごとの出力ディレクトリ
    """

    def __init__(self, base_dir: str = "capture") -> None:
        """
        CaptureManagerを初期化します.

        Args:
            base_dir (str): キャプチャ出力のベースディレクトリ. デフォルトは"capture"
        """
        self.base_dir = Path(base_dir)
        self.camera_dirs: Dict[int, Path] = {}

    def _get_next_suffix(self, camera_dir: Path, date_str: str) -> int:
        """
        指定された日付の次のサフィックス番号を取得します.

        同じ日付のディレクトリが存在する場合、最大のサフィックス番号に1を加えた値を返します.

        Args:
            camera_dir (Path): カメラのディレクトリパス.
            date_str (str): 日付文字列(YYYYMMDD形式).

        Returns:
            int: 次のサフィックス番号.
        """
        if not camera_dir.exists():
            return 0

        max_suffix = -1
        for dir_path in camera_dir.iterdir():
            if not dir_path.is_dir():
                continue

            # YYYYMMDD_N 形式のディレクトリを検索
            if dir_path.name.startswith(date_str + "_"):
                try:
                    suffix = int(dir_path.name.split("_")[-1])
                    max_suffix = max(max_suffix, suffix)
                except ValueError:
                    continue

        return max_suffix + 1

    def _create_camera_date_dir(self, camera_index: int) -> Path:
        """
        カメラごとの日付ベースのディレクトリを作成します.

        同じ日付のディレクトリが存在する場合、サフィックスをインクリメントします.

        Args:
            camera_index (int): カメラのインデックス.

        Returns:
            Path: 作成されたディレクトリのパス. 例：capture/camera{camera_index}/YYYYMMDD_{suffix}/
        """
        date_str = datetime.now().strftime("%Y%m%d")
        camera_dir = self.base_dir / f"camera{camera_index}"

        # カメラディレクトリが存在しない場合は作成
        if not camera_dir.exists():
            camera_dir.mkdir(parents=True, exist_ok=True)

        # 次のサフィックス番号を取得
        suffix = self._get_next_suffix(camera_dir, date_str)

        # 日付ディレクトリ作成
        final_path = camera_dir / f"{date_str}_{suffix}"
        final_path.mkdir(parents=True, exist_ok=True)
        return final_path

    def get_output_dir(self, camera_index: int = 0) -> Path:
        """
        カメラごとの出力ディレクトリを取得します.

        ディレクトリが存在しない場合は新規作成します.

        Args:
            camera_index (int): カメラのインデックス. デフォルトは0.

        Returns:
            Path: 出力ディレクトリのパス. 例：capture/camera{camera_index}/YYYYMMDD_{suffix}/
        """
        if camera_index not in self.camera_dirs:
            self.camera_dirs[camera_index] = self._create_camera_date_dir(camera_index)
        return self.camera_dirs[camera_index]

    def get_processing_dir(self, process_name: str, camera_index: int = 0) -> Path:
        """
        画像処理結果保存用のディレクトリを取得します.

        ディレクトリが存在しない場合は新規作成します.

        Args:
            process_name (str): 処理の名前. 例：'grayscale', 'gaussian_blur'.
            camera_index (int): カメラのインデックス. デフォルトは0.

        Returns:
            Path: 処理結果保存用ディレクトリのパス.
        """
        base_dir = self.get_output_dir(camera_index)
        path = base_dir / process_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_log_file_path(self, camera_index: int = 0) -> Path:
        """
        カメラごとのログファイルパスを取得します.

        Args:
            camera_index (int): カメラのインデックス. デフォルトは0.

        Returns:
            Path: capture/camera{camera_index}/YYYYMMDD_{suffix}/capture.log
        """
        return self.get_output_dir(camera_index) / "capture.log"
