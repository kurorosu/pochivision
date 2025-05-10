"""処理フォルダの検索を行うモジュール."""

from pathlib import Path
from typing import Dict, List


class ProcessorFolderFinder:
    """
    処理タイプごとのフォルダを検索するクラス.

    Attributes:
        base_dir: 検索を開始するカメラディレクトリ
    """

    def __init__(self, base_dir: Path) -> None:
        """
        ProcessorFolderFinderを初期化する.

        Args:
            base_dir: 検索を開始するカメラディレクトリ
        """
        self.base_dir = base_dir

    def find_processor_types(self) -> Dict[str, List[Path]]:
        """
        カメラフォルダ内のすべての機能（プロセッサタイプ）フォルダを検出する.

        Returns:
            プロセッサタイプをキー、そのフォルダパスのリストを値とする辞書
        """
        processor_types: Dict[str, List[Path]] = {}

        print(f"Scanning camera directory: {self.base_dir.name}")

        # 日付フォルダを検索
        date_dirs = [d for d in self.base_dir.glob("*") if d.is_dir()]
        print(f"Found {len(date_dirs)} date directories in {self.base_dir.name}")

        # 各日付フォルダ内の処理フォルダをすべて検索
        for date_dir in date_dirs:
            # 日付フォルダ内の直接のサブディレクトリを取得（機能フォルダ）
            for processor_dir in [d for d in date_dir.glob("*") if d.is_dir()]:
                processor_type = processor_dir.name

                # 辞書に追加
                if processor_type not in processor_types:
                    processor_types[processor_type] = []

                processor_types[processor_type].append(processor_dir)

        # 結果を表示
        for processor_type, folders in processor_types.items():
            print(f"Processor type '{processor_type}' found in {len(folders)} folders")

        return processor_types

    def find_by_processor_type(self, processor_type: str) -> List[Path]:
        """
        指定された処理タイプのフォルダを検索する.

        Args:
            processor_type: 検索する処理タイプ名

        Returns:
            該当する処理フォルダのパスのリスト
        """
        found_folders = []

        # 入力ディレクトリがカメラディレクトリの場合と、captureディレクトリの場合を処理
        if self.base_dir.name.startswith("camera"):
            # 直接カメラディレクトリが指定された場合
            camera_dirs = [self.base_dir]
        else:
            # captureディレクトリなどが指定された場合、カメラフォルダを検索
            camera_dirs = [d for d in self.base_dir.glob("camera*") if d.is_dir()]

        # 各カメラフォルダ内の日付フォルダを検索
        for camera_dir in camera_dirs:
            date_dirs = [d for d in camera_dir.glob("*") if d.is_dir()]

            # 各日付フォルダ内の処理フォルダを検索
            for date_dir in date_dirs:
                processor_dir = date_dir / processor_type
                if processor_dir.exists() and processor_dir.is_dir():
                    found_folders.append(processor_dir)

        return found_folders
