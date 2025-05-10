"""画像集約の中心となるクラスを提供するモジュール."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils.image_aggregation.folder_finder import ProcessorFolderFinder
from utils.image_aggregation.operations import OperationMode


class ImageAggregator:
    """
    画像集約処理のファサードとなるクラス.

    カメラフォルダ内のすべての日付フォルダから、処理タイプごとに画像を集約します。

    Attributes:
        input_dir: 入力カメラディレクトリパス
        output_base_dir: 出力先ベースディレクトリパス
        mode: ファイル操作モード
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        processor_type=None,  # 互換性のために残す
        mode: OperationMode = OperationMode.COPY,
        process_all=True,  # 互換性のために残す
    ) -> None:
        """
        ImageAggregatorを初期化する.

        Args:
            input_dir: カメラディレクトリのパス
            output_dir: 集約画像の出力先ベースフォルダパス
            processor_type: 未使用（互換性のために残す）
            mode: ファイル操作モード
            process_all: 未使用（互換性のために残す）
        """
        self.input_dir = Path(input_dir)

        # 出力先は常にimage_aggregatedとする
        self.output_base_dir = Path("image_aggregated")

        self.mode = mode

    def _create_dated_output_dir(self) -> Path:
        """
        日付とインデックスを含む出力ディレクトリを作成する.

        形式: image_aggregated/YYYYMMDD_INDEX

        Returns:
            作成された出力ディレクトリのパス
        """
        # 出力ベースディレクトリが存在しない場合は作成
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 現在の日付を取得
        today = datetime.now().strftime("%Y%m%d")

        # 同じ日付のフォルダが既に存在する場合、インデックスを増やす
        index = 0
        while True:
            output_dir = self.output_base_dir / f"{today}_{index}"
            if not output_dir.exists():
                break
            index += 1

        # 出力ディレクトリを作成
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        return output_dir

    def _collect_all_image_files(
        self, processor_types: Dict[str, List[Path]]
    ) -> Tuple[List[Tuple[Path, Path]], Path]:
        """
        すべての画像ファイルとその出力先を収集する.

        Args:
            processor_types: プロセッサタイプとそのフォルダリストの辞書

        Returns:
            入力ファイルパスと出力ディレクトリのタプルのリストと出力ディレクトリパスのタプル
        """
        all_files = []

        # 画像ファイルの拡張子リスト
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

        # 出力ディレクトリを作成
        output_dir = self._create_dated_output_dir()

        # 各処理タイプのフォルダから画像ファイルを収集
        for processor_type, folders in processor_types.items():
            if not folders:
                continue

            # 処理タイプごとのサブフォルダを作成
            type_output_dir = output_dir / processor_type
            type_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {type_output_dir}")

            # 各フォルダから画像ファイルを収集
            for folder in folders:
                for ext in image_extensions:
                    # 小文字と大文字の両方の拡張子を検索
                    for img_file in folder.glob(ext):
                        all_files.append((img_file, type_output_dir))
                    for img_file in folder.glob(ext.upper()):
                        all_files.append((img_file, type_output_dir))

        return all_files, output_dir

    def _process_single_file(self, source_file: Path, output_dir: Path) -> bool:
        """
        単一の画像ファイルを処理する.

        Args:
            source_file: 処理対象のファイルパス
            output_dir: 出力先ディレクトリ

        Returns:
            処理が成功したかどうか
        """
        dest_file = output_dir / source_file.name

        # 既存ファイルの処理（名前の競合を解決）
        if dest_file.exists():
            base_name = source_file.stem
            extension = source_file.suffix
            counter = 1

            # 重複しない名前を見つける
            while dest_file.exists():
                dest_file = output_dir / f"{base_name}_{counter}{extension}"
                counter += 1

        try:
            # コピーまたは移動
            if self.mode == OperationMode.COPY:
                shutil.copy2(source_file, dest_file)
            else:  # MOVE
                shutil.move(str(source_file), str(dest_file))
            return True
        except Exception as e:
            print(f"\nError processing file {source_file}: {str(e)}")
            return False

    def aggregate(self) -> int:
        """
        画像集約処理を実行する.

        カメラディレクトリ内のすべての日付フォルダから、処理タイプごとに
        専用のサブフォルダを作成して画像を集約します。

        Returns:
            集約された画像の総数
        """
        folder_finder = ProcessorFolderFinder(self.input_dir)

        # すべての処理タイプを検出
        processor_types = folder_finder.find_processor_types()
        if not processor_types:
            print("Warning: No processor folders found in the directory structure")
            return 0

        # すべての画像ファイルを収集
        all_files, output_dir = self._collect_all_image_files(processor_types)

        print(f"Found {len(all_files)} total image files to process")

        # 全体で1本のプログレスバーで処理
        total_processed = 0
        if all_files:
            for source_file, dest_dir in tqdm(
                all_files, desc=f"Processing {self.mode.value}", unit="files"
            ):
                if source_file.exists() and self._process_single_file(
                    source_file, dest_dir
                ):
                    total_processed += 1

        # 処理タイプごとの処理数をカウント
        type_counts = {}
        for processor_type in processor_types.keys():
            type_dir = output_dir / processor_type
            if type_dir.exists():
                type_counts[processor_type] = len(list(type_dir.glob("*.*")))
                print(
                    f"Processed {type_counts[processor_type]} "
                    f"files from {processor_type} folders"
                )

        print(
            f"Completed {self.mode.value} operation. "
            f"Processed {total_processed} images total in {output_dir}."
        )
        return total_processed
