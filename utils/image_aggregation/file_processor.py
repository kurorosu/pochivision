"""画像ファイルの処理（コピー/移動）を行うモジュール."""

import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from utils.image_aggregation.operations import OperationMode


class FileProcessor:
    """
    画像ファイルを処理するクラス.

    Attributes:
        output_dir: 出力先ディレクトリ
        mode: ファイル操作モード（コピーまたは移動）
    """

    def __init__(self, output_dir: Path, mode: OperationMode) -> None:
        """
        FileProcessorを初期化する.

        Args:
            output_dir: 出力先ディレクトリ
            mode: ファイル操作モード
        """
        self.output_dir = output_dir
        self.mode = mode

        # 出力ディレクトリが存在しない場合は作成
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")

    def process_files_from_folders(self, folders: List[Path]) -> int:
        """
        複数のフォルダから画像ファイルを処理する.

        Args:
            folders: 処理対象のフォルダのリスト

        Returns:
            処理されたファイルの総数
        """
        total_processed = 0
        all_image_files = []

        # まず、すべての画像ファイルを収集
        for folder in folders:
            # 画像ファイルの検索（拡張子は大文字小文字を区別しない）
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
                all_image_files.extend(list(folder.glob(ext)))
                all_image_files.extend(list(folder.glob(ext.upper())))

        print(f"Found {len(all_image_files)} image files to process")

        # プログレスバーを表示して処理
        if all_image_files:
            for img_file in tqdm(
                all_image_files, desc=f"Processing {self.mode.value}", unit="files"
            ):
                try:
                    if img_file.exists():
                        self._process_single_file(img_file)
                        total_processed += 1
                    else:
                        print(f"\nWarning: File does not exist: {img_file}")
                except Exception as e:
                    print(f"\nError processing file {img_file}: {str(e)}")
                    continue

        return total_processed

    def _process_single_file(self, source_file: Path) -> None:
        """
        単一の画像ファイルを処理する.

        Args:
            source_file: 処理対象のファイルパス
        """
        dest_file = self.output_dir / source_file.name

        # 既存ファイルの処理（名前の競合を解決）
        if dest_file.exists():
            dest_file = self._resolve_filename_conflict(source_file)

        try:
            # コピーまたは移動
            if self.mode == OperationMode.COPY:
                shutil.copy2(source_file, dest_file)
            else:  # MOVE
                shutil.move(str(source_file), str(dest_file))
        except FileNotFoundError:
            print(f"\nWarning: Source file not found: {source_file}")
        except PermissionError:
            print(f"\nWarning: Permission denied for file: {source_file}")
        except Exception as e:
            print(f"\nError during file operation: {str(e)}")
            raise

    def _resolve_filename_conflict(self, source_file: Path) -> Path:
        """
        ファイル名の競合を解決する.

        Args:
            source_file: 元のファイルパス

        Returns:
            競合しない新しい出力ファイルパス
        """
        base_name = source_file.stem
        extension = source_file.suffix
        counter = 1

        new_file = self.output_dir / f"{base_name}_{counter}{extension}"
        while new_file.exists():
            counter += 1
            new_file = self.output_dir / f"{base_name}_{counter}{extension}"

        return new_file
