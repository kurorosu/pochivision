"""画像の保存・ディレクトリ管理を行うモジュール."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from pochivision.capturelib import LogManager
from pochivision.utils.file_naming import get_file_naming_manager


class ImageSaver:
    """画像ファイルの保存とディレクトリ管理を担当するクラス.

    スレッドセーフ: `file_naming_manager` へのアクセスおよびファイル I/O は
    内部ロックで保護されており, parallel モード等から複数スレッド同時に
    `save()` が呼ばれても競合しない.

    Attributes:
        output_dir: 処理結果の保存先ルートディレクトリ.
        camera_index: このセーバーが対応するカメラのインデックス.
    """

    def __init__(self, output_dir: Path, camera_index: int = 0) -> None:
        """ImageSaverのコンストラクタ.

        Args:
            output_dir: 処理結果の保存先ディレクトリ.
            camera_index: カメラのインデックス.
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.logger = LogManager().get_logger()
        # file_naming_manager と cv2.imwrite を保護するロック.
        self._lock = threading.Lock()

    def get_processing_dir(self, process_name: str) -> Path:
        """処理結果保存用のサブディレクトリを取得する.

        Args:
            process_name: 処理の名前. 例: 'grayscale', 'gaussian_blur'.

        Returns:
            処理結果保存用ディレクトリのパス.
        """
        path = self.output_dir / process_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, image: np.ndarray, processor_name: str) -> None:
        """処理された画像を保存する.

        Args:
            image: 処理済み画像.
            processor_name: 処理に使われたプロセッサの名前.
        """
        save_dir = self.get_processing_dir(processor_name)

        # file_naming_manager のカウンタ採番とファイル書き込みはスレッド間で
        # 競合し得るためロックで直列化する.
        with self._lock:
            filename, id_index, image_index = get_file_naming_manager().get_filename(
                processor_name, self.camera_index
            )
            path = save_dir / filename

            save_start = time.perf_counter()
            try:
                success = cv2.imwrite(str(path), image)
                save_time = time.perf_counter() - save_start
                if success:
                    height, width = image.shape[:2]
                    self.logger.info(
                        f"Image saved ({processor_name}): {path} "
                        f"({width}x{height}, "
                        f"id={id_index}, image={image_index}, {save_time:.3f} sec)"
                    )
                else:
                    self.logger.warning(
                        f"Failed to save image ({processor_name}): {path}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to save image ({processor_name}): {e}")
