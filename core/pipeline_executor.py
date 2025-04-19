import time

import cv2
import numpy as np

from capturelib import CaptureManager, LogManager
from processors import BaseProcessor
from processors.registry import PROCESSOR_REGISTRY


class PipelineExecutor:
    """
    画像処理プロセッサ群を管理し、処理と保存を行うパイプライン実行クラス。

    Attributes:
        processors (list): 実行対象の画像処理プロセッサのリスト。
        capture_manager (CaptureManager): 処理結果の保存先ディレクトリを管理するオブジェクト。
        mode (str): 実行モード。"parallel" または "pipeline"。
    """

    def __init__(self, processors: list[BaseProcessor], capture_manager: CaptureManager, mode: str = "parallel") -> None:
        """
        PipelineExecutor のコンストラクタ。

        Args:
            processors (list): 画像処理プロセッサのインスタンス群。
            capture_manager (CaptureManager): 保存先ディレクトリ管理。
            mode (str): 実行モード（"parallel" または "pipeline"）。デフォルトは "parallel"。
        """
        self.processors = processors
        self.capture_manager = capture_manager
        self.mode = mode
        self.logger = LogManager().get_logger()

        # プロセッサ情報をログに記録
        self.logger.info(f"Pipeline mode: {mode}")
        self.logger.info(
            f"Processors: {', '.join([p.name for p in processors])}")

    @classmethod
    def from_config(cls, config: dict, capture_manager: CaptureManager) -> "PipelineExecutor":
        """
        設定ファイル（辞書）からインスタンスを生成。

        Args:
            config (dict): JSON等から読み込んだ設定辞書。
            capture_manager (CaptureManager): 保存用のディレクトリ管理インスタンス。

        Returns:
            PipelineExecutor: 構成済みの PipelineExecutor インスタンス。
        """
        processors: list[BaseProcessor] = []
        for name in config["processors"]:
            processor_cls = PROCESSOR_REGISTRY[name]
            processor = processor_cls(name=name, config=config.get(name, {}))
            processors.append(processor)
        return cls(
            processors=processors,
            capture_manager=capture_manager,
            mode=config.get("mode", "parallel")
        )

    def run(self, image: np.ndarray) -> None:
        """
        指定された画像に対してプロセッサを適用し、処理結果を保存する。

        Args:
            image (np.ndarray): 入力画像。
        """
        start_time = time.time()

        if self.mode == "parallel":
            for processor in self.processors:
                proc_start = time.time()
                result = processor.process(image)
                proc_time = time.time() - proc_start
                self.logger.info(
                    f"Processing time ({processor.name}): {proc_time:.3f} sec")
                self._save(result, processor.name)

        elif self.mode == "pipeline":
            result = image
            for processor in self.processors:
                proc_start = time.time()
                result = processor.process(result)
                proc_time = time.time() - proc_start
                self.logger.info(
                    f"Processing time ({processor.name}): {proc_time:.3f} sec")
            self._save(result, self.processors[-1].name)

        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.3f} sec")

    def _save(self, image: np.ndarray, processor_name: str) -> None:
        """
        処理された画像を保存する内部メソッド。

        Args:
            image (np.ndarray): 処理済み画像。
            processor_name (str): 処理に使われたプロセッサの名前。
        """
        save_dir = self.capture_manager.get_processing_dir(processor_name)
        filename = f"snapshot_{processor_name}_{int(cv2.getTickCount())}.bmp"
        path = save_dir / filename

        save_start = time.time()
        try:
            cv2.imwrite(str(path), image)
            save_time = time.time() - save_start
            self.logger.info(
                f"Image saved ({processor_name}): {path} ({image.shape[1]}x{image.shape[0]}, {save_time:.3f} sec)")
        except Exception as e:
            self.logger.error(f"Failed to save image ({processor_name}): {e}")
