"""画像処理パイプラインの実行・管理を行うモジュール."""

import time
from typing import Any, Dict, List

import cv2
import numpy as np

from capturelib import CaptureManager, LogManager
from capturelib.config_handler import CameraConfigHandler
from processors import BaseProcessor
from processors.registry import PROCESSOR_REGISTRY
from utils.file_naming import get_file_naming_manager


class PipelineExecutor:
    """
    画像処理プロセッサ群を管理し、処理と保存を行うパイプライン実行クラス.

    Attributes:
        processors (list): 実行対象の画像処理プロセッサのリスト.
        capture_manager (CaptureManager): 処理結果の保存先ディレクトリを管理するオブジェクト.
        mode (str): 実行モード. "parallel" または "pipeline".
        camera_index (int): このパイプラインが対応するカメラのインデックス.
        id_interval (int): ID値が増加する画像数の間隔.
    """

    def __init__(
        self,
        processors: List[BaseProcessor],
        capture_manager: CaptureManager,
        mode: str = "parallel",
        camera_index: int = 0,
        id_interval: int = 1,
    ) -> None:
        """
        PipelineExecutorのコンストラクタ.

        Args:
            processors (list): 画像処理プロセッサのインスタンス群.
            capture_manager (CaptureManager): 保存先ディレクトリ管理.
            mode (str): 実行モード（"parallel" または "pipeline"）. デフォルトは "parallel".
            camera_index (int): このパイプラインが対応するカメラのインデックス.
            id_interval (int): ID値が増加する画像数の間隔 デフォルトは1.
        """
        self.processors = processors
        self.capture_manager = capture_manager
        self.mode = mode
        self.camera_index = camera_index
        self.id_interval = id_interval
        self.logger = LogManager().get_logger()

        # ID間隔を設定
        file_naming_manager = get_file_naming_manager()
        # original用の間隔設定
        file_naming_manager.set_id_interval("original", camera_index, id_interval)
        # pipeline用の間隔設定（パイプラインモードの場合）
        if mode == "pipeline":
            file_naming_manager.set_id_interval("pipeline", camera_index, id_interval)
        # 各プロセッサ用の間隔設定（parallelモードの場合）
        if mode == "parallel":
            for processor in processors:
                file_naming_manager.set_id_interval(
                    processor.name, camera_index, id_interval
                )

        # プロセッサ情報をログに記録
        self.logger.info(f"Pipeline mode: {mode}")
        self.logger.info(f"Processors: {', '.join([p.name for p in processors])}")
        self.logger.info(f"ID interval: {id_interval}")

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        capture_manager: CaptureManager,
        camera_index: int = 0,
        profile_name: str = "0",
    ) -> "PipelineExecutor":
        """
        設定ファイル（辞書）からインスタンスを生成.

        カメラプロファイルごとの画像処理設定を使用します.

        Args:
            config (dict): JSON等から読み込んだ設定辞書.
            capture_manager (CaptureManager): 保存用のディレクトリ管理インスタンス.
            camera_index (int): このパイプラインが対応するカメラのインデックス.
            profile_name (str): 使用するカメラプロファイル名.

        Returns:
            PipelineExecutor: 構成済みの PipelineExecutor インスタンス.

        Raises:
            Exception: カメラプロファイルのプロセッサ設定が取得できない場合.
        """
        try:
            # カメラプロファイルからプロセッサ設定を取得
            processor_names, processor_configs, mode = (
                CameraConfigHandler.get_camera_processors(config, profile_name)
            )

            # カメラごとのid_intervalを参照
            camera_config = config.get("cameras", {}).get(profile_name, {})
            id_interval = camera_config.get("id_interval", config.get("id_interval", 1))

            label = camera_config.get("label", "NA_NA")
            file_naming_manager = get_file_naming_manager()
            file_naming_manager.set_label(camera_index, label)

            # プロセッサインスタンスの生成
            processors: List[BaseProcessor] = []
            for name in processor_names:
                if name not in PROCESSOR_REGISTRY:
                    raise ValueError(f"Processor '{name}' is not registered")

                processor_cls = PROCESSOR_REGISTRY[name]
                processor = processor_cls(
                    name=name, config=processor_configs.get(name, {})
                )
                processors.append(processor)

            # インスタンス生成
            return cls(
                processors=processors,
                capture_manager=capture_manager,
                mode=mode,
                camera_index=camera_index,
                id_interval=id_interval,
            )
        except Exception as e:
            logger = LogManager().get_logger()
            logger.error(f"Failed to create pipeline from config: {e}")
            raise

    def run(self, image: np.ndarray) -> None:
        """
        指定された画像に対してプロセッサを適用し、処理結果を保存する.

        Args:
            image (np.ndarray): 入力画像.
        """
        start_time = time.time()

        # オリジナル画像を保存
        original_dir = self.capture_manager.get_processing_dir(
            "original", self.camera_index
        )
        filename, id_index, image_index = get_file_naming_manager().get_filename(
            "original", self.camera_index
        )
        original_path = original_dir / filename
        try:
            cv2.imwrite(str(original_path), image)
            self.logger.info(
                f"Original image saved: {original_path} "
                f"({image.shape[1]}x{image.shape[0]}, "
                f"id={id_index}, image={image_index})"
            )
        except Exception as e:
            self.logger.error(f"Failed to save original image: {e}")

        if self.mode == "parallel":
            for processor in self.processors:
                proc_start = time.time()
                result = processor.process(image)
                proc_time = time.time() - proc_start
                self.logger.info(
                    f"Processing time ({processor.name}): {proc_time:.3f} sec"
                )
                self._save(result, processor.name)

        elif self.mode == "pipeline":
            result = image
            for processor in self.processors:
                proc_start = time.time()
                result = processor.process(result)
                proc_time = time.time() - proc_start
                self.logger.info(
                    f"Processing time ({processor.name}): {proc_time:.3f} sec"
                )
            self._save(result, "pipeline")

        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.3f} sec")

    def _save(self, image: np.ndarray, processor_name: str) -> None:
        """
        処理された画像を保存する内部メソッド.

        Args:
            image (np.ndarray): 処理済み画像.
            processor_name (str): 処理に使われたプロセッサの名前.
        """
        # パイプラインモード時は"pipeline"ディレクトリに保存
        save_dir_name = processor_name
        save_dir = self.capture_manager.get_processing_dir(
            save_dir_name, self.camera_index
        )

        # 新しいファイル命名規則を使用
        filename, id_index, image_index = get_file_naming_manager().get_filename(
            save_dir_name, self.camera_index
        )
        path = save_dir / filename

        save_start = time.time()
        try:
            cv2.imwrite(str(path), image)
            save_time = time.time() - save_start
            self.logger.info(
                f"Image saved ({save_dir_name}): {path} "
                f"({image.shape[1]}x{image.shape[0]}, "
                f"id={id_index}, image={image_index}, {save_time:.3f} sec)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save image ({save_dir_name}): {e}")
