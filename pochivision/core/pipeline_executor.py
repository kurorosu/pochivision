"""画像処理パイプラインの実行・管理を行うモジュール."""

import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from pochivision.capturelib import LogManager
from pochivision.capturelib.config_handler import CameraConfigHandler
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import PROCESSOR_REGISTRY
from pochivision.utils.file_naming import get_file_naming_manager


class PipelineExecutor:
    """画像処理プロセッサ群を管理し, 処理と保存を行うパイプライン実行クラス.

    Attributes:
        processors: 実行対象の画像処理プロセッサのリスト.
        output_dir: 処理結果の保存先ディレクトリ.
        mode: 実行モード. "parallel" または "pipeline".
        camera_index: このパイプラインが対応するカメラのインデックス.
        id_interval: ID値が増加する画像数の間隔.
        config_fps: 設定ファイルから取得したFPS値.
    """

    def __init__(
        self,
        processors: List[BaseProcessor],
        output_dir: Path,
        mode: str = "parallel",
        camera_index: int = 0,
        id_interval: int = 1,
        config_fps: float = 30.0,
    ) -> None:
        """PipelineExecutorのコンストラクタ.

        Args:
            processors: 画像処理プロセッサのインスタンス群.
            output_dir: 処理結果の保存先ディレクトリ.
            mode: 実行モード ("parallel" または "pipeline"). デフォルトは "parallel".
            camera_index: このパイプラインが対応するカメラのインデックス.
            id_interval: ID値が増加する画像数の間隔. デフォルトは1.
            config_fps: 設定ファイルから取得したFPS値. デフォルトは30.0.
        """
        valid_modes = ("parallel", "pipeline")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid pipeline mode: '{mode}'. Must be one of {valid_modes}."
            )

        self.processors = processors
        self.output_dir = output_dir
        self.mode = mode
        self.camera_index = camera_index
        self.id_interval = id_interval
        self.config_fps = config_fps
        self.logger = LogManager().get_logger()

        for processor in processors:
            if hasattr(processor, "set_pipeline_mode"):
                processor.set_pipeline_mode(mode)

        file_naming_manager = get_file_naming_manager()
        file_naming_manager.set_id_interval("original", camera_index, id_interval)
        if mode == "pipeline":
            file_naming_manager.set_id_interval("pipeline", camera_index, id_interval)
        if mode == "parallel":
            for processor in processors:
                file_naming_manager.set_id_interval(
                    processor.name, camera_index, id_interval
                )

        self.logger.info(f"Pipeline mode: {mode}")
        self.logger.info(f"Processors: {', '.join([p.name for p in processors])}")
        self.logger.info(f"ID interval: {id_interval}")

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        output_dir: Path,
        camera_index: int = 0,
        profile_name: str = "0",
    ) -> "PipelineExecutor":
        """設定ファイル (辞書) からインスタンスを生成する.

        Args:
            config: JSON等から読み込んだ設定辞書.
            output_dir: 処理結果の保存先ディレクトリ.
            camera_index: このパイプラインが対応するカメラのインデックス.
            profile_name: 使用するカメラプロファイル名.

        Returns:
            構成済みの PipelineExecutor インスタンス.
        """
        try:
            processor_names, processor_configs, mode = (
                CameraConfigHandler.get_camera_processors(config, profile_name)
            )

            camera_config = config.get("cameras", {}).get(profile_name, {})
            id_interval = camera_config.get("id_interval", config.get("id_interval", 1))
            config_fps = camera_config.get("fps", 30.0)

            label = camera_config.get("label", "NA_NA")
            file_naming_manager = get_file_naming_manager()
            file_naming_manager.set_label(camera_index, label)

            processors: List[BaseProcessor] = []
            for name in processor_names:
                if name not in PROCESSOR_REGISTRY:
                    raise ValueError(f"Processor '{name}' is not registered")

                processor_cls = PROCESSOR_REGISTRY[name]
                processor = processor_cls(
                    name=name, config=processor_configs.get(name, {})
                )
                processors.append(processor)

            return cls(
                processors=processors,
                output_dir=output_dir,
                mode=mode,
                camera_index=camera_index,
                id_interval=id_interval,
                config_fps=config_fps,
            )
        except Exception as e:
            logger = LogManager().get_logger()
            logger.error(f"Failed to create pipeline from config: {e}")
            raise

    def _get_processing_dir(self, process_name: str) -> Path:
        """処理結果保存用のサブディレクトリを取得する.

        Args:
            process_name: 処理の名前. 例: 'grayscale', 'gaussian_blur'.

        Returns:
            処理結果保存用ディレクトリのパス.
        """
        path = self.output_dir / process_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, image: np.ndarray) -> None:
        """指定された画像に対してプロセッサを適用し, 処理結果を保存する.

        Args:
            image: 入力画像.
        """
        start_time = time.time()

        original_dir = self._get_processing_dir("original")
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

        processed_images = {"original": image.copy()}
        if self.mode == "parallel":
            for processor in self.processors:
                try:
                    proc_start = time.time()
                    result = processor.process(image)
                    proc_time = time.time() - proc_start
                    self.logger.info(
                        f"Processing time ({processor.name}): {proc_time:.3f} sec"
                    )
                    processed_images[processor.name] = result
                    self._save(result, processor.name)
                except Exception as e:
                    self.logger.error(
                        f"Processor '{processor.name}' failed, skipping: {e}"
                    )

        elif self.mode == "pipeline":
            result = image

            for processor in self.processors:
                try:
                    proc_start = time.time()

                    if hasattr(processor, "target_image_name") and hasattr(
                        processor, "set_target_image"
                    ):
                        target_name = processor.target_image_name  # type: ignore
                        if target_name in processed_images:
                            processor.set_target_image(
                                processed_images[target_name]
                            )  # type: ignore
                        else:
                            self.logger.warning(
                                f"Target image '{target_name}' not found "
                                f"for {processor.name}"
                            )
                            processor.set_target_image(
                                processed_images["original"]
                            )  # type: ignore

                    result = processor.process(result)
                    proc_time = time.time() - proc_start
                    self.logger.info(
                        f"Processing time ({processor.name}): {proc_time:.3f} sec"
                    )
                    processed_images[processor.name] = result
                except Exception as e:
                    self.logger.error(
                        f"Processor '{processor.name}' failed, skipping: {e}"
                    )

            self._save(result, "pipeline")

        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.3f} sec")

    def _save(self, image: np.ndarray, processor_name: str) -> None:
        """処理された画像を保存する.

        Args:
            image: 処理済み画像.
            processor_name: 処理に使われたプロセッサの名前.
        """
        save_dir = self._get_processing_dir(processor_name)

        filename, id_index, image_index = get_file_naming_manager().get_filename(
            processor_name, self.camera_index
        )
        path = save_dir / filename

        save_start = time.time()
        try:
            cv2.imwrite(str(path), image)
            save_time = time.time() - save_start
            self.logger.info(
                f"Image saved ({processor_name}): {path} "
                f"({image.shape[1]}x{image.shape[0]}, "
                f"id={id_index}, image={image_index}, {save_time:.3f} sec)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save image ({processor_name}): {e}")
