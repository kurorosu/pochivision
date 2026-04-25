"""画像処理パイプラインの実行・管理を行うモジュール."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from pochivision.capturelib import LogManager
from pochivision.capturelib.config_handler import CameraConfigHandler
from pochivision.core.image_saver import ImageSaver
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import PROCESSOR_REGISTRY
from pochivision.utils.file_naming import get_file_naming_manager


class PipelineExecutor:
    """画像処理プロセッサ群を管理し, 処理を行うパイプライン実行クラス.

    Attributes:
        processors: 実行対象の画像処理プロセッサのリスト.
        output_dir: 処理結果の保存先ディレクトリ.
        mode: 実行モード. "parallel" または "pipeline".
        camera_index: このパイプラインが対応するカメラのインデックス.
        id_interval: ID値が増加する画像数の間隔.
        config_fps: 設定ファイルから取得したFPS値.
        saver: 画像保存を担当する ImageSaver インスタンス.
    """

    def __init__(
        self,
        processors: list[BaseProcessor],
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
        self.saver = ImageSaver(output_dir, camera_index)

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
        config: dict[str, Any],
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

            processors: list[BaseProcessor] = []
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

    def run(self, image: np.ndarray) -> None:
        """指定された画像に対してプロセッサを適用し, 処理結果を保存する.

        Args:
            image: 入力画像.
        """
        start_time = time.perf_counter()

        self.saver.save(image, "original")

        # エラー発生時はプロセッサ名に対してエラー情報の dict を格納するため
        # Any を許容する.
        processed_images: dict[str, Any] = {"original": image.copy()}
        if self.mode == "parallel":
            for processor in self.processors:
                try:
                    proc_start = time.perf_counter()
                    # parallel モードでは複数プロセッサが同一フレームを
                    # 共有するため, in-place 変更による相互干渉を防ぐため
                    # 各プロセッサに独立した copy を渡す.
                    result = processor.process(image.copy())
                    proc_time = time.perf_counter() - proc_start
                    self.logger.info(
                        f"Processing time ({processor.name}): {proc_time:.3f} sec"
                    )
                    processed_images[processor.name] = result
                    self.saver.save(result, processor.name)
                except Exception as e:
                    self.logger.error(
                        f"Processor '{processor.name}' failed, skipping: {e}"
                    )
                    # 失敗したプロセッサの痕跡を残して追跡可能にする.
                    processed_images[processor.name] = {
                        "error": True,
                        "processor_name": processor.name,
                        "exception": repr(e),
                    }

        elif self.mode == "pipeline":
            result = image
            pipeline_aborted = False

            for processor in self.processors:
                if pipeline_aborted:
                    # 直前のプロセッサ失敗時は後続を明示的にスキップし,
                    # 古い `result` が次のプロセッサへ渡るのを防ぐ.
                    self.logger.warning(
                        f"Skipping processor '{processor.name}' "
                        f"due to earlier pipeline failure"
                    )
                    processed_images[processor.name] = {
                        "error": True,
                        "processor_name": processor.name,
                        "exception": "skipped_due_to_earlier_failure",
                    }
                    continue

                try:
                    proc_start = time.perf_counter()

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
                    proc_time = time.perf_counter() - proc_start
                    self.logger.info(
                        f"Processing time ({processor.name}): {proc_time:.3f} sec"
                    )
                    processed_images[processor.name] = result
                except Exception as e:
                    # pipeline モードでは失敗時にパイプラインを中断する.
                    # 古い `result` を次プロセッサへ渡さないことで状態不整合を防ぐ.
                    self.logger.error(
                        f"Processor '{processor.name}' failed, "
                        f"aborting pipeline: {e}"
                    )
                    processed_images[processor.name] = {
                        "error": True,
                        "processor_name": processor.name,
                        "exception": repr(e),
                    }
                    pipeline_aborted = True

            if not pipeline_aborted:
                self.saver.save(result, "pipeline")
            else:
                self.logger.warning(
                    "Pipeline aborted due to processor failure; "
                    "final result not saved"
                )

        total_time = time.perf_counter() - start_time
        self.logger.info(f"Total processing time: {total_time:.3f} sec")
