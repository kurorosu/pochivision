"""プロファイルベースの画像処理ビジネスロジッククラス."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError
from pochivision.processors.registry import get_processor
from pochivision.utils.image import (
    DEFAULT_IMAGE_EXTENSIONS,
    get_image_files,
    load_image,
)
from pochivision.workspace import OutputManager


class ProfileProcessor:
    """プロファイルベースの画像処理クラス."""

    def __init__(
        self,
        config_path: str,
        profile_name: str,
        output_manager: OutputManager | None = None,
    ) -> None:
        """ProfileProcessorのコンストラクタ.

        Args:
            config_path: config.jsonファイルのパス.
            profile_name: 使用するカメラプロファイル名.
            output_manager: 出力ディレクトリの統一管理クラス.
                None の場合はデフォルトの OutputManager を使用.

        Raises:
            ConfigLoadError: 設定ファイルの読み込みに失敗した場合.
            ConfigValidationError: 設定ファイルのバリデーションに失敗した場合.
            ValueError: プロファイルが見つからない場合.
        """
        self.logger = LogManager().get_logger()
        self.config_path = config_path
        self.profile_name = profile_name
        self.output_manager = output_manager or OutputManager()
        self.config = ConfigHandler.load(config_path)
        self.profile_config = self._get_profile_config(profile_name)
        self.processors = self._initialize_processors()

    def _get_profile_config(self, profile_name: str) -> dict[str, Any]:
        """指定されたプロファイルの設定を取得する.

        Args:
            profile_name: プロファイル名.

        Returns:
            プロファイル設定.

        Raises:
            ValueError: プロファイルが見つからない場合.
        """
        cameras = self.config.get("cameras", {})
        if profile_name not in cameras:
            available_profiles = list(cameras.keys())
            raise ValueError(
                f"プロファイル '{profile_name}' が見つかりません. "
                f"利用可能なプロファイル: {available_profiles}"
            )

        result: dict[str, Any] = cameras[profile_name]
        return result

    def _initialize_processors(self) -> list[Any]:
        """プロファイル設定に基づいてプロセッサを初期化する.

        Returns:
            初期化されたプロセッサのリスト.
        """
        processors = []
        processor_names = self.profile_config.get("processors", [])

        for processor_name in processor_names:
            try:
                processor_config = self.profile_config.get(processor_name, {})
                processor = get_processor(processor_name, processor_config)
                processors.append(processor)
                self.logger.info(f"プロセッサを初期化しました: {processor_name}")
            except Exception as e:
                self.logger.warning(
                    f"プロセッサの初期化に失敗しました ({processor_name}): {e}"
                )

        if not processors:
            self.logger.warning("使用可能なプロセッサがありません")

        return processors

    def _get_image_files(self, input_dir: Path) -> list[Path]:
        """入力ディレクトリから画像ファイルを取得する.

        Args:
            input_dir: 入力ディレクトリのパス.

        Returns:
            画像ファイルのパスリスト.

        Raises:
            FileNotFoundError: 入力ディレクトリが存在しない場合.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

        image_files = get_image_files(input_dir)

        if not image_files:
            self.logger.warning(
                f"入力ディレクトリに画像ファイルが見つかりません: {input_dir}"
            )
            self.logger.warning(f"対象拡張子: {DEFAULT_IMAGE_EXTENSIONS}")

        return image_files

    def _create_output_directory(self) -> Path:
        """出力ディレクトリを作成する.

        Returns:
            作成された出力ディレクトリのパス.
        """
        return self.output_manager.create_output_dir("processed")

    def _process_image(self, image_path: Path) -> np.ndarray | None:
        """単一画像を処理する.

        Args:
            image_path: 画像ファイルのパス.

        Returns:
            処理された画像. エラーの場合は None.
        """
        try:
            image = load_image(image_path)
            if image is None:
                self.logger.warning(f"画像の読み込みに失敗しました: {image_path}")
                return None

            mode = self.profile_config.get("mode", "pipeline")

            if mode == "pipeline":
                processed_image = image.copy()
                original_image = image.copy()

                for processor in self.processors:
                    if hasattr(processor, "set_target_image"):
                        processor.set_target_image(original_image)

                    processed_image = processor.process(processed_image)
                    if processed_image is None:
                        self.logger.warning(
                            f"プロセッサでの処理に失敗しました: {image_path}"
                        )
                        return None
                return processed_image

            elif mode == "parallel":
                results = {}
                for processor in self.processors:
                    if hasattr(processor, "set_target_image"):
                        self.logger.warning(
                            f"{processor.__class__.__name__}はパラレルモードでは使用できません"
                        )
                        continue

                    processor_name = processor.__class__.__name__.lower().replace(
                        "processor", ""
                    )
                    result = processor.process(image.copy())
                    if result is not None:
                        results[processor_name] = result

                if results:
                    last_result: np.ndarray = list(results.values())[-1]
                    return last_result
                return None

            else:
                self.logger.warning(f"不明な処理モード: {mode}")
                return None

        except Exception as e:
            self.logger.error(f"画像処理中にエラーが発生しました ({image_path}): {e}")
            return None

    def process_directory(self, input_dir: str, save_original: bool = True) -> None:
        """ディレクトリ内のすべての画像を処理する.

        Args:
            input_dir: 入力ディレクトリのパス.
            save_original: 元画像も保存するかどうか.
        """
        input_path = Path(input_dir)

        output_path = self._create_output_directory()

        image_files = self._get_image_files(input_path)

        if not image_files:
            return

        self.logger.info(f"処理開始: {len(image_files)}個の画像を処理します")
        self.logger.info(f"使用プロファイル: {self.profile_name}")
        self.logger.info(f"出力ディレクトリ: {output_path}")

        self._save_config_info(output_path)

        processed_count = 0
        failed_count = 0

        for image_path in image_files:
            self.logger.info(f"処理中: {image_path.name}")

            if save_original:
                original_dir = output_path / "original"
                original_dir.mkdir(exist_ok=True)
                original_output_path = original_dir / image_path.name
                try:
                    shutil.copy2(image_path, original_output_path)
                except Exception as e:
                    self.logger.warning(f"元画像のコピーに失敗しました: {e}")

            processed_image = self._process_image(image_path)

            if processed_image is not None:
                processed_dir = output_path / "processed"
                processed_dir.mkdir(exist_ok=True)
                processed_output_path = processed_dir / image_path.name

                success = cv2.imwrite(str(processed_output_path), processed_image)
                if success:
                    processed_count += 1
                    self.logger.info(f"保存完了: {processed_output_path}")
                else:
                    failed_count += 1
                    self.logger.warning(f"保存失敗: {processed_output_path}")
            else:
                failed_count += 1
                self.logger.warning(f"処理失敗: {image_path.name}")

        self.logger.info("処理完了:")
        self.logger.info(f"  成功: {processed_count}個")
        self.logger.info(f"  失敗: {failed_count}個")
        self.logger.info(f"  出力先: {output_path}")

    def _save_config_info(self, output_path: Path) -> None:
        """使用した設定情報を保存する.

        Args:
            output_path: 出力ディレクトリのパス.
        """
        try:
            profile_info = {
                "profile_name": self.profile_name,
                "profile_config": self.profile_config,
                "timestamp": datetime.now().isoformat(),
                "config_file": self.config_path,
            }

            info_file = output_path / "profile_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(profile_info, f, indent=2, ensure_ascii=False)

            self.logger.info(f"プロファイル情報を保存しました: {info_file}")

        except Exception as e:
            self.logger.warning(f"プロファイル情報の保存に失敗しました: {e}")

    def list_available_profiles(self) -> None:
        """利用可能なプロファイルを一覧表示する."""
        cameras = self.config.get("cameras", {})
        self.logger.info("利用可能なプロファイル:")
        for profile_name, profile_config in cameras.items():
            label = profile_config.get("label", "No Label")
            processors = profile_config.get("processors", [])
            mode = profile_config.get("mode", "pipeline")
            self.logger.info(f"  {profile_name}: {label}")
            self.logger.info(f"    モード: {mode}")
            self.logger.info(f"    プロセッサ: {', '.join(processors)}")
