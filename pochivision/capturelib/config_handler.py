"""
設定ファイルの読み込み・保存・バリデーションを行うモジュール.

カメラ設定やアプリ全体の設定の管理、例外クラスも含みます。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from pochivision.capturelib.schema import ConfigModel
from pochivision.exceptions.config import (
    CameraConfigError,
    ConfigLoadError,
    ConfigValidationError,
)

from .log_manager import LogManager


class ConfigHandler:
    """
    config.json の読み込み・保存だけを担当.

    単一責任の原則に従い、設定ファイルの入出力のみを扱う.
    """

    _logger = LogManager().get_logger()

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む.

        Args:
            path (str): 設定ファイルのパス。

        Returns:
            dict: 読み込んだ設定の辞書。

        Raises:
            ConfigLoadError: 設定ファイルが見つからない、またはJSONデコードに失敗した場合。
        """
        try:
            logger = LogManager().get_logger()
            logger.debug(f"Loading configuration file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                config: Dict[str, Any] = json.load(f)
            logger.info(f"Configuration file loaded successfully: {path}")
            # バリデーション追加
            try:
                ConfigModel(**config)
            except ValidationError as e:
                logger.error(f"Config validation failed: {e}")
                raise ConfigValidationError(f"設定ファイルのバリデーションに失敗: {e}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {path}")
            raise ConfigLoadError(f"Configuration file not found: {path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON configuration: {e}")
            raise ConfigLoadError(f"Failed to decode JSON configuration: {e}")

    @staticmethod
    def save(config: Dict[str, Any], output_dir: Path) -> None:
        """
        設定をファイルに保存する.

        Args:
            config (dict): 保存する設定辞書。
            output_dir (Path): 保存先ディレクトリのパス。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{timestamp}_config.json"

        config_copy = config.copy()
        config_copy["timestamp"] = timestamp

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config_copy, f, indent=4)

        ConfigHandler._logger.info(f"Configuration saved: {filename}")


class CameraConfigHandler:
    """
    カメラ設定の処理を担当.

    単一責任の原則に従い、カメラ関連の設定のみを扱う.
    """

    _logger = LogManager().get_logger()

    @staticmethod
    def get_camera_processors(config: Dict[str, Any], profile_name: str) -> Tuple:
        """
        指定されたカメラプロファイルのプロセッサ設定を取得する.

        Args:
            config (dict): 設定辞書。
            profile_name (str): カメラプロファイル名。

        Returns:
            tuple: (プロセッサリスト, プロセッサ設定辞書, モード)のタプル。

        Raises:
            CameraConfigError: カメラプロファイルにプロセッサ設定がない場合。
        """
        if "cameras" not in config:
            raise CameraConfigError("No camera configurations found in config")

        if profile_name not in config["cameras"]:
            raise CameraConfigError(
                f"No configuration found for camera profile: {profile_name}"
            )

        camera_config = config["cameras"][profile_name]

        # プロセッサリストの取得
        if "processors" not in camera_config:
            raise CameraConfigError(
                f"No processors defined for camera profile: {profile_name}"
            )

        processors = camera_config["processors"]
        if not processors:
            raise CameraConfigError(
                f"Empty processors list for camera profile: {profile_name}"
            )

        # プロセッサごとの設定を取得
        processor_configs = {}
        for processor_name in processors:
            if processor_name in camera_config:
                processor_configs[processor_name] = camera_config[processor_name]
            else:
                processor_configs[processor_name] = {}

        # 実行モードの取得（カメラプロファイル固有のモードがなければデフォルトは"parallel"）
        mode = camera_config.get("mode", "parallel")

        return processors, processor_configs, mode
