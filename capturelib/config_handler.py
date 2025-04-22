import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from .log_manager import LogManager


class ConfigLoadError(Exception):
    """
    設定ファイルの読み込みエラー用カスタム例外。
    """
    pass


class CameraConfigError(Exception):
    """
    カメラ設定に関するエラー用カスタム例外。
    """
    pass


class ConfigHandler:
    """
    config.json の読み込み・保存だけを担当。
    単一責任の原則に従い、設定ファイルの入出力のみを扱う。
    """
    _logger = LogManager().get_logger()

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む。

        Args:
            path (str): 設定ファイルのパス。

        Returns:
            dict: 読み込んだ設定の辞書。

        Raises:
            ConfigLoadError: 設定ファイルが見つからない、またはJSONデコードに失敗した場合。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigLoadError(f"Configuration file not found: {path}")
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Failed to decode JSON configuration: {e}")

    @staticmethod
    def save(config: Dict[str, Any], output_dir: Path) -> None:
        """
        設定をファイルに保存する。

        Args:
            config (dict): 保存する設定辞書。
            output_dir (Path): 保存先ディレクトリのパス。
        """
        timestamp = datetime.now().strftime("%Y-%m%d-%H%M-%S")
        filename = output_dir / f"{timestamp}_config.json"

        config_copy = config.copy()
        config_copy["timestamp"] = timestamp

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config_copy, f, indent=4)

        ConfigHandler._logger.info(f"Configuration saved: {filename}")


class CameraConfigHandler:
    """
    カメラ設定の処理を担当。
    単一責任の原則に従い、カメラ関連の設定のみを扱う。
    """
    _logger = LogManager().get_logger()

    @staticmethod
    def get_camera_config(config: Dict[str, Any], camera_index: int = None) -> Dict[str, Any]:
        """
        指定されたカメラインデックスの設定を取得する。
        カメラインデックスが指定されていない場合はselected_camera_indexの設定を使用。

        Args:
            config (dict): 設定辞書。
            camera_index (int, optional): カメラインデックス。Noneの場合はselected_camera_indexを使用。

        Returns:
            dict: カメラ設定の辞書。

        Raises:
            CameraConfigError: カメラ設定がない場合、または指定されたカメラインデックスの設定がない場合。
        """
        if "cameras" not in config:
            raise CameraConfigError("No camera configurations found in config")

        if camera_index is None:
            if "selected_camera_index" in config:
                camera_index = config["selected_camera_index"]
            else:
                raise CameraConfigError(
                    "No selected camera index specified in config")

        camera_id_str = str(camera_index)
        if camera_id_str not in config["cameras"]:
            raise CameraConfigError(
                f"No configuration found for camera index: {camera_index}")

        return config["cameras"][camera_id_str]

    @staticmethod
    def get_camera_processors(config: Dict[str, Any], profile_name: str) -> Tuple:
        """
        指定されたカメラプロファイルのプロセッサ設定を取得する。

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
                f"No configuration found for camera profile: {profile_name}")

        camera_config = config["cameras"][profile_name]

        # プロセッサリストの取得
        if "processors" not in camera_config:
            raise CameraConfigError(
                f"No processors defined for camera profile: {profile_name}")

        processors = camera_config["processors"]
        if not processors:
            raise CameraConfigError(
                f"Empty processors list for camera profile: {profile_name}")

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

    @staticmethod
    def get_all_camera_indices(config: Dict[str, Any]) -> List[int]:
        """
        設定ファイルに定義されているすべてのカメラインデックスを取得する。

        Args:
            config (dict): 設定辞書。

        Returns:
            List[int]: カメラインデックスのリスト。

        Raises:
            CameraConfigError: カメラ設定が設定ファイルに存在しない場合。
        """
        if "cameras" not in config:
            raise CameraConfigError("No camera configurations found in config")

        # 文字列のカメラインデックスを整数に変換して返す
        return [int(camera_id) for camera_id in config["cameras"].keys()]

    @staticmethod
    def get_selected_camera_index(config: Dict[str, Any]) -> int:
        """
        選択されたカメラインデックスを取得する。

        Args:
            config (dict): 設定辞書。

        Returns:
            int: 選択されたカメラインデックス。

        Raises:
            CameraConfigError: 選択されたカメラインデックスの指定がない場合。
        """
        if "selected_camera_index" in config:
            return config["selected_camera_index"]

        # カメラが定義されていれば、最初のカメラを選択
        if "cameras" in config and config["cameras"]:
            return int(list(config["cameras"].keys())[0])

        raise CameraConfigError("No selected camera index specified in config")
