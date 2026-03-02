"""ロギング管理・システム情報記録を行うモジュール."""

import logging
import platform
import sys
from pathlib import Path
from typing import Optional

import colorlog
import cv2


class LogManager:
    """
    ロギング管理クラス（粒度・出力先分離対応版).

    シングルトンパターンを採用し、複数箇所から同じインスタンスにアクセスできるようにする.
    """

    _instance: Optional["LogManager"] = None

    def __new__(cls, *args, **kwargs):
        """LogManagerのシングルトンインスタンスを生成・返却する."""
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """LogManagerの初期化処理."""
        if self._initialized:
            return
        self._logger: logging.Logger = logging.getLogger("vcc")
        self._logger.setLevel(logging.DEBUG)
        self._initialized = True

        # コンソールハンドラ（INFO以上）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        color_formatter = colorlog.ColoredFormatter(
            "[%(asctime)s][%(log_color)s%(levelname)s%(reset)s][%(name)s]"
            "[%(log_color)s%(filename)s:%(lineno)d%(reset)s] %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(color_formatter)
        self._logger.addHandler(console_handler)

        self._file_handler = None

    def setup_file_logging(self, log_file_path: Path) -> None:
        """
        ファイルへのログ出力を設定する.

        Args:
            log_file_path (Path): ログファイルのパス。
        """
        if self._file_handler:
            self._logger.removeHandler(self._file_handler)
        self._file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        self._file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s]"
            "[%(filename)s:%(lineno)d] %(message)s"
        )
        self._file_handler.setFormatter(formatter)
        self._logger.addHandler(self._file_handler)
        self._logger.info(f"Log file configured: {log_file_path}")

    def get_logger(self) -> logging.Logger:
        """
        設定済みのロガーを取得する.

        Returns:
            logging.Logger: 設定済みのロガーインスタンス。
        """
        return self._logger

    def log_system_info(self) -> None:
        """システム情報をログに記録する."""
        self._logger.info(
            f"System: {platform.system()} {platform.release()}"
            f"({platform.architecture()[0]})"
        )
        self._logger.info(f"Python: {sys.version.split()[0]}")
        self._logger.info(f"OpenCV: {cv2.__version__}")

    def log_camera_info(
        self,
        cap: cv2.VideoCapture,
        camera_id: int,
        requested_width: int,
        requested_height: int,
        profile_name: Optional[str] = None,
    ) -> None:
        """
        カメラの情報をログに記録する.

        Args:
            cap (cv2.VideoCapture): 初期化済みのカメラキャプチャオブジェクト
            camera_id (int): カメラID
            requested_width (int): 要求した幅
            requested_height (int): 要求した高さ
            profile_name (Optional[str]): 使用しているカメラプロファイル名
        """
        profile_info = f", Profile: {profile_name}" if profile_name else ""
        self._logger.info(
            f"Initializing camera (ID: {camera_id}{profile_info}, "
            f"Resolution: {requested_width}x{requested_height})"
        )

        if not cap.isOpened():
            self._logger.error("Failed to initialize camera.")
            return

        # カメラの設定情報をログに記録
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # その他のカメラパラメータも取得可能であれば記録
        backend = cap.getBackendName()
        self._logger.info(f"Camera ready - Backend: {backend}")
        self._logger.info(
            f"Camera settings - "
            f"Actual resolution: {actual_width:.0f}x{actual_height:.0f}, "
            f"FPS: {fps:.1f}"
        )

        # 要求した解像度と実際の解像度が異なる場合は警告
        if (
            abs(actual_width - requested_width) > 1
            or abs(actual_height - requested_height) > 1
        ):
            self._logger.warning(
                f"Camera resolution differs from requested: "
                f"{requested_width}x{requested_height}"
            )
