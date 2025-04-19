import logging
import platform
import sys
from pathlib import Path
from typing import Optional

import cv2


class LogManager:
    """
    アプリケーション全体のロギングを管理するクラス。
    シングルトンパターンを採用し、複数箇所から同じインスタンスにアクセスできるようにする。
    """
    _instance: Optional['LogManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._logger = logging.getLogger('vision-capture')
        self._initialized = True
        self._log_file_handler = None

        # デフォルトのログレベルと標準出力へのハンドラを設定
        self._logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self._logger.addHandler(console_handler)

    def setup_file_logging(self, log_file_path: Path) -> None:
        """
        ファイルへのログ出力を設定する。

        Args:
            log_file_path (Path): ログファイルのパス。
        """
        # 既存のファイルハンドラがあれば削除
        if self._log_file_handler is not None:
            self._logger.removeHandler(self._log_file_handler)

        # 新しいファイルハンドラを追加
        self._log_file_handler = logging.FileHandler(log_file_path)
        self._log_file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self._logger.addHandler(self._log_file_handler)
        self._logger.info(f"Log file configured: {log_file_path}")

    def get_logger(self) -> logging.Logger:
        """
        設定済みのロガーを取得する。

        Returns:
            logging.Logger: 設定済みのロガーインスタンス。
        """
        return self._logger

    def log_system_info(self) -> None:
        """
        システム情報をログに記録する。
        """
        self._logger.info(
            f"System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
        self._logger.info(f"Python: {sys.version.split()[0]}")
        self._logger.info(f"OpenCV: {cv2.__version__}")

    def log_camera_info(self, cap: cv2.VideoCapture, camera_id: int,
                        requested_width: int, requested_height: int) -> None:
        """
        カメラの情報をログに記録する。

        Args:
            cap (cv2.VideoCapture): 初期化済みのカメラキャプチャオブジェクト
            camera_id (int): カメラID
            requested_width (int): 要求した幅
            requested_height (int): 要求した高さ
        """
        self._logger.info(
            f"Initializing camera (ID: {camera_id}, Resolution: {requested_width}x{requested_height})")

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
            f"Camera settings - Actual resolution: {actual_width:.0f}x{actual_height:.0f}, FPS: {fps:.1f}")

        # 要求した解像度と実際の解像度が異なる場合は警告
        if abs(actual_width - requested_width) > 1 or abs(actual_height - requested_height) > 1:
            self._logger.warning(
                f"Camera resolution differs from requested: {requested_width}x{requested_height}")
