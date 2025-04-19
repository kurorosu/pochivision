import cv2
from typing import Dict, Any

from capturelib.config_handler import CameraConfigError
from capturelib.log_manager import LogManager


class CameraSetup:
    """
    カメラの設定と初期化を担当するクラス。
    """

    def __init__(self, config: Dict[str, Any], log_manager: LogManager):
        """
        CameraSetupクラスのコンストラクタ。

        Args:
            config (Dict[str, Any]): アプリケーション設定。
            log_manager (LogManager): ロギングマネージャー。
        """
        self.config = config
        self.log_manager = log_manager
        self.logger = log_manager.get_logger()
        self.camera_index = 0
        self.width = 0
        self.height = 0
        self.fps = 30
        self.backend = None

    def load_camera_config(self) -> None:
        """
        設定からカメラ設定を読み込む。
        """
        try:
            # 選択されたカメラインデックスを取得
            self.camera_index = self.config.get("selected_camera_index", 0)

            # カメラの設定をcameras辞書から取得
            camera_index_str = str(self.camera_index)
            camera_config = self.config.get(
                "cameras", {}).get(camera_index_str, {})

            # カメラの解像度設定を取得
            self.width = camera_config.get("width", 640)
            self.height = camera_config.get("height", 480)
            self.fps = camera_config.get("fps", 30)
            self.backend = camera_config.get("backend", None)

            self.logger.info(f"Camera configuration loaded: Camera Index={self.camera_index}, "
                             f"Width={self.width}, Height={self.height}, "
                             f"FPS={self.fps}, Backend={self.backend}")
        except Exception as e:
            self.logger.error(f"Error loading camera config: {e}")
            raise CameraConfigError(
                f"Failed to load camera configuration: {e}")

    def initialize_camera(self) -> cv2.VideoCapture:
        """
        カメラを初期化する。

        Returns:
            cv2.VideoCapture: 初期化されたカメラオブジェクト。
        """
        self.logger.info(f"Initializing camera {self.camera_index}")

        # バックエンドが指定されている場合は適用
        if self.backend:
            backend_constant = getattr(cv2, f"CAP_{self.backend}", None)
            if backend_constant is not None:
                cap = cv2.VideoCapture(self.camera_index, backend_constant)
            else:
                self.logger.warning(
                    f"Unknown backend: {self.backend}, using default")
                cap = cv2.VideoCapture(self.camera_index)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_index}")
            return cap

        # 解像度の設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # FPSの設定
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 実際に設定された解像度を取得
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

        if actual_width != self.width or actual_height != self.height:
            self.logger.warning(
                f"Requested resolution ({self.width}x{self.height}) not set. "
                f"Actual resolution: {actual_width}x{actual_height}"
            )

            # 実際の解像度で更新
            self.width = actual_width
            self.height = actual_height

        if actual_fps != self.fps:
            self.logger.warning(
                f"Requested FPS ({self.fps}) not set. "
                f"Actual FPS: {actual_fps}"
            )
            self.fps = actual_fps

        return cap
