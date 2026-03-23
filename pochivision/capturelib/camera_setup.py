"""カメラの設定・初期化を行うためのユーティリティモジュール."""

from typing import Any, Dict

import cv2

from pochivision.capturelib.config_handler import CameraConfigError, LogManager


class CameraSetup:
    """カメラの設定と初期化を担当するクラス."""

    def __init__(
        self,
        config: Dict[str, Any],
        log_manager: LogManager,
        camera_index: int,
        profile_name: str,
    ):
        """
        CameraSetupクラスのコンストラクタ.

        Args:
            config (Dict[str, Any]): アプリケーション設定.
            log_manager (LogManager): ロギングマネージャー.
            camera_index (Optional[int]): 使用するカメラのインデックス. None の場合は設定ファイルの値を使用.
            profile_name (Optional[str]): 使用するカメラプロファイル名. None の場合はカメラインデックスに対応する設定を使用.
        """
        self.config = config
        self.log_manager = log_manager
        self.logger = log_manager.get_logger()
        self.camera_index = camera_index
        self.profile_name = profile_name
        self.width = 0
        self.height = 0
        self.fps = 30
        self.backend = None

    def load_camera_config(self) -> None:
        """設定からカメラ設定を読み込む. CLIで指定されたカメラインデックスとプロファイル名を優先する."""
        try:
            # カメラインデックスが指定されていない場合は設定ファイルから取得
            # argsでデフォルト0にしたから不要かも
            if self.camera_index is None:
                self.camera_index = self.config.get("selected_camera_index", 0)

            # カメラプロファイルが指定されていない場合は常にプロファイル "0" を使用
            if self.profile_name is None:
                self.profile_name = "0"

            # プロファイルキーを設定
            profile_key = self.profile_name

            # カメラの設定をcameras辞書から取得
            camera_config = self.config.get("cameras", {}).get(profile_key, {})

            if not camera_config:
                self.logger.warning(
                    f"Camera profile '{profile_key}' "
                    f"not found in config, using default settings"
                )

            # カメラの解像度設定を取得
            self.width = camera_config.get("width", 640)
            self.height = camera_config.get("height", 480)
            self.fps = camera_config.get("fps", 30)
            self.backend = camera_config.get("backend", None)

            self.logger.info(
                f"Camera configuration loaded: Camera Index={self.camera_index}, "
                f"Profile={profile_key}, Width={self.width}, Height={self.height}, "
                f"FPS={self.fps}, Backend={self.backend}"
            )
        except Exception as e:
            self.logger.error(f"Error loading camera config: {e}")
            raise CameraConfigError(f"Failed to load camera configuration: {e}")

    def initialize_camera(self) -> cv2.VideoCapture:
        """
        カメラを初期化する.

        Returns:
            cv2.VideoCapture: 初期化されたカメラオブジェクト.
        """
        self.logger.info(f"Initializing camera {self.camera_index}")

        # バックエンドが指定されている場合は適用
        if self.backend:
            backend_constant = getattr(cv2, f"CAP_{self.backend}", None)
            if backend_constant is not None:
                cap = cv2.VideoCapture(self.camera_index, backend_constant)
            else:
                self.logger.warning(f"Unknown backend: {self.backend}, using default")
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
                f"Requested FPS ({self.fps}) not set. " f"Actual FPS: {actual_fps}"
            )
            self.fps = actual_fps

        return cap
