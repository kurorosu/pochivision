# cap_app.py
import cv2

from capture_runner import LivePreviewRunner
from core import PipelineExecutor
from capturelib.capture_manager import CaptureManager
from capturelib.log_manager import LogManager
from capturelib.config_handler import ConfigHandler
from capturelib.camera_setup import CameraSetup


# ロギングの初期化
log_manager = LogManager()
logger = log_manager.get_logger()
logger.info("Starting Vision Capture Core application")

# システム情報のログ出力
log_manager.log_system_info()

# 設定ファイルの読み込み
try:
    config = ConfigHandler.load("config.json")
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    exit(1)

# CameraSetupを使用したカメラの設定
try:
    # カメラセットアップの初期化
    camera_setup = CameraSetup(config, log_manager)
    camera_setup.load_camera_config()
    cap = camera_setup.initialize_camera()

    # カメラが正常に開かれたか確認
    if not cap.isOpened():
        logger.error(
            f"Failed to open camera {camera_setup.camera_index}. Exiting.")
        exit(1)

    # camera_setupの解像度情報を使用してカメラ情報をログに記録
    log_manager.log_camera_info(
        cap,
        camera_setup.camera_index,
        camera_setup.width,
        camera_setup.height
    )

except Exception as e:
    logger.error(f"Error setting up camera: {e}")
    exit(1)

# キャプチャマネージャとパイプラインの作成
try:
    # キャプチャマネージャを初期化し、ファイルへのロギングを設定
    capture_manager = CaptureManager()
    log_manager.setup_file_logging(capture_manager.get_log_file_path())
    logger.info("Capture manager initialized")

    # 設定をキャプチャディレクトリに保存
    ConfigHandler.save(config, capture_manager.get_output_dir())

    # パイプラインエグゼキューターの初期化
    pipeline = PipelineExecutor.from_config(
        config, capture_manager=capture_manager)

    # アプリケーションの作成と実行
    app = LivePreviewRunner(cap, pipeline)
    logger.info("Starting application main loop")
    app.run()

except Exception as e:
    logger.error(f"Error during execution: {e}")

finally:
    # リソースのクリーンアップ
    logger.info("Releasing resources")
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application shutdown complete")
