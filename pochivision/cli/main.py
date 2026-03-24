"""pochivisionのメインエントリーポイント."""

import argparse

from pochivision.capture_runner import LivePreviewRunner
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.capture_manager import CaptureManager
from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.core import PipelineExecutor
from pochivision.exceptions.config import ConfigValidationError


def parse_arguments():
    """
    コマンドライン引数を解析する.

    Returns:
        argparse.Namespace: パース済みの引数オブジェクト.
    """
    parser = argparse.ArgumentParser(description="pochivision application")
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        help="カメラインデックス（接続するカメラのデバイス番号）",
        default=0,
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        help="使用するカメラ設定プロファイル名（config.jsonのcameras内のキー）",
        default=None,
    )
    parser.add_argument(
        "--list-profiles",
        "-l",
        action="store_true",
        help="利用可能なカメラプロファイル一覧を表示して終了",
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="設定ファイルのパス"
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="録画機能を無効にする（録画機能なしで起動）",
    )
    return parser.parse_args()


def main():
    """アプリケーションのメインエントリーポイント."""
    args = parse_arguments()

    log_manager = LogManager()
    logger = log_manager.get_logger()
    logger.info("Starting pochivision application")

    try:
        config = ConfigHandler.load(args.config)
        logger.info("Configuration loaded successfully")

        if args.list_profiles:
            logger.info("Available camera profiles:")
            for profile in config.get("cameras", {}).keys():
                profile_config = config["cameras"][profile]
                print(
                    f"Profile: {profile}, "
                    f"Resolution: {profile_config.get('width', 'default')}x"
                    f"{profile_config.get('height', 'default')}, "
                    f"FPS: {profile_config.get('fps', 'default')}"
                )
            exit(0)

    except ConfigValidationError as e:
        logger.error(str(e))
        print("設定ファイルに誤りがあります。詳細はログを確認してください。")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        exit(1)

    try:
        camera_setup = CameraSetup(
            config, log_manager, camera_index=args.camera, profile_name=args.profile
        )
        camera_setup.load_camera_config()
        cap = camera_setup.initialize_camera()

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_setup.camera_index}. Exiting.")
            exit(1)

        log_manager.log_camera_info(
            cap,
            camera_setup.camera_index,
            camera_setup.width,
            camera_setup.height,
            profile_name=camera_setup.profile_name,
        )

    except Exception as e:
        logger.error(f"Error setting up camera: {e}")
        exit(1)

    try:
        capture_manager = CaptureManager()
        log_manager.setup_file_logging(
            capture_manager.get_log_file_path(camera_index=camera_setup.camera_index)
        )
        logger.info(
            f"Capture manager initialized for camera {camera_setup.camera_index}"
        )

        log_manager.log_system_info()

        used_profile = camera_setup.profile_name
        minimal_config = {
            "cameras": {used_profile: config["cameras"][used_profile]},
            "selected_camera_index": camera_setup.camera_index,
        }
        ConfigHandler.save(
            minimal_config,
            capture_manager.get_output_dir(camera_index=camera_setup.camera_index),
        )

        pipeline = PipelineExecutor.from_config(
            config,
            capture_manager=capture_manager,
            camera_index=camera_setup.camera_index,
            profile_name=camera_setup.profile_name,
        )

        recording_manager = None
        if not args.no_recording:
            recording_config = config.get("recording", {})
            select_format = recording_config.get("select_format", "mjpg")

            recording_manager = RecordingManager(default_format=select_format)
            logger.info(f"Recording manager initialized with format: {select_format}")
        else:
            logger.info("Recording functionality disabled")

        preview_config = config.get("preview", {})
        preview_size = (
            preview_config.get("width", 1280),
            preview_config.get("height", 720),
        )

        app = LivePreviewRunner(cap, pipeline, recording_manager, preview_size)
        logger.info("Starting application main loop")
        app.run()

    except Exception as e:
        logger.error(f"Error during execution: {e}")

    finally:
        logger.info("Application shutdown complete")
