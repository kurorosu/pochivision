"""run サブコマンド: ライブプレビュー起動."""

import logging
from pathlib import Path

import click
import cv2

from pochivision.capture_runner import LivePreviewRunner
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.constants import (
    DEFAULT_DETECT_CONFIG_PATH,
    DEFAULT_DETECTION_FPS,
    DEFAULT_INFER_CONFIG_PATH,
    DEFAULT_PREVIEW_HEIGHT,
    DEFAULT_PREVIEW_WIDTH,
)
from pochivision.core import PipelineExecutor
from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError
from pochivision.request.api.detection import DetectionClient, load_detect_config
from pochivision.request.api.inference import InferenceClient, load_infer_config
from pochivision.workspace import OutputManager


@click.command()
@click.option("--camera", "-c", type=int, default=0, help="カメラデバイスインデックス")
@click.option("--profile", "-p", type=str, default=None, help="カメラプロファイル名")
@click.option("--list-profiles", "-l", is_flag=True, help="プロファイル一覧を表示")
@click.option(
    "--config", type=str, default="config/config.json", help="設定ファイルパス"
)
@click.option("--no-recording", is_flag=True, help="録画機能を無効化")
@click.option(
    "--infer-config",
    type=str,
    default=DEFAULT_INFER_CONFIG_PATH,
    help=f"推論設定ファイルのパス (デフォルト: {DEFAULT_INFER_CONFIG_PATH})",
)
@click.option(
    "--detect-config",
    type=str,
    default=DEFAULT_DETECT_CONFIG_PATH,
    help=f"検出設定ファイルのパス (デフォルト: {DEFAULT_DETECT_CONFIG_PATH})",
)
@click.option(
    "--detect",
    "detect_enabled",
    is_flag=True,
    default=False,
    help="常時検出ランタイムを有効化 (pochidetection WebAPI を使用)",
)
@click.pass_context
def run(
    ctx: click.Context,
    camera: int,
    profile: str | None,
    list_profiles: bool,
    config: str,
    no_recording: bool,
    infer_config: str,
    detect_config: str,
    detect_enabled: bool,
) -> None:
    """ライブプレビューを起動する (従来の pochi コマンド)."""
    log_manager = LogManager()
    logger = log_manager.get_logger()
    logger.info("Starting pochivision application")

    config_data = _load_config(config, logger)

    if list_profiles:
        _print_profiles(config_data)
        return

    output_manager = ctx.obj.get("output_manager") if ctx.obj else None
    if output_manager is None:
        output_manager = OutputManager()

    cap, camera_setup = _setup_camera(config_data, log_manager, camera, profile)
    _run_preview(
        config_data,
        log_manager,
        cap,
        camera_setup,
        no_recording,
        output_manager,
        infer_config,
        detect_config,
        detect_enabled,
    )


def _load_config(config_path: str, logger: logging.Logger) -> dict:
    """設定ファイルを読み込む.

    Args:
        config_path: 設定ファイルのパス.
        logger: ロガーインスタンス.

    Returns:
        設定辞書.

    Raises:
        click.ClickException: 設定ファイルの読み込みに失敗した場合.
    """
    try:
        config_data = ConfigHandler.load(config_path)
        logger.info("Configuration loaded successfully")
        return config_data
    except ConfigValidationError as e:
        logger.error(str(e))
        raise click.ClickException(
            "設定ファイルに誤りがあります. 詳細はログを確認してください."
        )
    except (ConfigLoadError, Exception) as e:
        logger.error(f"Failed to load configuration: {e}")
        raise click.ClickException(str(e))


def _print_profiles(config_data: dict) -> None:
    """プロファイル一覧を表示する.

    Args:
        config_data: 設定辞書.
    """
    for prof in config_data.get("cameras", {}).keys():
        prof_config = config_data["cameras"][prof]
        click.echo(
            f"Profile: {prof}, "
            f"Resolution: {prof_config.get('width', 'default')}x"
            f"{prof_config.get('height', 'default')}, "
            f"FPS: {prof_config.get('fps', 'default')}"
        )


def _setup_camera(
    config_data: dict,
    log_manager: LogManager,
    camera: int,
    profile: str | None,
) -> tuple[cv2.VideoCapture, CameraSetup]:
    """カメラをセットアップする.

    Args:
        config_data: 設定辞書.
        log_manager: ログマネージャ.
        camera: カメラインデックス.
        profile: プロファイル名.

    Returns:
        (cap, camera_setup) のタプル.

    Raises:
        click.ClickException: カメラのセットアップに失敗した場合.
    """
    logger = log_manager.get_logger()
    try:
        camera_setup = CameraSetup(
            config_data,
            log_manager,
            camera_index=camera,
            profile_name=profile or "0",
        )
        camera_setup.load_camera_config()
        cap = camera_setup.initialize_camera()

        if cap is None or not cap.isOpened():
            logger.error(f"Failed to open camera {camera_setup.camera_index}.")
            raise click.ClickException(
                f"カメラ {camera_setup.camera_index} を開けませんでした."
            )

        log_manager.log_camera_info(
            cap,
            camera_setup.camera_index,
            camera_setup.width,
            camera_setup.height,
            profile_name=camera_setup.profile_name,
        )
        return cap, camera_setup

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Error setting up camera: {e}")
        raise click.ClickException(str(e))


def _run_preview(
    config_data: dict,
    log_manager: LogManager,
    cap: cv2.VideoCapture,
    camera_setup: CameraSetup,
    no_recording: bool,
    output_manager: OutputManager,
    infer_config_path: str,
    detect_config_path: str,
    detect_enabled: bool,
) -> None:
    """プレビューを実行する.

    Args:
        config_data: 設定辞書.
        log_manager: ログマネージャ.
        cap: カメラキャプチャオブジェクト (cv2.VideoCapture).
        camera_setup: カメラセットアップ.
        no_recording: 録画無効フラグ.
        output_manager: 出力ディレクトリの統一管理クラス.
        infer_config_path: 推論設定ファイルのパス.
        detect_config_path: 検出設定ファイルのパス.
        detect_enabled: `--detect` フラグで検出ランタイムを有効化するか.
    """
    logger = log_manager.get_logger()
    try:
        output_dir = output_manager.create_output_dir("capture")

        log_manager.setup_file_logging(output_dir / "capture.log")
        log_manager.log_system_info()

        used_profile = camera_setup.profile_name
        minimal_config = {
            "cameras": {used_profile: config_data["cameras"][used_profile]},
            "selected_camera_index": camera_setup.camera_index,
        }
        ConfigHandler.save(minimal_config, output_dir)

        pipeline = PipelineExecutor.from_config(
            config_data,
            output_dir=output_dir,
            camera_index=camera_setup.camera_index,
            profile_name=camera_setup.profile_name,
        )

        recording_manager = None
        if not no_recording:
            recording_config = config_data.get("recording", {})
            select_format = recording_config.get("select_format", "mjpg")
            recording_manager = RecordingManager(default_format=select_format)

        preview_config = config_data.get("preview", {})
        preview_size = (
            preview_config.get("width", DEFAULT_PREVIEW_WIDTH),
            preview_config.get("height", DEFAULT_PREVIEW_HEIGHT),
        )

        detection_client, detect_fps = _build_detection_client(
            detect_config_path, detect_enabled, logger
        )
        inference_client = None
        if detection_client is None:
            inference_client = _build_inference_client(infer_config_path, logger)

        app = LivePreviewRunner(
            cap,
            pipeline,
            recording_manager,
            preview_size,
            inference_client,
            camera_setup=camera_setup,
            detection_client=detection_client,
            detect_fps=detect_fps,
        )
        app.run()

    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        logger.info("Application shutdown complete")


def _build_inference_client(
    infer_config_path: str, logger: logging.Logger
) -> InferenceClient | None:
    """分類推論クライアントを構築する.

    設定ファイルが存在しない, または読み込みに失敗した場合は None を返す
    (推論機能は無効化されるがアプリは継続起動).

    Args:
        infer_config_path: 推論設定ファイルのパス.
        logger: ロガー.

    Returns:
        構築された InferenceClient, または None.
    """
    if not Path(infer_config_path).exists():
        return None
    try:
        infer_cfg = load_infer_config(infer_config_path)
        client = InferenceClient(
            base_url=infer_cfg.base_url,
            image_format=infer_cfg.image_format,
            resize=infer_cfg.resize,
            save_frame=infer_cfg.save_frame,
            save_csv=infer_cfg.save_csv,
        )
        logger.info(f"Inference API enabled: {infer_cfg.base_url}")
        return client
    except (ConfigLoadError, ConfigValidationError, ValueError) as e:
        logger.warning(f"Inference config not loaded, skipping: {e}")
        return None


def _build_detection_client(
    detect_config_path: str, detect_enabled: bool, logger: logging.Logger
) -> tuple[DetectionClient | None, float]:
    """検出クライアントを構築する.

    `detect_enabled=True` (CLI の `--detect` フラグ指定) のときのみクライアントを
    生成する. 設定ファイルが存在しない, または読み込みに失敗した場合は
    (None, デフォルト fps) を返す (分類モードにフォールバック).

    Args:
        detect_config_path: 検出設定ファイルのパス.
        detect_enabled: `--detect` フラグで検出モードが有効化されているか.
        logger: ロガー.

    Returns:
        (DetectionClient or None, detect_fps).
    """
    if not detect_enabled:
        return None, DEFAULT_DETECTION_FPS

    if not Path(detect_config_path).exists():
        logger.warning(
            f"--detect is set but config file is missing: {detect_config_path} "
            f"(falling back to classify mode)"
        )
        return None, DEFAULT_DETECTION_FPS
    try:
        detect_cfg = load_detect_config(detect_config_path)
    except (ConfigLoadError, ConfigValidationError, ValueError) as e:
        logger.warning(f"Detect config not loaded, skipping: {e}")
        return None, DEFAULT_DETECTION_FPS

    try:
        client = DetectionClient(
            base_url=detect_cfg.base_url,
            timeout=detect_cfg.timeout,
            image_format=detect_cfg.image_format,
            score_threshold=detect_cfg.score_threshold,
            jpeg_quality=detect_cfg.jpeg_quality,
        )
        logger.info(
            f"Detection API enabled: {detect_cfg.base_url} "
            f"(fps={detect_cfg.detect_fps})"
        )
        return client, detect_cfg.detect_fps
    except ValueError as e:
        logger.warning(f"Detection client not initialized: {e}")
        return None, detect_cfg.detect_fps
