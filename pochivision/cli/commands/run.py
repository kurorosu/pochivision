"""run サブコマンド: ライブプレビュー起動."""

import logging

import click
import cv2

from pochivision.capture_runner import LivePreviewRunner
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.constants import DEFAULT_PREVIEW_HEIGHT, DEFAULT_PREVIEW_WIDTH
from pochivision.core import PipelineExecutor
from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError
from pochivision.request.api.inference.client import InferenceClient
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
    "--inference-url",
    type=str,
    default=None,
    help="pochitrain 推論 API の URL (例: http://192.168.1.100:8000)",
)
@click.option(
    "--inference-format",
    type=click.Choice(["raw", "jpeg"]),
    default="jpeg",
    help="推論 API への画像送信フォーマット",
)
@click.pass_context
def run(
    ctx: click.Context,
    camera: int,
    profile: str | None,
    list_profiles: bool,
    config: str,
    no_recording: bool,
    inference_url: str | None,
    inference_format: str,
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
        inference_url,
        inference_format,
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
            "設定ファイルに誤りがあります. 詳細はログ��確認してください."
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
                f"カメラ {camera_setup.camera_index} を開けませ���でした."
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
    inference_url: str | None,
    inference_format: str,
) -> None:
    """プレビューを実行する.

    Args:
        config_data: 設定辞書.
        log_manager: ログマネージャ.
        cap: カメラキャプチャオブジェクト (cv2.VideoCapture).
        camera_setup: カメラセットアップ.
        no_recording: 録画無効フラグ.
        output_manager: 出力ディレクトリの統一管理クラス.
        inference_url: pochitrain 推論 API の URL (None で無効).
        inference_format: 画像送信フォーマット ("raw" or "jpeg").
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

        inference_client = None
        if inference_url:
            inference_client = InferenceClient(
                base_url=inference_url,
                image_format=inference_format,
            )
            logger.info(f"Inference API enabled: {inference_url}")

        app = LivePreviewRunner(
            cap, pipeline, recording_manager, preview_size, inference_client
        )
        app.run()

    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        logger.info("Application shutdown complete")
