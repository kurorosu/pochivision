"""pochivision の CLI エントリーポイント."""

import click

from pochivision.capture_runner import LivePreviewRunner
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.capture_manager import CaptureManager
from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.core import PipelineExecutor
from pochivision.exceptions.config import ConfigValidationError

# tools/ のインポートは未インストール依存 (tqdm 等) を引き込むため遅延インポート


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Pochivision - AI ビジョン向けリアルタイム画像キャプチャ・前処理エンジン."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option("--camera", "-c", type=int, default=0, help="カメラデバイスインデックス")
@click.option("--profile", "-p", type=str, default=None, help="カメラプロファイル名")
@click.option("--list-profiles", "-l", is_flag=True, help="プロファイル一覧を表示")
@click.option("--config", type=str, default="config.json", help="設定ファイルパス")
@click.option("--no-recording", is_flag=True, help="録画機能を無効化")
def run(
    camera: int,
    profile: str | None,
    list_profiles: bool,
    config: str,
    no_recording: bool,
) -> None:
    """ライブプレビューを起動する (従来の pochi コマンド)."""
    log_manager = LogManager()
    logger = log_manager.get_logger()
    logger.info("Starting pochivision application")

    try:
        config_data = ConfigHandler.load(config)
        logger.info("Configuration loaded successfully")

        if list_profiles:
            for prof in config_data.get("cameras", {}).keys():
                prof_config = config_data["cameras"][prof]
                click.echo(
                    f"Profile: {prof}, "
                    f"Resolution: {prof_config.get('width', 'default')}x"
                    f"{prof_config.get('height', 'default')}, "
                    f"FPS: {prof_config.get('fps', 'default')}"
                )
            return

    except ConfigValidationError as e:
        logger.error(str(e))
        click.echo("設定ファイルに誤りがあります. 詳細はログを確認してください.")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise SystemExit(1)

    try:
        camera_setup = CameraSetup(
            config_data,
            log_manager,
            camera_index=camera,
            profile_name=profile or "0",
        )
        camera_setup.load_camera_config()
        cap = camera_setup.initialize_camera()

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_setup.camera_index}.")
            raise SystemExit(1)

        log_manager.log_camera_info(
            cap,
            camera_setup.camera_index,
            camera_setup.width,
            camera_setup.height,
            profile_name=camera_setup.profile_name,
        )

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error setting up camera: {e}")
        raise SystemExit(1)

    try:
        capture_manager = CaptureManager()
        log_manager.setup_file_logging(
            capture_manager.get_log_file_path(camera_index=camera_setup.camera_index)
        )
        log_manager.log_system_info()

        used_profile = camera_setup.profile_name
        minimal_config = {
            "cameras": {used_profile: config_data["cameras"][used_profile]},
            "selected_camera_index": camera_setup.camera_index,
        }
        ConfigHandler.save(
            minimal_config,
            capture_manager.get_output_dir(camera_index=camera_setup.camera_index),
        )

        pipeline = PipelineExecutor.from_config(
            config_data,
            capture_manager=capture_manager,
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
            preview_config.get("width", 1280),
            preview_config.get("height", 720),
        )

        app = LivePreviewRunner(cap, pipeline, recording_manager, preview_size)
        app.run()

    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        logger.info("Application shutdown complete")


@main.command()
@click.option(
    "--config", "-c", type=str, default="extractor_config.json", help="設定ファイルパス"
)
def extract(config: str) -> None:
    """画像から特徴量を抽出する."""
    from tools.feature_extraction import main as run_extract  # 遅延: tqdm 等の依存回避

    run_extract(config)


@main.command()
@click.option(
    "--config", "-c", type=str, default="extractor_config.json", help="設定ファイルパス"
)
@click.option(
    "--input", "-i", "input_dir", type=str, default=None, help="入力ディレクトリ"
)
@click.option(
    "--output", "-o", "output_dir", type=str, default=None, help="出力ディレクトリ"
)
@click.option("--profile", "-p", type=str, default=None, help="カメラプロファイル名")
def process(
    config: str, input_dir: str | None, output_dir: str | None, profile: str | None
) -> None:
    """カメラプロファイルを画像に適用する."""
    import sys

    sys.argv = ["process", "--config", config]
    if input_dir:
        sys.argv.extend(["--input", input_dir])
    if output_dir:
        sys.argv.extend(["--output", output_dir])
    if profile:
        sys.argv.extend(["--profile", profile])

    from tools.profile_processor import main as run_process  # 遅延: 依存回避

    run_process()


@main.command()
@click.option(
    "--input", "-i", "input_dir", type=str, required=True, help="入力ディレクトリ"
)
@click.option(
    "--output", "-o", "output_dir", type=str, default=None, help="出力ディレクトリ"
)
@click.option(
    "--mode", "-m", type=str, default="mean", help="集約モード (mean, median 等)"
)
def aggregate(input_dir: str, output_dir: str | None, mode: str) -> None:
    """画像を集約する."""
    import sys

    from tools.image_aggregator import main as run_aggregate  # 遅延: tqdm 依存回避

    sys.argv = ["aggregate", "--input", input_dir, "--mode", mode]
    if output_dir:
        sys.argv.extend(["--output", output_dir])

    run_aggregate()


@main.command()
@click.option(
    "--input", "-i", "input_path", type=str, required=True, help="入力画像パス"
)
@click.option("--width", "-w", type=int, default=1920, help="ウィンドウ幅")
@click.option(
    "--height", "-h", "win_height", type=int, default=1080, help="ウィンドウ高さ"
)
def fft(input_path: str, width: int, win_height: int) -> None:
    """FFT ビジュアライザーを起動する."""
    import sys

    from tools.fft_visualizer import main as run_fft  # 遅延: dearpygui 依存回避

    sys.argv = [
        "fft",
        "--input",
        input_path,
        "--width",
        str(width),
        "--height",
        str(win_height),
    ]

    run_fft()
