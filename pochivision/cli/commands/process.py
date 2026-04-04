"""process サブコマンド: プロファイル適用."""

import json

import click

from pochivision.capturelib.log_manager import LogManager
from pochivision.core.profile_processing import ProfileProcessor
from pochivision.exceptions.config import ConfigLoadError, ConfigValidationError


@click.command()
@click.option(
    "--config", "-c", type=str, default="config/config.json", help="設定ファイルパス"
)
@click.option(
    "--input", "-i", "input_dir", type=str, required=True, help="入力ディレクトリ"
)
@click.option("--profile", "-p", type=str, required=True, help="カメラプロファイル名")
@click.option("--no-save-original", is_flag=True, help="元画像を保存しない")
@click.option("--list-profiles", is_flag=True, help="利用可能なプロファイルを一覧表示")
@click.pass_context
def process(
    ctx: click.Context,
    config: str,
    input_dir: str,
    profile: str,
    no_save_original: bool,
    list_profiles: bool,
) -> None:
    """カメラプロファイルを画像に適用する."""
    output_manager = ctx.obj.get("output_manager") if ctx.obj else None

    if list_profiles:
        try:
            with open(config, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            cameras = config_data.get("cameras", {})
            if cameras:
                first_profile = list(cameras.keys())[0]
                proc = ProfileProcessor(config, first_profile, output_manager)
                proc.list_available_profiles()
            else:
                LogManager().get_logger().warning("利用可能なプロファイルがありません")
        except Exception as e:
            LogManager().get_logger().error(
                f"プロファイル一覧の表示に失敗しました: {e}"
            )
        return

    try:
        processor = ProfileProcessor(config, profile, output_manager)
        processor.process_directory(input_dir, save_original=not no_save_original)
    except (ConfigLoadError, ConfigValidationError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
