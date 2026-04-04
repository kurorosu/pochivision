"""extract サブコマンド: 特徴量抽出."""

import click

from pochivision.core.feature_extraction import FeatureExtractionRunner
from pochivision.exceptions.config import ConfigLoadError


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="config/extractor_config.json",
    help="設定ファイルパス",
)
@click.pass_context
def extract(ctx: click.Context, config: str) -> None:
    """画像から特徴量を抽出する."""
    output_manager = ctx.obj.get("output_manager") if ctx.obj else None
    try:
        runner = FeatureExtractionRunner(config, output_manager)
        runner.run()
    except (ConfigLoadError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
