"""pochivision の CLI エントリーポイント."""

import click

from pochivision.cli.commands import register_commands
from pochivision.workspace import OutputManager


@click.group(invoke_without_command=True)
@click.option(
    "--output-root",
    type=str,
    default="outputs",
    help="出力ルートディレクトリ (デフォルト: outputs)",
)
@click.pass_context
def main(ctx: click.Context, output_root: str) -> None:
    """Pochivision - AI ビジョン向けリアルタイム画像キャプチャ・前処理エンジン."""
    ctx.ensure_object(dict)
    ctx.obj["output_manager"] = OutputManager(output_root)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


register_commands(main)
