"""pochivision の CLI エントリーポイント."""

import click

from pochivision.cli.commands import register_commands


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Pochivision - AI ビジョン向けリアルタイム画像キャプチャ・前処理エンジン."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


register_commands(main)
