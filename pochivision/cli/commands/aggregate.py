"""aggregate サブコマンド: 画像集約."""

import click

from pochivision.utils.image_aggregation import ImageAggregator, OperationMode


@click.command()
@click.option(
    "--input", "-i", "input_dir", type=str, required=True, help="入力ディレクトリ"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["copy", "move"]),
    default="copy",
    help="操作モード (copy または move)",
)
@click.pass_context
def aggregate(ctx: click.Context, input_dir: str, mode: str) -> None:
    """画像を集約する."""
    output_manager = ctx.obj.get("output_manager") if ctx.obj else None
    operation_mode = OperationMode.MOVE if mode == "move" else OperationMode.COPY
    aggregator = ImageAggregator(input_dir, operation_mode, output_manager)
    num_processed = aggregator.aggregate()

    if num_processed > 0:
        click.echo(f"Successfully {mode}d {num_processed} images")
    else:
        click.echo("No images were processed.")
