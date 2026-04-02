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
def aggregate(input_dir: str, mode: str) -> None:
    """画像を集約する."""
    operation_mode = OperationMode.MOVE if mode == "move" else OperationMode.COPY
    aggregator = ImageAggregator(input_dir, "", operation_mode)
    num_processed = aggregator.aggregate()

    if num_processed > 0:
        click.echo(f"Successfully {mode}d {num_processed} images to image_aggregated")
    else:
        click.echo("No images were processed.")
