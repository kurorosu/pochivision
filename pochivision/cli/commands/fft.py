"""fft サブコマンド: FFT ビジュアライザー."""

import click

from pochivision.core.fft_visualization import SimpleFFTVisualizer


@click.command()
@click.option(
    "--input", "-i", "input_path", type=str, required=True, help="入力画像パス"
)
def fft(input_path: str) -> None:
    """FFT ビジュアライザーを起動する."""
    try:
        visualizer = SimpleFFTVisualizer(input_path)
        visualizer.run()
    except KeyboardInterrupt:
        print("\n中断されました")
    except Exception as e:
        raise click.ClickException(str(e))
