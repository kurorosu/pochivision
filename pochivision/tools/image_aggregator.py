"""
画像処理後のファイルを集約するユーティリティ.

カメラフォルダ内のすべての日付フォルダから、処理タイプごとに画像を集約して
新しい出力フォルダ（image_aggregated）にコピーまたは移動します.
"""

import argparse
import sys

from pochivision.utils.image_aggregation import ImageAggregator, OperationMode


def parse_arguments():
    """
    コマンドライン引数の解析.

    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(
        description="Aggregate processed images from camera folders by processor type"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input camera directory (e.g. ./capture/camera0)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Operation mode: copy (default) or move files",
    )

    return parser.parse_args()


def main():
    """メイン実行関数."""
    args = parse_arguments()

    # 画像集約の実行
    mode = OperationMode.MOVE if args.mode == "move" else OperationMode.COPY

    try:
        # 出力ディレクトリは固定で"image_aggregated"
        aggregator = ImageAggregator(
            args.input,
            "",  # 出力ディレクトリは内部で固定
            mode,
        )
        num_processed = aggregator.aggregate()

        if num_processed > 0:
            print(
                f"Successfully {args.mode}d {num_processed} images to image_aggregated"
            )
            return 0
        else:
            print("No images were processed.")
            return 1
    except Exception as e:
        print(f"Error during image aggregation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
