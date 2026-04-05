"""推論結果の CSV 出力を管理するモジュール."""

import csv
from datetime import datetime
from pathlib import Path

from pochivision.request.api.inference.models import PredictResponse

_CSV_COLUMNS = [
    "timestamp",
    "class_id",
    "class_name",
    "confidence",
    "e2e_time_ms",
    "rtt_ms",
    "backend",
    "image_file",
]

_CSV_FILENAME = "inference_results.csv"


class InferenceCsvWriter:
    """推論結果を CSV ファイルに追記するライター.

    Attributes:
        csv_path: CSV ファイルのパス.
    """

    def __init__(self, output_dir: Path) -> None:
        """初期化する.

        Args:
            output_dir: 推論出力ディレクトリ (inference/).
        """
        self.csv_path = output_dir / _CSV_FILENAME

    def write_row(
        self,
        result: PredictResponse,
        image_file: str | None = None,
    ) -> None:
        """推論結果を CSV に 1 行追記する.

        ファイルが存在しない場合はヘッダ行付きで新規作成する.

        Args:
            result: 推論レスポンス.
            image_file: 保存された推論フレームのファイル名 (None の場合は空文字).
        """
        write_header = not self.csv_path.exists()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "class_id": result.class_id,
                    "class_name": result.class_name,
                    "confidence": result.confidence,
                    "e2e_time_ms": result.e2e_time_ms,
                    "rtt_ms": result.rtt_ms,
                    "backend": result.backend,
                    "image_file": image_file or "",
                }
            )
