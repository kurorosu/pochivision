"""出力ディレクトリの統一管理を行うモジュール."""

from datetime import datetime
from pathlib import Path


class OutputManager:
    """全サブコマンドの出力ディレクトリを統一管理するクラス.

    出力は以下の構造で作成されます:
    {root}/
        capture/YYYYMMDD_{suffix}/
        processed/YYYYMMDD_{suffix}/
        aggregated/YYYYMMDD_{suffix}/
        features/YYYYMMDD_{suffix}/

    Attributes:
        root: 出力ルートディレクトリ.
    """

    def __init__(self, root: str = "outputs") -> None:
        """OutputManagerを初期化する.

        Args:
            root: 出力ルートディレクトリ. デフォルトは "outputs".
        """
        self.root = Path(root)

    def _get_next_suffix(self, base_dir: Path, date_str: str) -> int:
        """指定された日付の次のサフィックス番号を取得する.

        同じ日付のディレクトリが存在する場合,
        最大のサフィックス番号に1を加えた値を返します.

        Args:
            base_dir: サフィックスを検索するディレクトリ.
            date_str: 日付文字列 (YYYYMMDD形式).

        Returns:
            次のサフィックス番号.
        """
        if not base_dir.exists():
            return 0

        max_suffix = -1
        for dir_path in base_dir.iterdir():
            if not dir_path.is_dir():
                continue

            if dir_path.name.startswith(date_str + "_"):
                try:
                    suffix = int(dir_path.name.split("_")[-1])
                    max_suffix = max(max_suffix, suffix)
                except ValueError:
                    continue

        return max_suffix + 1

    def create_output_dir(self, command_type: str) -> Path:
        """コマンド種別に応じた出力ディレクトリを作成する.

        Args:
            command_type: コマンド種別.
                "capture", "processed", "aggregated", "features" のいずれか.

        Returns:
            作成されたディレクトリのパス.
            例: outputs/capture/YYYYMMDD_{suffix}/
        """
        base_dir = self.root / command_type
        base_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        suffix = self._get_next_suffix(base_dir, date_str)

        output_dir = base_dir / f"{date_str}_{suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
