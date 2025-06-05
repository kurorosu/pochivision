"""Path setup utility for CSV Analytics."""

import sys
from pathlib import Path


def setup_analytics_path() -> None:
    """analyticsパッケージのパスを設定します."""
    # 現在のファイルからプロジェクトルートを特定
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    tools_path = project_root / "tools"

    # パスが既に追加されていない場合のみ追加
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))
