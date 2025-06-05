"""
CSV Analytics CLI Package.

A command-line tool for basic CSV data analysis with interactive features.
"""

import sys
from pathlib import Path

# パッケージ初期化時にパス設定を自動実行
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
tools_path = project_root / "tools"

if str(tools_path) not in sys.path:
    sys.path.insert(0, str(tools_path))

__version__ = "0.1.0"
__author__ = "Vision Capture Core Team"
