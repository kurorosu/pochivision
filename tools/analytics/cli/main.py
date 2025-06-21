#!/usr/bin/env python3
"""Main entry point for CSV Analytics CLI."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# パス設定を実行（linterの自動整形を回避）
sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402
from utils.path_setup import setup_analytics_path  # noqa: E402

setup_analytics_path()  # noqa: E402

# 注意: パス設定の関係で、通常のPEP 8インポート順序（標準→サードパーティ→自作）を守れません
# サードパーティライブラリ
import click  # noqa: E402

# 自作モジュール
from analytics.cli.commands.data_overview import (  # noqa: E402
    show_data_overview_command,
)
from analytics.cli.commands.file_loader import load_csv_file_command  # noqa: E402
from analytics.cli.commands.histogram import HistogramManager  # noqa: E402
from analytics.cli.commands.scatter_plot import ScatterPlotManager  # noqa: E402
from analytics.core.data_processor import DataProcessor  # noqa: E402
from analytics.ui.display import (  # noqa: E402
    show_goodbye_message,
    show_main_menu_header,
    show_startup_loading,
    show_welcome_message,
)
from analytics.ui.prompts import select_main_menu_option  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()


class CSVAnalyticsCLI:
    """CSV Analytics CLIアプリケーション."""

    def __init__(self):
        """初期化処理."""
        self.data_processor = DataProcessor()
        self.histogram_manager = HistogramManager()
        self.scatter_plot_manager = ScatterPlotManager()

    def run(self) -> None:
        """メインアプリケーションを実行します."""
        # 起動時の読み込みアニメーションを最初に表示（ターミナルクリア含む）
        show_startup_loading()

        # ウェルカムメッセージを表示
        show_welcome_message()

        # メインメニューループ
        while True:
            # メインメニューのヘッダーを表示
            show_main_menu_header(
                self.data_processor.data,
                self.data_processor.file_path,
                self.histogram_manager.display_mode,
                self.histogram_manager.selected_class_column,
            )

            # メニュー選択
            choice = select_main_menu_option()
            if not choice:
                continue

            if choice == "1. CSVファイルを読み込む":
                success = load_csv_file_command(self.data_processor)
                if success:
                    # 新しいファイルを読み込んだ場合は表示設定をリセット
                    self.histogram_manager.reset_settings()
                    self.scatter_plot_manager.reset_settings()

            elif choice == "2. データの概要を表示":
                show_data_overview_command(self.data_processor)

            elif choice == "3. ヒストグラムを表示":
                self.histogram_manager.show_histogram_command(self.data_processor)

            elif choice == "4. 散布図を表示":
                self.scatter_plot_manager.show_scatter_plot_command(self.data_processor)

            elif choice == "5. 終了":
                show_goodbye_message()
                break


@click.command()
def main():
    """CSV Analytics CLI のメインエントリーポイント."""
    try:
        app = CSVAnalyticsCLI()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]アプリケーションが中断されました。[/yellow]")
    except Exception as e:
        console.print(f"\n[red]予期しないエラーが発生しました: {str(e)}[/red]")


if __name__ == "__main__":
    main()
