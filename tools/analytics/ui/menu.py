"""Menu display components for CSV Analytics."""

import os
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()


def show_welcome_panel() -> None:
    """ウェルカムパネルを表示します."""
    welcome_text = """
[bold blue]CSV Analytics CLI[/bold blue]

CSVファイルの基本的なデータ分析を行うコマンドラインツールです。

主な機能:
• CSVファイルの読み込みと基本統計情報の表示
• インタラクティブなヒストグラム表示
• Long形式からWide形式への自動変換
• クラス別データ分析
    """
    console.print(Panel(welcome_text, title="ようこそ", border_style="blue"))


def show_main_menu_status(
    data: Optional[pd.DataFrame],
    file_path: Optional[str],
    display_mode: Optional[str],
    selected_class_column: Optional[str],
) -> None:
    """メインメニューの現在の状態を表示します."""
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]CSV Analytics CLI - メインメニュー[/bold cyan]")
    console.print("=" * 50)

    # 現在の設定状況を表示
    if data is not None and file_path is not None:
        console.print(
            f"[dim]📁 読み込み済みファイル: {os.path.basename(file_path)}[/dim]"
        )
        console.print(
            f"[dim]📊 データサイズ: {len(data):,} 行 × {len(data.columns):,} 列[/dim]"
        )

        if display_mode == "simple":
            console.print("[dim]🎨 表示設定: 単純なヒストグラム[/dim]")
        elif display_mode == "class":
            console.print(
                f"[dim]🎨 表示設定: クラス別色分け（{selected_class_column}）[/dim]"
            )
        else:
            console.print("[dim]🎨 表示設定: 未設定[/dim]")
    else:
        console.print("[dim]📁 ファイル: 未読み込み[/dim]")


def show_menu_separator() -> None:
    """メニューセパレータを表示します."""
    console.print("-" * 50)


def show_section_header(title: str, icon: str = "📋") -> None:
    """セクションヘッダーを表示します."""
    console.print(f"\n{icon} [bold]{title}[/bold]")
    console.print("-" * (len(title) + 4))


def show_success_message(message: str) -> None:
    """成功メッセージを表示します."""
    console.print(f"[green]✅ {message}[/green]")


def show_error_message(message: str) -> None:
    """エラーメッセージを表示します."""
    console.print(f"[red]❌ {message}[/red]")


def show_warning_message(message: str) -> None:
    """警告メッセージを表示します."""
    console.print(f"[yellow]⚠️  {message}[/yellow]")


def show_info_message(message: str) -> None:
    """情報メッセージを表示します."""
    console.print(f"[blue]ℹ️  {message}[/blue]")


def show_loading_message(message: str) -> None:
    """読み込み中メッセージを表示します."""
    console.print(f"[yellow]⏳ {message}[/yellow]")


def show_completion_message(message: str) -> None:
    """完了メッセージを表示します."""
    console.print(f"[green]🎉 {message}[/green]")
