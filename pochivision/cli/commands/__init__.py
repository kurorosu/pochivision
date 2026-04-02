"""CLI サブコマンドパッケージ."""

import click

from pochivision.cli.commands.aggregate import aggregate
from pochivision.cli.commands.extract import extract
from pochivision.cli.commands.fft import fft
from pochivision.cli.commands.process import process
from pochivision.cli.commands.run import run


def register_commands(group: click.Group) -> None:
    """全サブコマンドをグループに登録する.

    Args:
        group: click.Group インスタンス.
    """
    group.add_command(run)
    group.add_command(extract)
    group.add_command(process)
    group.add_command(aggregate)
    group.add_command(fft)
