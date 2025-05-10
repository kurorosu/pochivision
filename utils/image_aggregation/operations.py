"""画像集約のファイル操作モードを定義するモジュール."""

from enum import Enum


class OperationMode(Enum):
    """
    ファイル操作モード.

    COPY: 元のファイルを保持したままコピーする
    MOVE: 元のファイルを削除して移動する
    """

    COPY = "copy"
    MOVE = "move"
