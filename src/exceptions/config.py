"""設定ファイル関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ConfigValidationError(VisionCaptureError):
    """設定ファイルのバリデーションエラー用例外クラス."""

    pass
