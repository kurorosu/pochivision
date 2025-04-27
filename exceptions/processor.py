"""画像処理プロセッサ関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ProcessorValidationError(VisionCaptureError):
    """プロセッサのバリデーションエラー用例外クラス."""

    pass


class ProcessorRuntimeError(VisionCaptureError):
    """プロセッサ実行時のエラー用例外クラス."""

    pass
