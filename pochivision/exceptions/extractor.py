"""特徴量抽出器関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ExtractorValidationError(VisionCaptureError, ValueError):
    """特徴量抽出器の入力バリデーションエラー用例外クラス."""

    pass


class ExtractorRuntimeError(VisionCaptureError, RuntimeError):
    """特徴量抽出器の実行時エラー用例外クラス."""

    pass
