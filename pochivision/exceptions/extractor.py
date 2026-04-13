"""特徴量抽出器関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ExtractorValidationError(VisionCaptureError, ValueError):
    """特徴量抽出器の入力バリデーションエラー用例外クラス."""

    pass


class ExtractorRegistrationError(VisionCaptureError, ValueError):
    """特徴量抽出器のレジストリ登録エラー用例外クラス."""

    pass
