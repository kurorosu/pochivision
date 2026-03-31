"""設定ファイル関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ConfigValidationError(VisionCaptureError):
    """設定ファイルのバリデーションエラー用例外クラス."""

    pass


class ConfigLoadError(VisionCaptureError):
    """設定ファイルの読み込みエラー用例外クラス."""

    pass


class CameraConfigError(VisionCaptureError):
    """カメラ設定のエラー用例外クラス."""

    pass
