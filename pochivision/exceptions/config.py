"""設定ファイル関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class ConfigValidationError(VisionCaptureError, ValueError):
    """設定ファイルのバリデーションエラー用例外クラス."""

    pass


class ConfigLoadError(VisionCaptureError, OSError):
    """設定ファイルの読み込みエラー用例外クラス."""

    pass


class CameraConfigError(VisionCaptureError, ValueError):
    """カメラ設定のエラー用例外クラス."""

    pass
