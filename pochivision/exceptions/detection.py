"""検出 API 関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class DetectionError(VisionCaptureError, RuntimeError):
    """検出 API 呼び出し全般のエラー用例外クラス."""

    pass


class DetectionConnectionError(DetectionError, ConnectionError):
    """検出 API サーバーへの接続失敗用例外クラス."""

    pass
