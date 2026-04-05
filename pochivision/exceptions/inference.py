"""推論 API 関連の例外クラスを定義するモジュール."""

from .base import VisionCaptureError


class InferenceError(VisionCaptureError, RuntimeError):
    """推論 API 呼び出し全般のエラー用例外クラス."""

    pass


class InferenceConnectionError(InferenceError, ConnectionError):
    """推論 API サーバーへの接続失敗用例外クラス."""

    pass
