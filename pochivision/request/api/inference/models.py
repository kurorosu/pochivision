"""推論 API のレスポンスモデルを定義するモジュール."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictResponse:
    """pochitrain 推論 API のレスポンスデータ.

    Attributes:
        class_id: 予測クラス ID.
        class_name: 予測クラス名.
        confidence: 信頼度 (0.0-1.0).
        probabilities: 全クラスの確率.
        e2e_time_ms: サーバー側エンドツーエンド処理時間 (ミリ秒).
        backend: 使用バックエンド.
        rtt_ms: クライアント側ネットワーク往復時間 (ミリ秒).
    """

    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]
    e2e_time_ms: float
    backend: str
    rtt_ms: float
