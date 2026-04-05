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
        processing_time_ms: 推論時間 (ミリ秒).
        backend: 使用バックエンド.
    """

    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]
    processing_time_ms: float
    backend: str
