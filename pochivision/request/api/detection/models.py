"""検出 API のレスポンスモデルを定義するモジュール."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """1 つの検出結果を表すデータ.

    Attributes:
        class_id: 予測クラス ID.
        class_name: 予測クラス名.
        confidence: 信頼度 (0.0-1.0).
        bbox: バウンディングボックス [x1, y1, x2, y2] (元画像座標系のピクセル値).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResponse:
    """pochidetection 検出 API のレスポンスデータ.

    Attributes:
        detections: 検出結果のリスト.
        e2e_time_ms: サーバー側エンドツーエンド処理時間 (ミリ秒).
        backend: 使用バックエンド.
        rtt_ms: クライアント側ネットワーク往復時間 (ミリ秒).
    """

    detections: tuple[Detection, ...]
    e2e_time_ms: float
    backend: str
    rtt_ms: float
