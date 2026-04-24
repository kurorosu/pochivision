"""検出 API のレスポンスモデルを定義するモジュール."""

from dataclasses import dataclass, field


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
        phase_times_ms: Pipeline 内のフェーズ別タイミング (ms). サーバー未提供時は空 dict.
            想定キー: api_preprocess_ms / pipeline_preprocess_ms / pipeline_inference_ms /
            pipeline_postprocess_ms / pipeline_inference_gpu_ms / api_postprocess_ms.
            api_* は API 境界 (deserialize + cvtColor / response 組立) のコスト.
        gpu_clock_mhz: GPU graphics clock (MHz). サーバーが取得できない場合 None.
        gpu_vram_used_mb: GPU VRAM 使用量 (MB). サーバーが取得できない場合 None.
        gpu_temperature_c: GPU 温度 (℃). サーバーが取得できない場合 None.
    """

    detections: tuple[Detection, ...]
    e2e_time_ms: float
    backend: str
    rtt_ms: float
    phase_times_ms: dict[str, float] = field(default_factory=dict)
    gpu_clock_mhz: int | None = None
    gpu_vram_used_mb: int | None = None
    gpu_temperature_c: int | None = None
