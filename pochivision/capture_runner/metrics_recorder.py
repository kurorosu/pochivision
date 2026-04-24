"""検出 API の処理時間メトリクスをサンプリングして CSV に保存するモジュール."""

import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pochivision.capturelib.log_manager import LogManager
from pochivision.request.api.detection.models import DetectionResponse

_CSV_COLUMNS: list[str] = [
    "timestamp_iso",
    "elapsed_s",
    "num_detections",
    "total_ms",
    "e2e_time_ms",
    "rtt_ms",
    "backend",
    # 時系列順: api_preprocess → pipeline_* → api_postprocess.
    "api_preprocess_ms",
    "phase_preprocess_ms",
    "phase_inference_ms",
    "phase_inference_gpu_ms",
    "phase_postprocess_ms",
    "api_postprocess_ms",
    "gpu_clock_mhz",
    "gpu_vram_used_mb",
    "gpu_temperature_c",
]


class MetricsRecorder:
    """検出レスポンスから処理時間メトリクスをサンプリングして保存する.

    `maybe_record()` は呼び出しごとにサンプリング間隔を判定し, 間隔経過時のみ
    内部リストに追記する. `flush()` で pandas DataFrame 化して CSV に書き出す.

    Attributes:
        interval_s: サンプリング間隔 (秒). 0 以下は無効.
        out_path: CSV 出力先パス.
    """

    def __init__(self, interval_s: float, out_path: Path) -> None:
        """初期化する.

        Args:
            interval_s: サンプリング間隔 (秒). 0 以下のとき常に記録しない.
            out_path: CSV 出力先パス.
        """
        self.interval_s = interval_s
        self.out_path = Path(out_path)
        self._logger = LogManager().get_logger()
        # Why: elapsed_s を monotonic 起点で計測するため, 生成時刻を記録する.
        self._start_monotonic = time.monotonic()
        # Why: スレッドセーフな append / flush のため lock で保護する.
        self._lock = threading.Lock()
        self._rows: list[dict[str, object]] = []
        self._last_sampled_monotonic: float = -float("inf")

    def maybe_record(
        self,
        response: DetectionResponse,
        now_monotonic: float | None = None,
    ) -> bool:
        """サンプリング間隔が経過していれば内部バッファに 1 行追記する.

        Args:
            response: 検出レスポンス.
            now_monotonic: 現在時刻 (``time.monotonic()``). テスト用. None なら自動計測.

        Returns:
            追記した場合 True, スキップした場合 False.
        """
        if self.interval_s <= 0:
            return False

        current = now_monotonic if now_monotonic is not None else time.monotonic()
        with self._lock:
            if current - self._last_sampled_monotonic < self.interval_s:
                return False
            self._last_sampled_monotonic = current
            phases = response.phase_times_ms
            row: dict[str, object] = {
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": round(current - self._start_monotonic, 3),
                "num_detections": len(response.detections),
                "total_ms": response.total_ms,
                "e2e_time_ms": response.e2e_time_ms,
                "rtt_ms": response.rtt_ms,
                "backend": response.backend,
                "api_preprocess_ms": phases.get("api_preprocess_ms"),
                "phase_preprocess_ms": phases.get("pipeline_preprocess_ms"),
                "phase_inference_ms": phases.get("pipeline_inference_ms"),
                "phase_inference_gpu_ms": phases.get("pipeline_inference_gpu_ms"),
                "phase_postprocess_ms": phases.get("pipeline_postprocess_ms"),
                "api_postprocess_ms": phases.get("api_postprocess_ms"),
                "gpu_clock_mhz": response.gpu_clock_mhz,
                "gpu_vram_used_mb": response.gpu_vram_used_mb,
                "gpu_temperature_c": response.gpu_temperature_c,
            }
            self._rows.append(row)
        return True

    def flush(self) -> Path | None:
        """内部バッファを pandas DataFrame 化して CSV に書き出す.

        バッファが空のときは何もしない. 書き出し後はバッファをクリアする.

        Returns:
            書き出した CSV パス. バッファが空だった場合は None.
        """
        with self._lock:
            if not self._rows:
                return None
            df = pd.DataFrame(self._rows, columns=_CSV_COLUMNS)
            rows_to_write = self._rows
            self._rows = []

        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.out_path, index=False, encoding="utf-8")
        except OSError as e:
            # Why: 書き出し失敗時はバッファに戻してやり直せるようにする.
            with self._lock:
                self._rows = rows_to_write + self._rows
            self._logger.error(f"Failed to write metrics CSV: {e}")
            return None
        self._logger.info(f"Metrics saved: {self.out_path} ({len(rows_to_write)} rows)")
        return self.out_path

    @property
    def row_count(self) -> int:
        """現在のバッファ行数を返す (テスト用)."""
        with self._lock:
            return len(self._rows)
