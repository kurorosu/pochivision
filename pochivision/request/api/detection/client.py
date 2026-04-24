"""pochidetection 検出 API クライアントモジュール."""

import base64
import time
from typing import Any

import cv2
import httpx
import numpy as np

from pochivision.constants import (
    DEFAULT_DETECTION_FORMAT,
    DEFAULT_DETECTION_JPEG_QUALITY,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_DETECTION_TIMEOUT,
)
from pochivision.exceptions import DetectionConnectionError, DetectionError

from .models import Detection, DetectionResponse


class DetectionClient:
    """pochidetection 検出 API と通信するクライアント.

    キャプチャフレームを base64 エンコードして API に送信し,
    検出結果を DetectionResponse として返す.

    Attributes:
        base_url: API サーバーのベース URL.
        image_format: 画像送信形式 ("raw" or "jpeg").
        score_threshold: 検出信頼度の下限しきい値.
        jpeg_quality: JPEG 圧縮品質.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_DETECTION_TIMEOUT,
        image_format: str = DEFAULT_DETECTION_FORMAT,
        score_threshold: float = DEFAULT_DETECTION_SCORE_THRESHOLD,
        jpeg_quality: int = DEFAULT_DETECTION_JPEG_QUALITY,
    ) -> None:
        """クライアントを初期化する.

        Args:
            base_url: pochidetection API サーバーの URL (例: "http://localhost:8000").
            timeout: リクエストタイムアウト (秒).
            image_format: 画像送信形式 ("raw" or "jpeg").
            score_threshold: 検出信頼度の下限しきい値 (0.0-1.0).
            jpeg_quality: JPEG 圧縮品質 (1-100). image_format="jpeg" のとき使用.
        """
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"base_url は http:// または https:// で始まる必要があります: {base_url}"
            )
        self.base_url = base_url.rstrip("/")
        valid_formats = {"raw", "jpeg"}
        if image_format not in valid_formats:
            raise ValueError(
                f"image_format は {valid_formats} のいずれかである必要があります: "
                f"{image_format!r}"
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                f"score_threshold は 0.0-1.0 である必要があります: {score_threshold!r}"
            )
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"jpeg_quality は 1-100 である必要があります: {jpeg_quality!r}"
            )
        self.image_format = image_format
        self.score_threshold = score_threshold
        self.jpeg_quality = jpeg_quality
        self._client = httpx.Client(timeout=timeout)

    def detect(self, frame: np.ndarray) -> DetectionResponse:
        """フレームを API に送信して検出結果を取得する.

        Args:
            frame: BGR 形式の numpy 配列 (cv2 キャプチャフレーム).

        Returns:
            検出結果.

        Raises:
            DetectionConnectionError: API サーバーへの接続に失敗した場合.
            DetectionError: 検出リクエストが失敗した場合.
            ValueError: フレームが uint8 以外の dtype の場合.
        """
        if frame.dtype != np.uint8:
            raise ValueError(f"frame は uint8 dtype 必須 (サーバー仕様): {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame は (H, W, 3) の 3 チャネル画像必須: {frame.shape}")
        # Why: detect() の呼び出し全体時間 (画像エンコード + RTT + JSON parse) を計測する.
        total_start = time.perf_counter()
        payload = self._build_payload(frame)
        url = f"{self.base_url}/api/v1/detect"

        rtt_start = time.perf_counter()
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ) as e:
            raise DetectionConnectionError(
                f"検出 API サーバーに接続できません: {self.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise DetectionError(
                f"検出リクエストが失敗しました (status={e.response.status_code})"
            ) from e
        except httpx.HTTPError as e:
            raise DetectionError(f"検出リクエスト中にエラーが発生しました: {e}") from e

        rtt_ms = (time.perf_counter() - rtt_start) * 1000

        try:
            data = response.json()
        except ValueError as e:
            raise DetectionError("検出レスポンスの JSON パースに失敗しました") from e

        total_ms = (time.perf_counter() - total_start) * 1000
        return self._parse_response(data, rtt_ms, total_ms)

    def _build_payload(self, frame: np.ndarray) -> dict[str, Any]:
        """API リクエストのペイロードを構築する.

        Args:
            frame: BGR 形式の numpy 配列.

        Returns:
            API リクエスト用の辞書.
        """
        if self.image_format == "jpeg":
            payload = self._encode_jpeg(frame)
        else:
            payload = self._encode_raw(frame)
        payload["score_threshold"] = self.score_threshold
        return payload

    def _encode_raw(self, frame: np.ndarray) -> dict[str, Any]:
        """Raw 形式で base64 エンコードしてペイロードを構築する.

        Args:
            frame: BGR 形式の numpy 配列.

        Returns:
            raw 形式のペイロード辞書.
        """
        image_data = base64.b64encode(frame.tobytes()).decode("ascii")
        return {
            "image_data": image_data,
            "format": "raw",
            "shape": list(frame.shape),
            "dtype": frame.dtype.name,
        }

    def _encode_jpeg(self, frame: np.ndarray) -> dict[str, Any]:
        """JPEG 圧縮後に base64 エンコードしてペイロードを構築する.

        Args:
            frame: BGR 形式の numpy 配列.

        Returns:
            jpeg 形式のペイロード辞書.
        """
        try:
            success, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
        except cv2.error as e:
            raise DetectionError(
                f"フレームの JPEG エンコードに失敗しました: {e}"
            ) from e
        if not success:
            raise DetectionError("フレームの JPEG エンコードに失敗しました")
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {
            "image_data": image_data,
            "format": "jpeg",
        }

    def _parse_response(
        self, data: dict[str, Any], rtt_ms: float, total_ms: float
    ) -> DetectionResponse:
        """レスポンス JSON を DetectionResponse に変換する.

        Args:
            data: レスポンス JSON.
            rtt_ms: クライアント側で計測した RTT (ミリ秒).
            total_ms: クライアント側で計測した detect() 全体時間 (ミリ秒).
                画像エンコード + RTT + JSON parse を含む.

        Returns:
            検出結果.

        Raises:
            DetectionError: 必須フィールドが欠落している場合.
        """
        try:
            raw_detections = data["detections"]
            detections = tuple(
                Detection(
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=(
                        d["bbox"][0],
                        d["bbox"][1],
                        d["bbox"][2],
                        d["bbox"][3],
                    ),
                )
                for d in raw_detections
            )
            # 旧バージョンの pochidetection (phase_times_ms 未対応) や, サーバー側で
            # null を返すケースに備え, 空 dict に正規化する.
            phase_times_ms = data.get("phase_times_ms") or {}
            return DetectionResponse(
                detections=detections,
                e2e_time_ms=data["e2e_time_ms"],
                backend=data["backend"],
                rtt_ms=round(rtt_ms, 3),
                total_ms=round(total_ms, 3),
                phase_times_ms=dict(phase_times_ms),
                gpu_clock_mhz=data.get("gpu_clock_mhz"),
                gpu_vram_used_mb=data.get("gpu_vram_used_mb"),
                gpu_temperature_c=data.get("gpu_temperature_c"),
            )
        except (KeyError, IndexError, TypeError) as e:
            raise DetectionError(f"検出レスポンスのフォーマットが不正です: {e}") from e

    def close(self) -> None:
        """HTTP クライアントを閉じる."""
        self._client.close()

    def __enter__(self) -> "DetectionClient":
        """コンテキストマネージャのエントリ."""
        return self

    def __exit__(self, *args: object) -> None:
        """コンテキストマネージャのイグジット."""
        self.close()
