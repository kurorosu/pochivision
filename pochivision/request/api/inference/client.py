"""pochitrain 推論 API クライアントモジュール."""

import base64
from typing import Any

import cv2
import httpx
import numpy as np

from pochivision.capturelib.log_manager import LogManager
from pochivision.constants import DEFAULT_INFERENCE_FORMAT, DEFAULT_INFERENCE_TIMEOUT
from pochivision.exceptions import InferenceConnectionError, InferenceError

from .models import PredictResponse


class InferenceClient:
    """pochitrain 推論 API と通信するクライアント.

    キャプチャフレームを base64 エンコードして API に送信し,
    推論結果を PredictResponse として返す.

    Attributes:
        base_url: API サーバーのベース URL.
        image_format: 画像送信形式 ("raw" or "jpeg").
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_INFERENCE_TIMEOUT,
        image_format: str = DEFAULT_INFERENCE_FORMAT,
    ) -> None:
        """クライアントを初期化する.

        Args:
            base_url: pochitrain API サーバーの URL (例: "http://localhost:8000").
            timeout: リクエストタイムアウト (秒).
            image_format: 画像送信形式 ("raw" or "jpeg").
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
        self.image_format = image_format
        self.logger = LogManager().get_logger()
        self._client = httpx.Client(timeout=timeout)

    def predict(self, frame: np.ndarray) -> PredictResponse:
        """フレームを API に送信して推論結果を取得する.

        Args:
            frame: BGR 形式の numpy 配列 (cv2 キャプチャフレーム).

        Returns:
            推論結果.

        Raises:
            InferenceConnectionError: API サーバーへの接続に失敗した場合.
            InferenceError: 推論リクエストが失敗した場合.
        """
        payload = self._build_payload(frame)
        url = f"{self.base_url}/api/v1/predict"

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise InferenceConnectionError(
                f"推論 API サーバーに接続できません: {self.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise InferenceError(
                f"推論リクエストが失敗しました (status={e.response.status_code})"
            ) from e
        except httpx.HTTPError as e:
            raise InferenceError(f"推論リクエスト中にエラーが発生しました: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            raise InferenceError("推論レスポンスの JSON パースに失敗しました") from e

        try:
            return PredictResponse(
                class_id=data["class_id"],
                class_name=data["class_name"],
                confidence=data["confidence"],
                probabilities=data["probabilities"],
                processing_time_ms=data["processing_time_ms"],
                backend=data["backend"],
            )
        except KeyError as e:
            raise InferenceError(
                f"推論レスポンスに必要なフィールドがありません: {e}"
            ) from e

    def _build_payload(self, frame: np.ndarray) -> dict[str, Any]:
        """API リクエストのペイロードを構築する.

        Args:
            frame: BGR 形式の numpy 配列.

        Returns:
            API リクエスト用の辞書.
        """
        if self.image_format == "jpeg":
            return self._encode_jpeg(frame)
        return self._encode_raw(frame)

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
            "dtype": str(frame.dtype),
        }

    def _encode_jpeg(self, frame: np.ndarray) -> dict[str, Any]:
        """JPEG 圧縮後に base64 エンコードしてペイロードを構築する.

        Args:
            frame: BGR 形式の numpy 配列.

        Returns:
            jpeg 形式のペイロード辞書.
        """
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise InferenceError("フレームの JPEG エンコードに失敗しました")
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {
            "image_data": image_data,
            "format": "jpeg",
        }

    def close(self) -> None:
        """HTTP クライアントを閉じる."""
        self._client.close()
