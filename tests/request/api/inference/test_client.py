"""InferenceClient のテスト."""

import base64
import json

import cv2
import httpx
import numpy as np
import pytest

from pochivision.exceptions import InferenceConnectionError, InferenceError
from pochivision.request.api.inference.client import InferenceClient
from pochivision.request.api.inference.models import PredictResponse

_VALID_RESPONSE = {
    "class_id": 0,
    "class_name": "class_a",
    "confidence": 0.95,
    "probabilities": [0.95, 0.05],
    "e2e_time_ms": 12.3,
    "backend": "onnx",
}


def _make_frame(height: int = 48, width: int = 64) -> np.ndarray:
    """テスト用のフレームを生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def raw_client():
    """Raw 形式の InferenceClient を提供する."""
    with InferenceClient(
        base_url="http://localhost:8000", image_format="raw"
    ) as client:
        yield client


@pytest.fixture
def jpeg_client():
    """JPEG 形式の InferenceClient を提供する."""
    with InferenceClient(
        base_url="http://localhost:8000", image_format="jpeg"
    ) as client:
        yield client


class TestEncodeRaw:
    """_encode_raw のテスト."""

    def test_payload_format(self, raw_client):
        frame = _make_frame()
        payload = raw_client._encode_raw(frame)

        assert payload["format"] == "raw"
        assert payload["shape"] == [48, 64, 3]
        assert payload["dtype"] == "uint8"
        assert isinstance(payload["image_data"], str)

    def test_roundtrip(self, raw_client):
        frame = _make_frame()
        payload = raw_client._encode_raw(frame)

        decoded = np.frombuffer(
            base64.b64decode(payload["image_data"]),
            dtype=np.uint8,
        ).reshape(payload["shape"])
        np.testing.assert_array_equal(frame, decoded)


class TestEncodeJpeg:
    """_encode_jpeg のテスト."""

    def test_payload_format(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._encode_jpeg(frame)

        assert payload["format"] == "jpeg"
        assert "shape" not in payload
        assert isinstance(payload["image_data"], str)

    def test_decodable(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._encode_jpeg(frame)

        jpeg_bytes = base64.b64decode(payload["image_data"])
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[2] == 3

    def test_encode_empty_frame(self, jpeg_client):
        frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(InferenceError, match="JPEG"):
            jpeg_client._encode_jpeg(frame)


class TestBuildPayload:
    """_build_payload のテスト."""

    def test_jpeg_format(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._build_payload(frame)
        assert payload["format"] == "jpeg"

    def test_raw_format(self, raw_client):
        frame = _make_frame()
        payload = raw_client._build_payload(frame)
        assert payload["format"] == "raw"


class TestPredictResponse:
    """PredictResponse モデルのテスト."""

    def test_from_dict(self):
        data = {
            "class_id": 1,
            "class_name": "dog",
            "confidence": 0.85,
            "probabilities": [0.15, 0.85],
            "e2e_time_ms": 10.5,
            "backend": "onnx",
            "rtt_ms": 50.0,
        }
        resp = PredictResponse(**data)
        assert resp.class_id == 1
        assert resp.class_name == "dog"
        assert resp.confidence == 0.85
        assert resp.probabilities == [0.15, 0.85]
        assert resp.e2e_time_ms == 10.5
        assert resp.backend == "onnx"
        assert resp.rtt_ms == 50.0

    def test_frozen(self):
        resp = PredictResponse(
            class_id=0,
            class_name="cat",
            confidence=0.9,
            probabilities=[0.9, 0.1],
            e2e_time_ms=5.0,
            backend="onnx",
            rtt_ms=30.0,
        )
        with pytest.raises(AttributeError):
            resp.class_name = "dog"  # type: ignore[misc]


class TestPredict:
    """predict() のテスト."""

    def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = InferenceClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        result = client.predict(_make_frame())
        assert result.class_id == 0
        assert result.class_name == "class_a"
        assert result.confidence == 0.95
        assert result.e2e_time_ms == 12.3
        assert result.rtt_ms > 0
        client.close()

    def test_http_status_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "server error"})

        client = InferenceClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(InferenceError, match="status=500"):
            client.predict(_make_frame())
        client.close()

    def test_invalid_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        client = InferenceClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(InferenceError, match="JSON"):
            client.predict(_make_frame())
        client.close()

    def test_missing_response_field(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"class_id": 0})

        client = InferenceClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(InferenceError, match="フィールドがありません"):
            client.predict(_make_frame())
        client.close()


class TestPredictConnectionError:
    """接続エラーのテスト."""

    def test_connection_refused(self):
        with InferenceClient(
            base_url="http://127.0.0.1:19999",
            timeout=1.0,
        ) as client:
            with pytest.raises(InferenceConnectionError):
                client.predict(_make_frame())
