"""InferenceClient のテスト."""

import base64
import json

import cv2
import numpy as np
import pytest

from pochivision.exceptions import InferenceConnectionError, InferenceError
from pochivision.request.api.inference.client import InferenceClient
from pochivision.request.api.inference.models import PredictResponse


def _make_frame(height: int = 48, width: int = 64) -> np.ndarray:
    """テスト用のフレームを生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestEncodeRaw:
    """_encode_raw のテスト."""

    def test_payload_format(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="raw")
        frame = _make_frame()
        payload = client._encode_raw(frame)

        assert payload["format"] == "raw"
        assert payload["shape"] == [48, 64, 3]
        assert payload["dtype"] == "uint8"
        assert isinstance(payload["image_data"], str)
        client.close()

    def test_roundtrip(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="raw")
        frame = _make_frame()
        payload = client._encode_raw(frame)

        decoded = np.frombuffer(
            base64.b64decode(payload["image_data"]),
            dtype=np.uint8,
        ).reshape(payload["shape"])
        np.testing.assert_array_equal(frame, decoded)
        client.close()


class TestEncodeJpeg:
    """_encode_jpeg のテスト."""

    def test_payload_format(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="jpeg")
        frame = _make_frame()
        payload = client._encode_jpeg(frame)

        assert payload["format"] == "jpeg"
        assert "shape" not in payload
        assert isinstance(payload["image_data"], str)
        client.close()

    def test_decodable(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="jpeg")
        frame = _make_frame()
        payload = client._encode_jpeg(frame)

        jpeg_bytes = base64.b64decode(payload["image_data"])
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[2] == 3
        client.close()


class TestBuildPayload:
    """_build_payload のテスト."""

    def test_jpeg_format(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="jpeg")
        frame = _make_frame()
        payload = client._build_payload(frame)
        assert payload["format"] == "jpeg"
        client.close()

    def test_raw_format(self):
        client = InferenceClient(base_url="http://localhost:8000", image_format="raw")
        frame = _make_frame()
        payload = client._build_payload(frame)
        assert payload["format"] == "raw"
        client.close()


class TestPredictResponse:
    """PredictResponse モデルのテスト."""

    def test_from_dict(self):
        data = {
            "class_id": 1,
            "class_name": "dog",
            "confidence": 0.85,
            "probabilities": [0.15, 0.85],
            "processing_time_ms": 10.5,
            "backend": "onnx",
        }
        resp = PredictResponse(**data)
        assert resp.class_id == 1
        assert resp.class_name == "dog"
        assert resp.confidence == 0.85
        assert resp.probabilities == [0.15, 0.85]
        assert resp.processing_time_ms == 10.5
        assert resp.backend == "onnx"

    def test_frozen(self):
        resp = PredictResponse(
            class_id=0,
            class_name="cat",
            confidence=0.9,
            probabilities=[0.9, 0.1],
            processing_time_ms=5.0,
            backend="onnx",
        )
        with pytest.raises(AttributeError):
            resp.class_name = "dog"  # type: ignore[misc]


class TestPredictConnectionError:
    """接続エラーのテスト."""

    def test_connection_refused(self):
        client = InferenceClient(
            base_url="http://127.0.0.1:19999",
            timeout=1.0,
        )
        with pytest.raises(InferenceConnectionError):
            client.predict(_make_frame())
        client.close()
