"""DetectionClient のテスト."""

import base64
import time

import cv2
import httpx
import numpy as np
import pytest

from pochivision.exceptions import DetectionConnectionError, DetectionError
from pochivision.request.api.detection.client import DetectionClient
from pochivision.request.api.detection.models import DetectionResponse

_VALID_RESPONSE = {
    "detections": [
        {
            "class_id": 0,
            "class_name": "pochi",
            "confidence": 0.95,
            "bbox": [10.5, 20.2, 100.8, 200.3],
        },
        {
            "class_id": 1,
            "class_name": "pochi2",
            "confidence": 0.72,
            "bbox": [300.0, 120.0, 380.0, 250.0],
        },
    ],
    "e2e_time_ms": 12.3,
    "backend": "onnx",
}


def _make_frame(height: int = 48, width: int = 64) -> np.ndarray:
    """テスト用のフレームを生成する."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def raw_client():
    """Raw 形式の DetectionClient を提供する."""
    with DetectionClient(
        base_url="http://localhost:8000", image_format="raw"
    ) as client:
        yield client


@pytest.fixture
def jpeg_client():
    """JPEG 形式の DetectionClient を提供する."""
    with DetectionClient(
        base_url="http://localhost:8000", image_format="jpeg"
    ) as client:
        yield client


class TestInit:
    """__init__ のテスト."""

    def test_invalid_base_url_raises(self):
        with pytest.raises(ValueError, match="http://"):
            DetectionClient(base_url="localhost:8000")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="image_format"):
            DetectionClient(base_url="http://localhost:8000", image_format="png")

    def test_invalid_score_threshold_raises(self):
        with pytest.raises(ValueError, match="score_threshold"):
            DetectionClient(base_url="http://localhost:8000", score_threshold=1.5)

    def test_invalid_jpeg_quality_raises(self):
        with pytest.raises(ValueError, match="jpeg_quality"):
            DetectionClient(base_url="http://localhost:8000", jpeg_quality=0)

    def test_base_url_trailing_slash_stripped(self):
        with DetectionClient(base_url="http://localhost:8000/") as client:
            assert client.base_url == "http://localhost:8000"

    def test_score_threshold_boundary_accepted(self):
        for v in (0.0, 1.0):
            with DetectionClient(
                base_url="http://localhost:8000", score_threshold=v
            ) as client:
                assert client.score_threshold == v

    def test_jpeg_quality_boundary_accepted(self):
        for v in (1, 100):
            with DetectionClient(
                base_url="http://localhost:8000", jpeg_quality=v
            ) as client:
                assert client.jpeg_quality == v

    def test_jpeg_quality_over_100_raises(self):
        with pytest.raises(ValueError, match="jpeg_quality"):
            DetectionClient(base_url="http://localhost:8000", jpeg_quality=101)


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

        restored = np.frombuffer(
            base64.b64decode(payload["image_data"]), dtype=np.uint8
        ).reshape(frame.shape)
        np.testing.assert_array_equal(restored, frame)


class TestEncodeJpeg:
    """_encode_jpeg のテスト."""

    def test_payload_format(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._encode_jpeg(frame)

        assert payload["format"] == "jpeg"
        assert isinstance(payload["image_data"], str)

    def test_decodable(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._encode_jpeg(frame)

        buf = np.frombuffer(base64.b64decode(payload["image_data"]), dtype=np.uint8)
        restored = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert restored is not None
        assert restored.shape == frame.shape

    def test_encode_empty_frame_raises(self, jpeg_client):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(Exception):
            jpeg_client._encode_jpeg(empty)

    def test_quality_affects_output_size(self):
        frame = _make_frame(height=256, width=256)
        with DetectionClient(
            base_url="http://localhost:8000", image_format="jpeg", jpeg_quality=10
        ) as low:
            low_size = len(low._encode_jpeg(frame)["image_data"])
        with DetectionClient(
            base_url="http://localhost:8000", image_format="jpeg", jpeg_quality=100
        ) as high:
            high_size = len(high._encode_jpeg(frame)["image_data"])

        assert high_size > low_size


class TestBuildPayload:
    """_build_payload のテスト."""

    def test_jpeg_format_includes_score_threshold(self, jpeg_client):
        frame = _make_frame()
        payload = jpeg_client._build_payload(frame)

        assert payload["format"] == "jpeg"
        assert payload["score_threshold"] == jpeg_client.score_threshold

    def test_raw_format_includes_score_threshold(self, raw_client):
        frame = _make_frame()
        payload = raw_client._build_payload(frame)

        assert payload["format"] == "raw"
        assert payload["score_threshold"] == raw_client.score_threshold

    def test_custom_score_threshold_reflected(self):
        with DetectionClient(
            base_url="http://localhost:8000", score_threshold=0.3
        ) as client:
            payload = client._build_payload(_make_frame())
            assert payload["score_threshold"] == 0.3


class TestDetect:
    """detect のテスト."""

    def test_non_uint8_frame_raises(self):
        client = DetectionClient(base_url="http://localhost:8000")
        frame = np.zeros((48, 64, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="uint8"):
            client.detect(frame)

        client.close()

    def test_non_3channel_frame_raises(self):
        client = DetectionClient(base_url="http://localhost:8000")
        frame_2d = np.zeros((48, 64), dtype=np.uint8)
        frame_4ch = np.zeros((48, 64, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            client.detect(frame_2d)
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            client.detect(frame_4ch)

        client.close()

    def test_read_timeout_mapped_to_connection_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timeout", request=request)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionConnectionError):
            client.detect(_make_frame())

    def test_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        response = client.detect(_make_frame())

        assert isinstance(response, DetectionResponse)
        assert len(response.detections) == 2
        assert response.detections[0].class_name == "pochi"
        assert response.detections[0].bbox == (10.5, 20.2, 100.8, 200.3)
        assert response.e2e_time_ms == 12.3
        assert response.backend == "onnx"
        assert response.rtt_ms >= 0

    def test_empty_detections(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "detections": [],
                    "e2e_time_ms": 5.0,
                    "backend": "pytorch",
                },
            )

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        response = client.detect(_make_frame())

        assert response.detections == ()
        assert response.backend == "pytorch"

    def test_http_status_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "server error"})

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="status=500"):
            client.detect(_make_frame())

    def test_invalid_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="JSON"):
            client.detect(_make_frame())

    def test_missing_response_field(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"detections": []})

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="フォーマット"):
            client.detect(_make_frame())

    def test_detections_field_wrong_type(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "detections": "not-a-list",
                    "e2e_time_ms": 1.0,
                    "backend": "onnx",
                },
            )

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="フォーマット"):
            client.detect(_make_frame())

    def test_detection_item_not_dict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "detections": [123],
                    "e2e_time_ms": 1.0,
                    "backend": "onnx",
                },
            )

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="フォーマット"):
            client.detect(_make_frame())

    def test_malformed_detection_item(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "detections": [{"class_id": 0, "class_name": "x"}],
                    "e2e_time_ms": 1.0,
                    "backend": "onnx",
                },
            )

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        with pytest.raises(DetectionError, match="フォーマット"):
            client.detect(_make_frame())

    def test_connection_refused(self):
        client = DetectionClient(base_url="http://127.0.0.1:1", timeout=0.5)

        with pytest.raises(DetectionConnectionError):
            client.detect(_make_frame())

        client.close()

    def test_request_sent_to_detect_endpoint(self):
        captured_url: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_url.append(str(request.url))
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        client.detect(_make_frame())

        assert captured_url[0].endswith("/api/v1/detect")

    def test_score_threshold_sent_in_payload(self):
        captured_payload: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.append(request.read().decode())
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000", score_threshold=0.25)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        client.detect(_make_frame())

        assert '"score_threshold":0.25' in captured_payload[0].replace(" ", "")

    def test_phase_times_and_gpu_metrics_parsed(self):
        """phase_times_ms / GPU メトリクスをレスポンスから取り込むことを確認."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "detections": [],
                    "e2e_time_ms": 5.0,
                    "backend": "onnx-cuda",
                    "phase_times_ms": {
                        "pipeline_preprocess_ms": 1.2,
                        "pipeline_inference_ms": 2.5,
                        "pipeline_inference_gpu_ms": 2.1,
                        "pipeline_postprocess_ms": 0.8,
                    },
                    "gpu_clock_mhz": 1770,
                    "gpu_vram_used_mb": 2048,
                    "gpu_temperature_c": 55,
                },
            )

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        response = client.detect(_make_frame())

        assert response.phase_times_ms["pipeline_preprocess_ms"] == 1.2
        assert response.phase_times_ms["pipeline_inference_ms"] == 2.5
        assert response.phase_times_ms["pipeline_inference_gpu_ms"] == 2.1
        assert response.phase_times_ms["pipeline_postprocess_ms"] == 0.8
        assert response.gpu_clock_mhz == 1770
        assert response.gpu_vram_used_mb == 2048
        assert response.gpu_temperature_c == 55

    def test_phase_times_and_gpu_metrics_default_when_missing(self):
        """phase_times_ms / GPU メトリクスが欠落していても既存クライアントが壊れない."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        response = client.detect(_make_frame())

        assert response.phase_times_ms == {}
        assert response.gpu_clock_mhz is None
        assert response.gpu_vram_used_mb is None
        assert response.gpu_temperature_c is None

    def test_total_ms_greater_than_rtt_ms(self):
        """total_ms は RTT に encode + JSON parse を含むため rtt_ms 以上になる.

        MockTransport で固定遅延 (perf_counter ベース) を入れて RTT 計測区間を
        引き延ばし, total_ms が rtt_ms 以上であることを検証する.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            # Why: handler 内で busy-wait し RTT 計測区間に確実な遅延を載せる.
            # (time.sleep より perf_counter ベースの方がテスト計測と整合する)
            target = time.perf_counter() + 0.005
            while time.perf_counter() < target:
                pass
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        # 画像エンコード時間を確保するためそこそこ大きいフレームを使う.
        response = client.detect(_make_frame(height=480, width=640))

        assert response.rtt_ms > 0
        assert response.total_ms >= response.rtt_ms

    def test_total_ms_recorded_on_default_response(self):
        """既定のレスポンスでも total_ms が 0 より大きい値で記録される."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_VALID_RESPONSE)

        client = DetectionClient(base_url="http://localhost:8000")
        client._client = httpx.Client(transport=httpx.MockTransport(handler))

        response = client.detect(_make_frame())

        assert response.total_ms > 0
