"""推論関連の例外クラスのテスト."""

import pytest

from pochivision.exceptions import (
    InferenceConnectionError,
    InferenceError,
    VisionCaptureError,
)


class TestInferenceError:
    """InferenceError のテスト."""

    def test_message(self):
        err = InferenceError("inference failed")
        assert str(err) == "inference failed"

    def test_inheritance(self):
        assert issubclass(InferenceError, VisionCaptureError)
        assert issubclass(InferenceError, RuntimeError)

    def test_raise_and_catch(self):
        with pytest.raises(InferenceError, match="inference"):
            raise InferenceError("inference failed")


class TestInferenceConnectionError:
    """InferenceConnectionError のテスト."""

    def test_message(self):
        err = InferenceConnectionError("connection refused")
        assert str(err) == "connection refused"

    def test_inheritance(self):
        assert issubclass(InferenceConnectionError, InferenceError)
        assert issubclass(InferenceConnectionError, ConnectionError)

    def test_raise_and_catch(self):
        with pytest.raises(InferenceConnectionError, match="connection"):
            raise InferenceConnectionError("connection refused")

    def test_catch_as_inference_error(self):
        with pytest.raises(InferenceError):
            raise InferenceConnectionError("connection refused")
