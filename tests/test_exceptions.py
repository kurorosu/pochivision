"""例外クラスのテスト."""

import pytest

from pochivision.exceptions import (
    CameraConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ExtractorValidationError,
    ProcessorRuntimeError,
    ProcessorValidationError,
    VisionCaptureError,
)


class TestVisionCaptureError:
    """VisionCaptureError のテスト."""

    def test_message(self):
        err = VisionCaptureError("base error")
        assert str(err) == "base error"

    def test_inheritance(self):
        assert issubclass(VisionCaptureError, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(VisionCaptureError, match="base"):
            raise VisionCaptureError("base")


class TestConfigValidationError:
    """ConfigValidationError のテスト."""

    def test_message(self):
        err = ConfigValidationError("invalid config")
        assert str(err) == "invalid config"

    def test_inheritance(self):
        assert issubclass(ConfigValidationError, VisionCaptureError)
        assert issubclass(ConfigValidationError, ValueError)

    def test_raise_and_catch(self):
        with pytest.raises(ValueError, match="invalid"):
            raise ConfigValidationError("invalid")


class TestConfigLoadError:
    """ConfigLoadError のテスト."""

    def test_message(self):
        err = ConfigLoadError("file not found")
        assert str(err) == "file not found"

    def test_inheritance(self):
        assert issubclass(ConfigLoadError, VisionCaptureError)
        assert issubclass(ConfigLoadError, OSError)

    def test_raise_and_catch(self):
        with pytest.raises(OSError, match="file not found"):
            raise ConfigLoadError("file not found")


class TestCameraConfigError:
    """CameraConfigError のテスト."""

    def test_message(self):
        err = CameraConfigError("bad camera")
        assert str(err) == "bad camera"

    def test_inheritance(self):
        assert issubclass(CameraConfigError, VisionCaptureError)
        assert issubclass(CameraConfigError, ValueError)

    def test_raise_and_catch(self):
        with pytest.raises(VisionCaptureError, match="bad camera"):
            raise CameraConfigError("bad camera")


class TestProcessorValidationError:
    """ProcessorValidationError のテスト."""

    def test_message(self):
        err = ProcessorValidationError("invalid input")
        assert str(err) == "invalid input"

    def test_inheritance(self):
        assert issubclass(ProcessorValidationError, VisionCaptureError)
        assert issubclass(ProcessorValidationError, ValueError)

    def test_raise_and_catch(self):
        with pytest.raises(ValueError, match="invalid input"):
            raise ProcessorValidationError("invalid input")


class TestProcessorRuntimeError:
    """ProcessorRuntimeError のテスト."""

    def test_message(self):
        err = ProcessorRuntimeError("processing failed")
        assert str(err) == "processing failed"

    def test_inheritance(self):
        assert issubclass(ProcessorRuntimeError, VisionCaptureError)
        assert issubclass(ProcessorRuntimeError, RuntimeError)

    def test_raise_and_catch(self):
        with pytest.raises(RuntimeError, match="processing failed"):
            raise ProcessorRuntimeError("processing failed")


class TestExtractorValidationError:
    """ExtractorValidationError のテスト."""

    def test_message(self):
        err = ExtractorValidationError("bad image")
        assert str(err) == "bad image"

    def test_inheritance(self):
        assert issubclass(ExtractorValidationError, VisionCaptureError)
        assert issubclass(ExtractorValidationError, ValueError)

    def test_raise_and_catch(self):
        with pytest.raises(VisionCaptureError, match="bad image"):
            raise ExtractorValidationError("bad image")
