"""exceptionsパッケージ:pochivisionの例外クラス群を提供します."""

from .base import VisionCaptureError
from .config import CameraConfigError, ConfigLoadError, ConfigValidationError
from .extractor import ExtractorValidationError
from .inference import InferenceConnectionError, InferenceError
from .processor import ProcessorRuntimeError, ProcessorValidationError

__all__ = [
    "VisionCaptureError",
    "ProcessorValidationError",
    "ProcessorRuntimeError",
    "ConfigValidationError",
    "ConfigLoadError",
    "CameraConfigError",
    "ExtractorValidationError",
    "InferenceError",
    "InferenceConnectionError",
]
