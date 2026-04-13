"""exceptionsパッケージ:pochivisionの例外クラス群を提供します."""

from .base import VisionCaptureError
from .config import CameraConfigError, ConfigLoadError, ConfigValidationError
from .extractor import ExtractorRegistrationError, ExtractorValidationError
from .inference import InferenceConnectionError, InferenceError
from .processor import (
    ProcessorRegistrationError,
    ProcessorRuntimeError,
    ProcessorValidationError,
)

__all__ = [
    "VisionCaptureError",
    "ProcessorValidationError",
    "ProcessorRuntimeError",
    "ProcessorRegistrationError",
    "ConfigValidationError",
    "ConfigLoadError",
    "CameraConfigError",
    "ExtractorValidationError",
    "ExtractorRegistrationError",
    "InferenceError",
    "InferenceConnectionError",
]
