"""exceptionsパッケージ:Vision Capture Coreの例外クラス群を提供します."""

from .base import VisionCaptureError
from .config import ConfigValidationError
from .processor import ProcessorRuntimeError, ProcessorValidationError

__all__ = [
    "VisionCaptureError",
    "ProcessorValidationError",
    "ProcessorRuntimeError",
    "ConfigValidationError",
]
