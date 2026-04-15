"""detection パッケージ: pochidetection 検出 API との連携機能を提供します."""

from .client import DetectionClient
from .config import DetectConfig, load_detect_config
from .models import Detection, DetectionResponse

__all__ = [
    "DetectConfig",
    "Detection",
    "DetectionClient",
    "DetectionResponse",
    "load_detect_config",
]
