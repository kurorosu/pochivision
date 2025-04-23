from typing import List, Optional, Dict
from pydantic import BaseModel, Field, StrictInt, StrictStr, StrictFloat


class GaussianBlurParams(BaseModel):
    kernel_size: List[StrictInt]
    sigma: StrictFloat


class AverageBlurParams(BaseModel):
    kernel_size: List[StrictInt]


class MedianBlurParams(BaseModel):
    kernel_size: StrictInt


class GrayscaleParams(BaseModel):
    pass  # パラメータなし


class StandardBinarizationParams(BaseModel):
    threshold: StrictInt


class BilateralFilterParams(BaseModel):
    d: StrictInt
    sigmaColor: StrictInt
    sigmaSpace: StrictInt


class MotionBlurParams(BaseModel):
    kernel_size: StrictInt
    angle: StrictFloat


class CameraProfile(BaseModel):
    width: StrictInt
    height: StrictInt
    fps: StrictInt
    backend: StrictStr
    processors: List[StrictStr]
    mode: StrictStr
    gaussian_blur: Optional[GaussianBlurParams] = None
    average_blur: Optional[AverageBlurParams] = None
    median_blur: Optional[MedianBlurParams] = None
    grayscale: Optional[GrayscaleParams] = None
    standard_binarization: Optional[StandardBinarizationParams] = None
    bilateral_filter: Optional[BilateralFilterParams] = None
    motion_blur: Optional[MotionBlurParams] = None


class ConfigModel(BaseModel):
    cameras: Dict[str, CameraProfile]
    selected_camera_index: StrictInt
