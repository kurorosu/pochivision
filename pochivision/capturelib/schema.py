"""
設定ファイルのスキーマ (pydantic モデル) を定義するモジュール.

カメラプロファイルと全体設定の構造を型安全に管理する.
プロセッサパラメータは processors/schema.py を参照.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, StrictInt, StrictStr

from pochivision.processors.schema import (
    AverageBlurParams,
    BilateralFilterParams,
    CannyEdgeParams,
    CLAHEParams,
    ContourParams,
    EqualizeParams,
    GaussianAdaptiveBinarizationParams,
    GaussianBlurParams,
    GrayscaleParams,
    MeanAdaptiveBinarizationParams,
    MedianBlurParams,
    MotionBlurParams,
    OtsuBinarizationParams,
    ResizeParams,
    StandardBinarizationParams,
)


class CameraProfile(BaseModel):
    """カメラプロファイルのスキーマ."""

    width: StrictInt
    height: StrictInt
    fps: StrictInt
    backend: StrictStr
    processors: List[StrictStr]
    mode: StrictStr
    id_interval: Optional[StrictInt] = Field(default=None)

    gaussian_blur: Optional[GaussianBlurParams] = None
    average_blur: Optional[AverageBlurParams] = None
    median_blur: Optional[MedianBlurParams] = None
    grayscale: Optional[GrayscaleParams] = None
    std_bin: Optional[StandardBinarizationParams] = None
    otsu_bin: Optional[OtsuBinarizationParams] = None
    gauss_adapt_bin: Optional[GaussianAdaptiveBinarizationParams] = None
    mean_adapt_bin: Optional[MeanAdaptiveBinarizationParams] = None
    bilateral_filter: Optional[BilateralFilterParams] = None
    motion_blur: Optional[MotionBlurParams] = None
    resize: Optional[ResizeParams] = None
    equalize: Optional[EqualizeParams] = None
    clahe: Optional[CLAHEParams] = None
    canny_edge: Optional[CannyEdgeParams] = None
    contour: Optional[ContourParams] = None


class PreviewConfig(BaseModel):
    """ライブプレビューウィンドウの表示設定スキーマ."""

    width: StrictInt = Field(default=1280, gt=0)
    height: StrictInt = Field(default=720, gt=0)


class ConfigModel(BaseModel):
    """全体設定 (カメラ一覧・選択インデックス) のスキーマ."""

    cameras: Dict[str, CameraProfile]
    selected_camera_index: StrictInt
    id_interval: Optional[StrictInt] = Field(default=None)
    preview: Optional[PreviewConfig] = Field(default=None)
