"""
設定ファイルのスキーマ (pydantic モデル) を定義するモジュール.

カメラプロファイルと全体設定の構造を型安全に管理する.
プロセッサパラメータは processors/schema.py を参照.
"""

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
    processors: list[StrictStr]
    mode: StrictStr
    id_interval: StrictInt | None = Field(default=None)

    gaussian_blur: GaussianBlurParams | None = None
    average_blur: AverageBlurParams | None = None
    median_blur: MedianBlurParams | None = None
    grayscale: GrayscaleParams | None = None
    std_bin: StandardBinarizationParams | None = None
    otsu_bin: OtsuBinarizationParams | None = None
    gauss_adapt_bin: GaussianAdaptiveBinarizationParams | None = None
    mean_adapt_bin: MeanAdaptiveBinarizationParams | None = None
    bilateral_filter: BilateralFilterParams | None = None
    motion_blur: MotionBlurParams | None = None
    resize: ResizeParams | None = None
    equalize: EqualizeParams | None = None
    clahe: CLAHEParams | None = None
    canny_edge: CannyEdgeParams | None = None
    contour: ContourParams | None = None


class PreviewConfig(BaseModel):
    """ライブプレビューウィンドウの表示設定スキーマ."""

    width: StrictInt = Field(default=1280, gt=0)
    height: StrictInt = Field(default=720, gt=0)


class ConfigModel(BaseModel):
    """全体設定 (カメラ一覧・選択インデックス) のスキーマ."""

    cameras: dict[str, CameraProfile]
    selected_camera_index: StrictInt
    id_interval: StrictInt | None = Field(default=None)
    preview: PreviewConfig | None = Field(default=None)
