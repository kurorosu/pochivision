"""
設定ファイルのスキーマ（pydanticモデル）を定義するモジュール.

各種画像処理パラメータやカメラプロファイル、全体設定の構造を型安全に管理します。
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr


class GaussianBlurParams(BaseModel):
    """ガウシアンブラーのパラメータスキーマ."""

    kernel_size: List[StrictInt]
    sigma: StrictFloat


class AverageBlurParams(BaseModel):
    """平均化ブラーのパラメータスキーマ."""

    kernel_size: List[StrictInt]


class MedianBlurParams(BaseModel):
    """メディアンブラーのパラメータスキーマ."""

    kernel_size: StrictInt


class GrayscaleParams(BaseModel):
    """グレースケール変換のパラメータスキーマ（パラメータなし）."""

    pass  # パラメータなし


class StandardBinarizationParams(BaseModel):
    """標準2値化のパラメータスキーマ."""

    threshold: StrictInt


class OtsuBinarizationParams(BaseModel):
    """大津の2値化のパラメータスキーマ（パラメータなし）."""

    pass  # パラメータなし


class GaussianAdaptiveBinarizationParams(BaseModel):
    """ガウシアン適応的2値化のパラメータスキーマ."""

    block_size: StrictInt
    c: StrictFloat


class MeanAdaptiveBinarizationParams(BaseModel):
    """平均適応的2値化のパラメータスキーマ."""

    block_size: StrictInt
    c: StrictFloat


class BilateralFilterParams(BaseModel):
    """バイラテラルフィルタのパラメータスキーマ."""

    d: StrictInt
    sigmaColor: StrictInt
    sigmaSpace: StrictInt


class MotionBlurParams(BaseModel):
    """モーションブラーのパラメータスキーマ."""

    kernel_size: StrictInt
    angle: StrictFloat


class ResizeParams(BaseModel):
    """リサイズプロセッサーのパラメータスキーマ."""

    width: Optional[StrictInt] = None
    height: Optional[StrictInt] = None
    preserve_aspect_ratio: Optional[bool] = Field(default=False)
    aspect_ratio_mode: Optional[StrictStr] = Field(
        default="width", pattern="^(width|height)$"
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


class ConfigModel(BaseModel):
    """全体設定（カメラ一覧・選択インデックス）のスキーマ."""

    cameras: Dict[str, CameraProfile]
    selected_camera_index: StrictInt
    id_interval: Optional[StrictInt] = Field(default=None)
