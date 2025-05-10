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


class BilateralFilterParams(BaseModel):
    """バイラテラルフィルタのパラメータスキーマ."""

    d: StrictInt
    sigmaColor: StrictInt
    sigmaSpace: StrictInt


class MotionBlurParams(BaseModel):
    """モーションブラーのパラメータスキーマ."""

    kernel_size: StrictInt
    angle: StrictFloat


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
    standard_binarization: Optional[StandardBinarizationParams] = None
    bilateral_filter: Optional[BilateralFilterParams] = None
    motion_blur: Optional[MotionBlurParams] = None


class ConfigModel(BaseModel):
    """全体設定（カメラ一覧・選択インデックス）のスキーマ."""

    cameras: Dict[str, CameraProfile]
    selected_camera_index: StrictInt
    id_interval: Optional[StrictInt] = Field(default=None)
