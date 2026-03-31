"""
画像処理プロセッサの設定スキーマを定義するモジュール.

各プロセッサのパラメータ構造を型安全に管理する.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


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
    """グレースケール変換のパラメータスキーマ."""

    pass


class StandardBinarizationParams(BaseModel):
    """標準2値化のパラメータスキーマ."""

    threshold: StrictInt


class OtsuBinarizationParams(BaseModel):
    """大津の2値化のパラメータスキーマ."""

    pass


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
    sigma_color: StrictFloat = Field(alias="sigmaColor")
    sigma_space: StrictFloat = Field(alias="sigmaSpace")

    model_config = ConfigDict(populate_by_name=True)


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


class EqualizeParams(BaseModel):
    """ヒストグラム平坦化のパラメータスキーマ."""

    color_mode: Optional[StrictStr] = Field(default="gray", pattern="^(gray|lab|bgr)$")


class CLAHEParams(BaseModel):
    """CLAHE のパラメータスキーマ."""

    color_mode: Optional[StrictStr] = Field(default="gray", pattern="^(gray|lab|bgr)$")
    clip_limit: Optional[StrictFloat] = Field(default=2.0, gt=0)
    tile_grid_size: Optional[List[StrictInt]] = Field(
        default=[8, 8], min_length=2, max_length=2
    )


class CannyEdgeParams(BaseModel):
    """Cannyエッジ検出のパラメータスキーマ."""

    threshold1: StrictFloat
    threshold2: StrictFloat
    aperture_size: Optional[StrictInt] = Field(default=3, ge=3, le=7)
    l2_gradient: Optional[bool] = Field(default=False)


class ContourParams(BaseModel):
    """輪郭抽出プロセッサのパラメータスキーマ."""

    retrieval_mode: Optional[StrictStr] = Field(
        default="list", pattern="^(external|list|ccomp|tree|floodfill)$"
    )
    approximation_method: Optional[StrictStr] = Field(
        default="simple", pattern="^(none|simple|tc89_l1|tc89_kcos)$"
    )
    min_area: Optional[StrictInt] = Field(default=100, ge=0)
    select_mode: Optional[StrictStr] = Field(default="rank", pattern="^(rank|all)$")
    contour_rank: Optional[StrictInt] = Field(default=0, ge=0)
    outside_color: Optional[List[StrictInt]] = Field(
        default=[0, 0, 0], min_length=3, max_length=3
    )
    inside_color: Optional[List[StrictInt]] = Field(
        default=[255, 255, 255], min_length=3, max_length=3
    )


# プロセッサ名とスキーマクラスのマッピング
PROCESSOR_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "gaussian_blur": GaussianBlurParams,
    "average_blur": AverageBlurParams,
    "median_blur": MedianBlurParams,
    "grayscale": GrayscaleParams,
    "std_bin": StandardBinarizationParams,
    "otsu_bin": OtsuBinarizationParams,
    "gauss_adapt_bin": GaussianAdaptiveBinarizationParams,
    "mean_adapt_bin": MeanAdaptiveBinarizationParams,
    "bilateral_filter": BilateralFilterParams,
    "motion_blur": MotionBlurParams,
    "resize": ResizeParams,
    "equalize": EqualizeParams,
    "clahe": CLAHEParams,
    "canny_edge": CannyEdgeParams,
    "contour": ContourParams,
}
