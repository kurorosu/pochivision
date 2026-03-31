"""
画像処理プロセッサの設定スキーマを定義するモジュール.

各プロセッサのパラメータ構造を型安全に管理する.
"""

from typing import List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


class GaussianBlurParams(BaseModel):
    """ガウシアンブラーのパラメータスキーマ."""

    kernel_size: List[StrictInt]
    sigma: StrictFloat = Field(ge=0)


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

    threshold: StrictInt = Field(ge=0, le=255)


class OtsuBinarizationParams(BaseModel):
    """大津の2値化のパラメータスキーマ."""

    pass


class GaussianAdaptiveBinarizationParams(BaseModel):
    """ガウシアン適応的2値化のパラメータスキーマ."""

    block_size: StrictInt = Field(ge=3)
    c: Union[StrictInt, StrictFloat]

    @field_validator("block_size")
    @classmethod
    def block_size_must_be_odd(cls, v: int) -> int:
        """block_size は奇数でなければならない."""
        if v % 2 == 0:
            raise ValueError(f"block_size must be odd, got {v}")
        return v


class MeanAdaptiveBinarizationParams(BaseModel):
    """平均適応的2値化のパラメータスキーマ."""

    block_size: StrictInt = Field(ge=3)
    c: Union[StrictInt, StrictFloat]

    @field_validator("block_size")
    @classmethod
    def block_size_must_be_odd(cls, v: int) -> int:
        """block_size は奇数でなければならない."""
        if v % 2 == 0:
            raise ValueError(f"block_size must be odd, got {v}")
        return v


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

    width: Optional[StrictInt] = Field(default=None, gt=0)
    height: Optional[StrictInt] = Field(default=None, gt=0)
    preserve_aspect_ratio: Optional[StrictBool] = Field(default=False)
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

    threshold1: StrictFloat = Field(ge=0)
    threshold2: StrictFloat = Field(ge=0)
    aperture_size: Optional[StrictInt] = Field(default=3, ge=3, le=7)
    l2_gradient: Optional[StrictBool] = Field(default=False)

    @field_validator("aperture_size")
    @classmethod
    def aperture_size_must_be_odd(cls, v: int) -> int:
        """aperture_size は奇数でなければならない."""
        if v is not None and v % 2 == 0:
            raise ValueError(f"aperture_size must be odd, got {v}")
        return v

    @model_validator(mode="after")
    def threshold1_le_threshold2(self) -> "CannyEdgeParams":
        """threshold1 <= threshold2 でなければならない."""
        if self.threshold1 > self.threshold2:
            raise ValueError(
                f"threshold1 ({self.threshold1}) must be <= "
                f"threshold2 ({self.threshold2})"
            )
        return self


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


class MaskCompositionParams(BaseModel):
    """マスク合成プロセッサのパラメータスキーマ."""

    target_image: Optional[StrictStr] = Field(default="original")
    use_white_pixels: Optional[StrictBool] = Field(default=True)
    enable_cropping: Optional[StrictBool] = Field(default=False)
    crop_margin: Optional[StrictInt] = Field(default=0, ge=0)


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
    "mask_composition": MaskCompositionParams,
}
