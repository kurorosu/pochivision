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
    sigma_color: StrictFloat = Field(alias="sigmaColor")
    sigma_space: StrictFloat = Field(alias="sigmaSpace")

    class Config:
        """Pydanticの設定クラス."""

        populate_by_name = True  # 名前またはエイリアスでフィールドにアクセス可能にする
        allow_population_by_field_name = True  # 後方互換性のため


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
    """CLAHE（適応的ヒストグラム平坦化）のパラメータスキーマ."""

    color_mode: Optional[StrictStr] = Field(default="gray", pattern="^(gray|lab|bgr)$")
    clip_limit: Optional[StrictFloat] = Field(default=2.0, gt=0)
    tile_grid_size: Optional[List[StrictInt]] = Field(
        default=[8, 8], min_items=2, max_items=2, each_item_gt=0
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
        default=[0, 0, 0], min_items=3, max_items=3
    )
    inside_color: Optional[List[StrictInt]] = Field(
        default=[255, 255, 255], min_items=3, max_items=3
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


class ConfigModel(BaseModel):
    """全体設定（カメラ一覧・選択インデックス）のスキーマ."""

    cameras: Dict[str, CameraProfile]
    selected_camera_index: StrictInt
    id_interval: Optional[StrictInt] = Field(default=None)
