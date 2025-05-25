"""
特徴量抽出器の設定スキーマを定義するモジュール.

各特徴量抽出器のパラメータ構造を型安全に管理します。
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class BrightnessStatisticsParams(BaseModel):
    """輝度統計特徴量抽出のパラメータスキーマ."""

    color_mode: Optional[StrictStr] = Field(
        default="gray", pattern="^(gray|lab_l|hsv_v)$"
    )


class RGBStatisticsParams(BaseModel):
    """RGB統計特徴量抽出のパラメータスキーマ."""

    exclude_black_pixels: Optional[StrictBool] = Field(
        default=True, description="RGB値がすべて0のピクセルを計算から除外するかどうか"
    )


class HSVStatisticsParams(BaseModel):
    """HSV統計特徴量抽出のパラメータスキーマ."""

    exclude_black_pixels: Optional[StrictBool] = Field(
        default=True, description="HSV値がすべて0のピクセルを計算から除外するかどうか"
    )


class GLCMTextureParams(BaseModel):
    """GLCMテクスチャ特徴量抽出のパラメータスキーマ."""

    distances: Optional[List[StrictInt]] = Field(
        default=[1, 2, 3], description="ピクセル間距離のリスト"
    )
    angles: Optional[Union[List[StrictFloat], StrictStr]] = Field(
        default="standard",
        description=(
            "角度設定（度数リスト、ラジアンリスト、またはプリセット名: 'horizontal', "
            "'vertical', 'diagonal', 'standard', 'all_8', 'fine_16'）"
        ),
    )
    levels: Optional[StrictInt] = Field(
        default=256, ge=2, le=256, description="グレーレベル数（2-256）"
    )
    symmetric: Optional[StrictBool] = Field(
        default=True, description="対称性を考慮するかどうか"
    )
    normed: Optional[StrictBool] = Field(default=True, description="正規化するかどうか")
    properties: Optional[List[StrictStr]] = Field(
        default=[
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ],
        description="計算するプロパティのリスト",
    )


class FFTFrequencyParams(BaseModel):
    """FFT周波数領域特徴量抽出のパラメータスキーマ."""

    frequency_bands: Optional[List[List[StrictFloat]]] = Field(
        default=[[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]],
        description="周波数帯域のリスト（各帯域は[最小周波数, 最大周波数]の形式）",
    )
    high_low_threshold: Optional[StrictFloat] = Field(
        default=0.2, ge=0.0, le=0.5, description="高周波/低周波の境界閾値（0.0-0.5）"
    )
    directional_tolerance: Optional[StrictFloat] = Field(
        default=10.0,
        ge=0.0,
        le=90.0,
        description="方向性エネルギー計算の許容角度（度）",
    )
    peak_threshold_ratio: Optional[StrictFloat] = Field(
        default=0.1, ge=0.0, le=1.0, description="ピーク検出の閾値比（0.0-1.0）"
    )
    mm_per_pixel: Optional[StrictFloat] = Field(
        default=None,
        gt=0.0,
        description="ピクセルあたりのmm（Noneの場合はピクセル単位）",
    )
