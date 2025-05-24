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
            "asm",
        ],
        description="計算するプロパティのリスト",
    )
