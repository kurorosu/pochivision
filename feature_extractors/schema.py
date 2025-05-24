"""
特徴量抽出器の設定スキーマを定義するモジュール.

各特徴量抽出器のパラメータ構造を型安全に管理します。
"""

from typing import Optional

from pydantic import BaseModel, Field, StrictBool, StrictStr


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
