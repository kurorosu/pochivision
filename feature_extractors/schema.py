"""
特徴量抽出器の設定スキーマを定義するモジュール.

各特徴量抽出器のパラメータ構造を型安全に管理します。
"""

from typing import Optional

from pydantic import BaseModel, Field, StrictStr


class BrightnessStatisticsParams(BaseModel):
    """輝度統計特徴量抽出のパラメータスキーマ."""

    color_mode: Optional[StrictStr] = Field(
        default="gray", pattern="^(gray|lab_l|hsv_v)$"
    )
