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


class SWTFrequencyParams(BaseModel):
    """SWT周波数変換特徴量抽出のパラメータスキーマ."""

    wavelet: Optional[StrictStr] = Field(
        default="db1",
        description="ウェーブレット種類（例: 'db1', 'db4', 'haar', 'bior2.2'）",
    )
    max_level: Optional[StrictInt] = Field(
        default=1, ge=1, le=6, description="最大分解レベル（1-6）"
    )
    multiscale: Optional[StrictBool] = Field(
        default=True,
        description=(
            "マルチスケール解析を行うかどうか。"
            "Trueの場合は各レベルの特徴量を抽出、"
            "Falseの場合は最高レベル（最も詳細な分解レベル）のみ抽出"
        ),
    )


class LBPTextureParams(BaseModel):
    """LBPテクスチャ特徴量抽出のパラメータスキーマ."""

    P: Optional[StrictInt] = Field(
        default=8, ge=4, le=24, description="近傍点数（4-24）"
    )
    R: Optional[Union[StrictInt, StrictFloat]] = Field(
        default=1, gt=0.0, description="半径（正の数値）"
    )
    method: Optional[StrictStr] = Field(
        default="uniform",
        pattern="^(default|ror|uniform|nri_uniform|var)$",
        description="LBP手法（default, ror, uniform, nri_uniform, var）",
    )
    resize_shape: Optional[List[StrictInt]] = Field(
        default=[128, 128],
        description="リサイズ形状 [高さ, 幅]（Noneの場合はリサイズしない）",
    )
    include_histogram: Optional[StrictBool] = Field(
        default=False,
        description="ヒストグラムの各ビンを特徴量として含むかどうか",
    )


class HLACTextureParams(BaseModel):
    """HLACテクスチャ特徴量抽出のパラメータスキーマ."""

    order: Optional[StrictInt] = Field(
        default=2, ge=1, le=2, description="自己相関の次数（1または2）"
    )
    rotate_invariant: Optional[StrictBool] = Field(
        default=False, description="回転不変性を有効にするかどうか"
    )
    normalize: Optional[StrictBool] = Field(
        default=True, description="特徴量を正規化するかどうか"
    )
    scales: Optional[List[StrictFloat]] = Field(
        default=[1.0, 0.75, 0.5],
        description="マルチスケール処理のスケール係数リスト",
    )
    resize_shape: Optional[List[StrictInt]] = Field(
        default=None,
        description="リサイズ形状 [高さ, 幅]（Noneの場合はリサイズしない）",
    )
