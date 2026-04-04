"""
特徴量抽出器の設定スキーマを定義するモジュール.

各特徴量抽出器のパラメータ構造を型安全に管理します。
"""

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class BrightnessStatisticsParams(BaseModel):
    """輝度統計特徴量抽出のパラメータスキーマ."""

    color_mode: StrictStr | None = Field(default="gray", pattern="^(gray|lab_l|hsv_v)$")
    exclude_zero_pixels: StrictBool | None = Field(
        default=True, description="輝度値0のピクセルを計算から除外するかどうか"
    )


class RGBStatisticsParams(BaseModel):
    """RGB統計特徴量抽出のパラメータスキーマ."""

    exclude_black_pixels: StrictBool | None = Field(
        default=True, description="RGB値がすべて0のピクセルを計算から除外するかどうか"
    )


class HSVStatisticsParams(BaseModel):
    """HSV統計特徴量抽出のパラメータスキーマ."""

    exclude_black_pixels: StrictBool | None = Field(
        default=True, description="HSV値がすべて0のピクセルを計算から除外するかどうか"
    )


class GLCMTextureParams(BaseModel):
    """GLCMテクスチャ特徴量抽出のパラメータスキーマ."""

    distances: list[StrictInt] | None = Field(
        default=[1, 2, 3], description="ピクセル間距離のリスト"
    )
    angles: list[StrictFloat] | StrictStr | None = Field(
        default="standard",
        description=(
            "角度設定（度数リスト、ラジアンリスト、またはプリセット名: 'horizontal', "
            "'vertical', 'diagonal', 'standard', 'all_8', 'fine_16'）"
        ),
    )
    levels: StrictInt | None = Field(
        default=256, ge=2, le=256, description="グレーレベル数（2-256）"
    )
    symmetric: StrictBool | None = Field(
        default=True, description="対称性を考慮するかどうか"
    )
    normed: StrictBool | None = Field(default=True, description="正規化するかどうか")
    properties: list[StrictStr] | None = Field(
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
    resize_shape: list[StrictInt] | None = Field(
        default=None, description="リサイズ形状 [高さ, 幅]"
    )
    preserve_aspect_ratio: StrictBool | None = Field(
        default=True, description="アスペクト比を保持するか"
    )
    aspect_ratio_mode: StrictStr | None = Field(
        default="width", pattern="^(width|height)$", description="基準軸"
    )


class FFTFrequencyParams(BaseModel):
    """FFT周波数領域特徴量抽出のパラメータスキーマ."""

    frequency_bands: list[list[StrictFloat]] | None = Field(
        default=[[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]],
        description="周波数帯域のリスト（各帯域は[最小周波数, 最大周波数]の形式）",
    )
    high_low_threshold: StrictFloat | None = Field(
        default=0.2, ge=0.0, le=0.5, description="高周波/低周波の境界閾値（0.0-0.5）"
    )
    directional_tolerance: StrictFloat | None = Field(
        default=10.0,
        ge=0.0,
        le=90.0,
        description="方向性エネルギー計算の許容角度（度）",
    )
    peak_threshold_ratio: StrictFloat | None = Field(
        default=0.1, ge=0.0, le=1.0, description="ピーク検出の閾値比（0.0-1.0）"
    )
    mm_per_pixel: StrictFloat | None = Field(
        default=None,
        gt=0.0,
        description="ピクセルあたりのmm（Noneの場合はピクセル単位）",
    )
    resize_shape: list[StrictInt] | None = Field(
        default=None, description="リサイズ形状 [高さ, 幅]"
    )
    preserve_aspect_ratio: StrictBool | None = Field(
        default=True, description="アスペクト比を保持するか"
    )
    aspect_ratio_mode: StrictStr | None = Field(
        default="width", pattern="^(width|height)$", description="基準軸"
    )


class SWTFrequencyParams(BaseModel):
    """SWT周波数変換特徴量抽出のパラメータスキーマ."""

    wavelet: StrictStr | None = Field(
        default="db1",
        description="ウェーブレット種類（例: 'db1', 'db4', 'haar', 'bior2.2'）",
    )
    max_level: StrictInt | None = Field(
        default=1, ge=1, le=6, description="最大分解レベル（1-6）"
    )
    multiscale: StrictBool | None = Field(
        default=True,
        description=(
            "マルチスケール解析を行うかどうか. "
            "True: 各レベルの特徴量を抽出, "
            "False: level 1 (高周波) のみ抽出"
        ),
    )
    resize_shape: list[StrictInt] | None = Field(
        default=None, description="リサイズ形状 [高さ, 幅]"
    )
    preserve_aspect_ratio: StrictBool | None = Field(
        default=True, description="アスペクト比を保持するか"
    )
    aspect_ratio_mode: StrictStr | None = Field(
        default="width", pattern="^(width|height)$", description="基準軸"
    )


class LBPTextureParams(BaseModel):
    """LBPテクスチャ特徴量抽出のパラメータスキーマ."""

    P: StrictInt | None = Field(default=8, ge=4, le=24, description="近傍点数（4-24）")
    R: StrictInt | StrictFloat | None = Field(
        default=1, gt=0.0, description="半径（正の数値）"
    )
    method: StrictStr | None = Field(
        default="uniform",
        pattern="^(default|ror|uniform|nri_uniform|var)$",
        description="LBP手法（default, ror, uniform, nri_uniform, var）",
    )
    resize_shape: list[StrictInt] | None = Field(
        default=[128, 128],
        description="リサイズ形状 [高さ, 幅]（Noneの場合はリサイズしない）",
    )
    include_histogram: StrictBool | None = Field(
        default=False,
        description="ヒストグラムの各ビンを特徴量として含むかどうか",
    )
    preserve_aspect_ratio: StrictBool | None = Field(
        default=True, description="アスペクト比を保持するか"
    )
    aspect_ratio_mode: StrictStr | None = Field(
        default="width", pattern="^(width|height)$", description="基準軸"
    )


class HLACTextureParams(BaseModel):
    """HLACテクスチャ特徴量抽出のパラメータスキーマ."""

    order: StrictInt | None = Field(
        default=2, ge=1, le=2, description="自己相関の次数（1または2）"
    )
    rotate_invariant: StrictBool | None = Field(
        default=False, description="回転不変性を有効にするかどうか"
    )
    normalize: StrictBool | None = Field(
        default=True, description="特徴量を正規化するかどうか"
    )
    scales: list[StrictFloat] | None = Field(
        default=[1.0, 0.75, 0.5],
        description="マルチスケール処理のスケール係数リスト",
    )
    resize_shape: list[StrictInt] | None = Field(
        default=None,
        description="リサイズ形状 [高さ, 幅]（Noneの場合はリサイズしない）",
    )
    preserve_aspect_ratio: StrictBool | None = Field(
        default=True, description="アスペクト比を保持するか"
    )
    aspect_ratio_mode: StrictStr | None = Field(
        default="width", pattern="^(width|height)$", description="基準軸"
    )
    binarization_method: StrictStr | None = Field(
        default="adaptive",
        pattern="^(otsu|adaptive)$",
        description="二値化方式 (otsu or adaptive)",
    )
    adaptive_block_size: StrictInt | None = Field(
        default=11, ge=3, description="adaptive 二値化のブロックサイズ (奇数)"
    )
    adaptive_c: StrictInt | StrictFloat | None = Field(
        default=2, description="adaptive 二値化の定数 C"
    )


class CircleCounterParams(BaseModel):
    """円カウント特徴量抽出のパラメータスキーマ."""

    min_radius: StrictInt | None = Field(
        default=5, ge=1, description="検出する円の最小半径（ピクセル）"
    )
    max_radius: StrictInt | None = Field(
        default=0,
        ge=0,
        description="検出する円の最大半径（0の場合は画像サイズの1/4を使用）",
    )
    min_dist_ratio: StrictFloat | None = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="円の中心間の最小距離の係数（max_radius * ratio）",
    )
    param1: StrictInt | None = Field(
        default=50,
        ge=10,
        le=200,
        description=(
            "Canny検出器の高閾値（エッジ検出感度）。"
            "高い値（70-100）:ノイズに強く、明確なエッジのみ検出（検出漏れあり）。"
            "低い値（30-50）:細かいエッジも検出、ノイズの影響を受けやすい（偽検出あり）"
        ),
    )
    param2: StrictInt | None = Field(
        default=30,
        ge=10,
        le=100,
        description=(
            "HoughCircles蓄積器の閾値（円検出感度）。"
            "低い値（10-25）:多くの円を検出、偽陽性（誤検出）が増加。"
            "高い値（40-60）:厳格な円のみ検出、検出漏れが増加"
        ),
    )
    circularity_threshold: StrictFloat | None = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="真円度の閾値（0.0-1.0、1.0が完全な円）",
    )
    blur_kernel_size: StrictInt | None = Field(
        default=5,
        ge=0,
        description="ガウシアンブラーのカーネルサイズ（0で無効化、奇数のみ有効）",
    )
    enable_circularity_filter: StrictBool | None = Field(
        default=True, description="真円度フィルタリングを有効にするかどうか"
    )


# 抽出器名とスキーマクラスのマッピング
EXTRACTOR_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "brightness": BrightnessStatisticsParams,
    "rgb": RGBStatisticsParams,
    "hsv": HSVStatisticsParams,
    "glcm": GLCMTextureParams,
    "fft": FFTFrequencyParams,
    "swt": SWTFrequencyParams,
    "lbp": LBPTextureParams,
    "hlac": HLACTextureParams,
    "circle_counter": CircleCounterParams,
}
