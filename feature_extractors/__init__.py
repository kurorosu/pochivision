"""
画像特徴量抽出パッケージ.

このパッケージは画像から様々な特徴量を抽出するための機能を提供します。
"""

from .base import BaseFeatureExtractor
from .brightness_statistics import BrightnessStatisticsExtractor
from .registry import (
    FEATURE_EXTRACTOR_REGISTRY,
    get_feature_extractor,
    register_feature_extractor,
)
from .rgb_statistics import RGBStatisticsExtractor
from .schema import BrightnessStatisticsParams, RGBStatisticsParams

__all__ = [
    "BaseFeatureExtractor",
    "BrightnessStatisticsExtractor",
    "RGBStatisticsExtractor",
    "FEATURE_EXTRACTOR_REGISTRY",
    "get_feature_extractor",
    "register_feature_extractor",
    "BrightnessStatisticsParams",
    "RGBStatisticsParams",
]
