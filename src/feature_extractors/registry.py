"""
特徴量抽出器のレジストリ・登録機構を提供するモジュール.

このモジュールでは、特徴量抽出クラスを名前付きで登録し、
名前から対応するクラスを取得できるようにします。

使用例:
    @register_feature_extractor("brightness_statistics")
    class BrightnessStatisticsExtractor(BaseFeatureExtractor):
        ...
"""

from typing import Any, Callable, Dict, Optional, Type

from .base import BaseFeatureExtractor

# 名前とクラスのマッピングを保持する辞書
FEATURE_EXTRACTOR_REGISTRY: Dict[str, Type[BaseFeatureExtractor]] = {}


def register_feature_extractor(
    name: str,
) -> Callable[[Type[BaseFeatureExtractor]], Type[BaseFeatureExtractor]]:
    """
    特徴量抽出器クラスを名前付きで登録するためのデコレータ.

    Args:
        name (str): 登録する特徴量抽出器の名前.

    Returns:
        Callable: デコレートされたクラスをそのまま返す.
    """

    def decorator(cls: Type[BaseFeatureExtractor]) -> Type[BaseFeatureExtractor]:
        FEATURE_EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_feature_extractor(
    name: str, config: Optional[Dict[str, Any]] = None
) -> BaseFeatureExtractor:
    """
    指定された名前の特徴量抽出器クラスを取得し、設定を使用してインスタンス化します.

    デフォルト設定とユーザー設定を自動的にマージします.

    Args:
        name (str): 取得する特徴量抽出器の名前.
        config (Optional[Dict[str, Any]]): 特徴量抽出器の初期化に使用する設定.
            指定されない場合はデフォルト設定のみを使用.

    Returns:
        BaseFeatureExtractor: 指定された特徴量抽出器のインスタンス.

    Raises:
        ValueError: 指定された名前の特徴量抽出器が見つからない場合.
    """
    extractor_class = FEATURE_EXTRACTOR_REGISTRY.get(name)
    if not extractor_class:
        raise ValueError(f"Feature extractor not found: {name}")

    # デフォルト設定の取得とマージはBaseFeatureExtractorで行われる
    return extractor_class(name=name, config=config or {})
