"""
プロセッサのレジストリ・登録機構を提供するモジュール.

このモジュールでは、処理クラスを名前付きで登録し、
名前から対応するクラスを取得できるようにします。

使用例:
    @register_processor("grayscale")
    class GrayscaleProcessor(BaseProcessor):
        ...
"""

from typing import Any, Callable, Dict, Type

from .base import BaseProcessor
from .schema import PROCESSOR_SCHEMA_MAP

# 名前とクラスのマッピングを保持する辞書
PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {}


def register_processor(
    name: str,
) -> Callable[[Type[BaseProcessor]], Type[BaseProcessor]]:
    """
    画像処理プロセッサクラスを名前付きで登録するためのデコレータ.

    Args:
        name (str): 登録するプロセッサの名前.

    Returns:
        Callable: デコレートされたクラスをそのまま返す.
    """

    def decorator(cls: Type[BaseProcessor]) -> Type[BaseProcessor]:
        PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_processor(name: str, config: Dict[str, Any]) -> BaseProcessor:
    """
    指定された名前のプロセッサクラスを取得し、設定を使用してインスタンス化します.

    Args:
        name (str): 取得するプロセッサの名前.
        config (Dict[str, Any]): プロセッサの初期化に使用する設定.

    Returns:
        BaseProcessor: 指定されたプロセッサのインスタンス.

    Raises:
        ValueError: 指定された名前のプロセッサが見つからない場合.
    """
    processor_class = PROCESSOR_REGISTRY.get(name)
    if not processor_class:
        raise ValueError(f"Processor not found: {name}")

    # スキーマによる設定バリデーション
    if config:
        schema_class = PROCESSOR_SCHEMA_MAP.get(name)
        if schema_class:
            try:
                schema_class(**config)
            except Exception as e:
                raise ValueError(f"Invalid config for processor '{name}': {e}") from e

    return processor_class(name=name, config=config)
