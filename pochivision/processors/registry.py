"""
プロセッサのレジストリ・登録機構を提供するモジュール.

このモジュールでは、処理クラスを名前付きで登録し、
名前から対応するクラスを取得できるようにします。

使用例:
    @register_processor("grayscale")
    class GrayscaleProcessor(BaseProcessor):
        ...
"""

from typing import Any, Callable

from pochivision.capturelib.log_manager import LogManager
from pochivision.exceptions import ProcessorRegistrationError

from .base import BaseProcessor
from .schema import PROCESSOR_SCHEMA_MAP

logger = LogManager().get_logger()

# 名前とクラスのマッピングを保持する辞書
PROCESSOR_REGISTRY: dict[str, type[BaseProcessor]] = {}


def register_processor(
    name: str,
    override: bool = False,
) -> Callable[[type[BaseProcessor]], type[BaseProcessor]]:
    """
    画像処理プロセッサクラスを名前付きで登録するためのデコレータ.

    Args:
        name (str): 登録するプロセッサの名前.
        override (bool): True の場合, 既存登録の上書きを許可する.
            False (デフォルト) の場合, 重複登録時に例外を送出する.

    Returns:
        Callable: デコレートされたクラスをそのまま返す.

    Raises:
        ProcessorRegistrationError: 同名のプロセッサが既に登録されており,
            ``override=False`` の場合.
    """

    def decorator(cls: type[BaseProcessor]) -> type[BaseProcessor]:
        if name in PROCESSOR_REGISTRY:
            existing = PROCESSOR_REGISTRY[name]
            if not override:
                raise ProcessorRegistrationError(
                    f"Processor '{name}' is already registered "
                    f"({existing.__name__}). "
                    f"Pass override=True to replace it with {cls.__name__}."
                )
            logger.warning(
                f"Processor '{name}' is already registered "
                f"({existing.__name__}), "
                f"overwriting with {cls.__name__} (override=True)"
            )
        PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_processor(name: str, config: dict[str, Any]) -> BaseProcessor:
    """
    指定された名前のプロセッサクラスを取得し、設定を使用してインスタンス化します.

    Args:
        name (str): 取得するプロセッサの名前.
        config (dict[str, Any]): プロセッサの初期化に使用する設定.

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
