# processors/registry.py

"""
画像処理プロセッサのレジストリモジュール。

このモジュールでは、処理クラスを名前付きで登録し、
名前から対応するクラスを取得できるようにします。

使用例:
    @register_processor("grayscale")
    class GrayscaleProcessor(BaseProcessor):
        ...
"""

# 名前とクラスのマッピングを保持する辞書
PROCESSOR_REGISTRY = {}


def register_processor(name):
    """
    画像処理プロセッサクラスを名前付きで登録するためのデコレータ。

    Args:
        name (str): 登録するプロセッサの名前

    Returns:
        Callable: デコレートされたクラスをそのまま返す
    """
    def decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator
