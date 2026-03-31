"""グレースケール変換プロセッサの実装を提供するモジュール."""

from typing import Any, Dict

import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.validators.grayscale import GrayscaleValidator
from pochivision.utils.image import to_grayscale


@register_processor("grayscale")
class GrayscaleProcessor(BaseProcessor):
    """
    グレースケール変換を行う画像処理プロセッサ.

    このプロセッサは、カラー画像（BGR）をグレースケール画像に変換します.
    設定項目は特に不要で、変換のみを行います.

    登録名:
        "grayscale"

    設定例:
        {
            "grayscale": {}
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        GrayscaleProcessorを初期化.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.
        """
        super().__init__(name, config)
        self.validator = GrayscaleValidator(config)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        グレースケール変換を実行します.

        Args:
            image (np.ndarray): 入力画像(BGRまたはグレースケール).

        Returns:
            np.ndarray: グレースケールに変換された画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: 画像変換に失敗した場合.
        """
        self.validator.validate_image(image)

        try:
            return to_grayscale(image)
        except Exception as e:
            error_msg = f"Error during grayscale conversion: {e}"
            # ログ出力など必要に応じて追加
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        グレースケール変換プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定（空の辞書）.
        """
        return {}
