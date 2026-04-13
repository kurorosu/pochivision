"""バリデータの基底クラスを定義するモジュール."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError


class BaseValidator(ABC):
    """
    すべてのバリデータの基底クラス.

    各プロセッサ用バリデータは, このクラスを継承して
    validate_config および validate_image メソッドを実装する必要があります.

    共通バリデーションメソッドも提供します.
    """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定辞書のバリデーションを実行する抽象メソッド.

        各サブクラスでオーバーライドして実装してください.
        検証が不要なプロセッサの場合は ``pass`` のみで構いません.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        pass

    @abstractmethod
    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像のバリデーションを実行する抽象メソッド.

        各サブクラスでオーバーライドして実装してください.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
        """
        pass

    @staticmethod
    def validate_image_type_and_nonempty(image: np.ndarray) -> None:
        """
        画像がnp.ndarray型かつ空でないことを検証する共通メソッド.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 型不正または空画像の場合.
        """
        if not isinstance(image, np.ndarray):
            raise ProcessorValidationError("image must be of type numpy.ndarray")
        if image.size == 0:
            raise ProcessorValidationError("input image is empty")
