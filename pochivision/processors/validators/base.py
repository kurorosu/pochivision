"""バリデータの基底クラスを定義するモジュール."""

from abc import ABC, abstractmethod

import numpy as np

from pochivision.exceptions import ProcessorValidationError


class BaseValidator(ABC):
    """
    すべてのバリデータの基底クラス.

    各プロセッサ用バリデータは、このクラスを継承して
    validate_config および validate_image メソッドを実装する必要があります.

    共通バリデーションメソッドも提供します.
    """

    @abstractmethod
    def validate_config(self) -> None:
        """
        設定値のバリデーションを実行する抽象メソッド.

        各サブクラスでオーバーライドして実装してください.

        Raises:
            ProcessorValidationError: 設定値が不正な場合.
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
