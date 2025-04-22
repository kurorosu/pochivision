from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from exceptions import ProcessorValidationError


class BaseValidator(ABC):
    """
    すべてのバリデータの基底クラス。
    共通バリデーションメソッドも提供する。
    """

    @abstractmethod
    def validate(self) -> None:
        """
        バリデーションを実行する抽象メソッド。

        Raises:
            ProcessorValidationError: バリデーションに失敗した場合
        """
        pass

    @staticmethod
    def validate_image_type_and_nonempty(image: Any) -> None:
        """
        画像がnp.ndarray型かつ空でないことを検証する共通メソッド。

        Args:
            image (Any): 入力画像

        Raises:
            ProcessorValidationError: 型不正または空画像の場合
        """
        if not isinstance(image, np.ndarray):
            raise ProcessorValidationError(
                "image must be of type numpy.ndarray")
        if image.size == 0:
            raise ProcessorValidationError("input image is empty")
