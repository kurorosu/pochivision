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

    各サブクラスは ``processor_name`` クラス属性 (プロセッサ登録名,
    例: ``"gaussian_blur"``) を定義し, エラーメッセージのプレフィックス
    ``[processor_name]`` として利用する. 未定義の場合は ``"unknown"`` が
    用いられる.
    """

    #: エラーメッセージのプレフィックスに用いるプロセッサ名.
    processor_name: str = "unknown"

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

    def _format_error(self, message: str) -> str:
        """
        エラーメッセージにプロセッサ名のプレフィックスを付与する.

        Args:
            message (str): 本体メッセージ.

        Returns:
            str: ``[processor_name] message`` 形式のメッセージ.
        """
        return f"[{self.processor_name}] {message}"

    def validate_image_type_and_nonempty(self, image: np.ndarray) -> None:
        """
        画像がnp.ndarray型かつ空でないことを検証する共通メソッド.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 型不正または空画像の場合.
        """
        if not isinstance(image, np.ndarray):
            raise ProcessorValidationError(
                self._format_error(
                    "image must be of type numpy.ndarray, "
                    f"got {type(image).__name__}"
                )
            )
        if image.size == 0:
            raise ProcessorValidationError(
                self._format_error(f"input image is empty, got shape {image.shape}")
            )
