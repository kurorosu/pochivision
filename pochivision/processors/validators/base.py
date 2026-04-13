"""バリデータの基底クラスを定義するモジュール."""

from abc import ABC, abstractmethod

import numpy as np

from pochivision.exceptions import ProcessorValidationError


class BaseValidator(ABC):
    """すべてのバリデータの基底クラス.

    各プロセッサ用バリデータは, このクラスを継承して `validate_image` を実装する.
    共通バリデーションメソッドも提供する.

    Note:
        設定 (config) の検証は本クラスの責務ではなく, プロセッサスキーマ
        (`get_processor` 経由) に一本化されている. 過去に存在した
        `validate_config` 抽象メソッドは PR #238 で意図的に削除済み.
        config 検証ロジックを追加したい場合は, スキーマ側 (`schemas/`) を
        拡張すること. バリデータに `validate_config` を再追加してはならない.
    """

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
