"""輪郭抽出プロセッサーのバリデータを定義するモジュール."""

from typing import Any

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator
from pochivision.utils.image import to_grayscale


class ContourValidator(BaseValidator):
    """輪郭抽出処理のためのバリデータ."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        ContourValidatorのコンストラクタ.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.
        """
        self.config = config

    def is_binary_image(self, image: np.ndarray) -> bool:
        """
        画像が二値化画像かどうかをチェックする.

        Args:
            image (np.ndarray): チェックする画像

        Returns:
            bool: 二値化画像であればTrue、そうでなければFalse
        """
        # グレースケール画像に変換
        if image.ndim == 3:
            gray = to_grayscale(image)
        else:
            gray = image

        # ユニークな値を取得
        unique_values = np.unique(gray)

        # 厳密な二値画像は値が0と255の2つのみ
        # ノイズを考慮して、少し緩めのチェックを行う（値が3つ以下）
        return len(unique_values) <= 3

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証する.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        # 基本的な画像の型と空チェック
        self.validate_image_type_and_nonempty(image)

        # 画像は二値化されていることが望ましいが、
        # グレースケールや色付き画像も処理可能なので、チャンネル数に関する検証は緩めにする
        is_2d = image.ndim == 2
        has_valid_channels = False
        if image.ndim == 3:
            has_valid_channels = image.shape[2] in (1, 3, 4)

        if not (is_2d or has_valid_channels):
            msg_part1 = "Input image for ContourProcessor must be 2D grayscale "
            msg_part2 = "or 3/4 channel color image."
            raise ProcessorValidationError(msg_part1 + msg_part2)

    def validate_image_for_contour(self, image: np.ndarray) -> tuple[bool, str]:
        """
        輪郭抽出処理のための画像検証を行い、二値化画像かどうかを返す.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            tuple[bool, str]: 画像が適切かどうかのフラグとメッセージのタプル.
                              (True, "") なら問題なし、(False, "エラーメッセージ") なら問題あり
        """
        try:
            # 基本的な画像検証
            self.validate_image(image)

            # 二値化画像かチェック
            if not self.is_binary_image(image):
                return (
                    False,
                    "Input image is not binary. Contour detection "
                    "requires a binary image.",
                )

            return (True, "")
        except ProcessorValidationError as e:
            return (False, str(e))
