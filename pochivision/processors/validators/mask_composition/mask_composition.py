"""マスク合成プロセッサのバリデータを定義."""

from typing import Any

import cv2
import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator


class MaskCompositionValidator(BaseValidator):
    """マスク合成プロセッサの設定と入力画像を検証するバリデータ."""

    def __init__(self, config: dict[str, Any]):
        """
        MaskCompositionValidatorを初期化.

        Args:
            config (dict[str, Any]): 設定パラメータ.

        Raises:
            ProcessorValidationError: 設定が不正な場合.
        """
        self.config = config
        self.validate_config(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """
        設定のバリデーションを実行する.

        ``crop_margin`` は 0 以上の int でなければならない.

        Args:
            config (dict[str, Any]): バリデーション対象の設定辞書.

        Raises:
            ProcessorValidationError: ``crop_margin`` が不正な場合.
        """
        if "crop_margin" in config and config["crop_margin"] is not None:
            crop_margin = config["crop_margin"]
            if not isinstance(crop_margin, int) or isinstance(crop_margin, bool):
                raise ProcessorValidationError(
                    f"crop_margin must be an int, got {crop_margin!r}"
                )
            if crop_margin < 0:
                raise ProcessorValidationError(
                    f"crop_margin must be >= 0, got {crop_margin}"
                )

    def validate_image(self, image: np.ndarray) -> None:
        """
        入力画像を検証.

        Args:
            image (np.ndarray): 入力画像.

        Raises:
            ProcessorValidationError: 入力画像が無効な場合.
        """
        # 基本的な画像のバリデーション
        self._validate_image_basics(image)

        # 2値画像かどうかのチェック
        self._validate_binary_image(image)

    def _validate_image_basics(self, image: np.ndarray) -> None:
        """
        画像の基本的な検証を行う.

        Args:
            image (np.ndarray): 検証する画像.

        Raises:
            ProcessorValidationError: 画像が無効な場合.
        """
        # 画像がnumpy配列かチェック
        if not isinstance(image, np.ndarray):
            raise ProcessorValidationError("image must be of type numpy.ndarray")

        # 画像が空でないか確認
        if image.size == 0:
            raise ProcessorValidationError("input image is empty")

        # 画像の次元が正しいか確認
        if len(image.shape) not in [2, 3]:
            raise ProcessorValidationError(
                "Input image for MaskComposition must be 2D "
                "grayscale or 3-channel color"
            )

        # 3チャンネルの場合はRGBかチェック
        if len(image.shape) == 3 and image.shape[2] != 3:
            raise ProcessorValidationError("Input color image must have 3 channels")

    def _validate_binary_image(self, image: np.ndarray) -> None:
        """
        画像が2値画像かどうかを検証.

        Args:
            image (np.ndarray): 検証する画像.

        Raises:
            ProcessorValidationError: 画像が2値画像でない場合.
        """
        # グレースケール画像の場合
        if len(image.shape) == 2:
            unique_values = np.unique(image)
            if not (
                np.array_equal(unique_values, [0, 255])
                or np.array_equal(unique_values, [0])
                or np.array_equal(unique_values, [255])
            ):
                raise ProcessorValidationError(
                    "Input image for MaskComposition must be a "
                    "binary image (only 0 and 255 values)"
                )
        # カラー画像の場合
        elif len(image.shape) == 3:
            # 全チャネルが同じ値か確認
            b, g, r = cv2.split(image)
            if not (np.array_equal(b, g) and np.array_equal(g, r)):
                raise ProcessorValidationError(
                    "For color images, all channels must have "
                    "identical values for binary image"
                )
            # 値が0か255のみか確認
            unique_values = np.unique(b)
            if not (
                np.array_equal(unique_values, [0, 255])
                or np.array_equal(unique_values, [0])
                or np.array_equal(unique_values, [255])
            ):
                raise ProcessorValidationError(
                    "Input image for MaskComposition must be a "
                    "binary image (only 0 and 255 values)"
                )
