"""輪郭抽出プロセッサーのバリデータを定義するモジュール."""

from typing import Any, Dict, List, Tuple

import numpy as np

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator
from pochivision.utils.image import to_grayscale


class ContourValidator(BaseValidator):
    """輪郭抽出処理のためのバリデータ."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        ContourValidatorのコンストラクタ.

        Args:
            config (Dict[str, Any]): バリデーション対象の設定辞書.
        """
        self.config = config

    def validate_config(self) -> None:
        """
        輪郭抽出設定を検証する.

        Raises:
            ProcessorValidationError: 設定が無効な場合.
        """
        # retrieval_mode の検証
        if "retrieval_mode" in self.config:
            retrieval_mode = self.config["retrieval_mode"]
            if not isinstance(retrieval_mode, str):
                error_msg = (
                    f"Contour 'retrieval_mode' must be a string, "
                    f"got {retrieval_mode}."
                )
                raise ProcessorValidationError(error_msg)

            valid_modes = ["external", "list", "ccomp", "tree", "floodfill"]
            if retrieval_mode not in valid_modes:
                err_msg = (
                    f"Contour 'retrieval_mode' must be one of {valid_modes}, "
                    f"got {retrieval_mode}."
                )
                raise ProcessorValidationError(err_msg)

        # approximation_method の検証
        if "approximation_method" in self.config:
            approx_method = self.config["approximation_method"]
            if not isinstance(approx_method, str):
                err_msg = (
                    f"Contour 'approximation_method' must be a string, "
                    f"got {approx_method}."
                )
                raise ProcessorValidationError(err_msg)

            valid_methods = ["none", "simple", "tc89_l1", "tc89_kcos"]
            if approx_method not in valid_methods:
                err_msg = (
                    f"Contour 'approximation_method' must be one of {valid_methods}, "
                    f"got {approx_method}."
                )
                raise ProcessorValidationError(err_msg)

        # min_area の検証
        if "min_area" in self.config:
            min_area = self.config["min_area"]
            if not isinstance(min_area, (int, float)):
                raise ProcessorValidationError(
                    f"Contour 'min_area' must be a number, got {min_area}."
                )
            if min_area < 0:
                raise ProcessorValidationError(
                    f"Contour 'min_area' must be non-negative, got {min_area}."
                )

        # select_mode の検証
        if "select_mode" in self.config:
            select_mode = self.config["select_mode"]
            if not isinstance(select_mode, str):
                raise ProcessorValidationError(
                    f"Contour 'select_mode' must be a string, got {select_mode}."
                )

            valid_modes = ["rank", "all"]
            if select_mode not in valid_modes:
                err_msg = (
                    f"Contour 'select_mode' must be one of {valid_modes}, "
                    f"got {select_mode}."
                )
                raise ProcessorValidationError(err_msg)

        # contour_rank の検証
        if "contour_rank" in self.config:
            contour_rank = self.config["contour_rank"]
            if not isinstance(contour_rank, int):
                raise ProcessorValidationError(
                    f"Contour 'contour_rank' must be an integer, got {contour_rank}."
                )
            if contour_rank < 0:
                raise ProcessorValidationError(
                    f"Contour 'contour_rank' must be non-negative, got {contour_rank}."
                )

        # inside_color の検証
        if "inside_color" in self.config:
            inside_color = self.config["inside_color"]
            self._validate_color("inside_color", inside_color)

        # outside_color の検証
        if "outside_color" in self.config:
            outside_color = self.config["outside_color"]
            self._validate_color("outside_color", outside_color)

    def _validate_color(self, param_name: str, color_value: Any) -> None:
        """
        色の値を検証する.

        Args:
            param_name (str): パラメータ名.
            color_value (Any): 検証する色の値.

        Raises:
            ProcessorValidationError: 色の値が無効な場合.
        """
        if not isinstance(color_value, List):
            raise ProcessorValidationError(
                f"Contour '{param_name}' must be a list, got {type(color_value)}."
            )

        if len(color_value) != 3:
            raise ProcessorValidationError(
                f"Contour '{param_name}' must be a list of 3 integers, "
                f"got {len(color_value)} elements."
            )

        for value in color_value:
            if not isinstance(value, int):
                raise ProcessorValidationError(
                    f"Contour '{param_name}' elements must be integers, "
                    f"got {type(value)}."
                )
            if value < 0 or value > 255:
                raise ProcessorValidationError(
                    f"Contour '{param_name}' values must be between 0 and 255, "
                    f"got {value}."
                )

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

    def validate_image_for_contour(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        輪郭抽出処理のための画像検証を行い、二値化画像かどうかを返す.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            Tuple[bool, str]: 画像が適切かどうかのフラグとメッセージのタプル.
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
