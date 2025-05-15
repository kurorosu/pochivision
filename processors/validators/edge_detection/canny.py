"""Cannyエッジ検出プロセッサーの設定バリデーターを定義します."""

from typing import Any, Dict

from ..base import BaseValidator, ProcessorValidationError

# from ..edge_detection import CannyEdgeProcessor # 循環参照を避けるためにコメントアウト


class CannyConfigValidator(BaseValidator):
    """CannyEdgeProcessor設定のバリデーター."""

    def __init__(self, config: Dict[str, Any]):
        """
        CannyConfigValidatorを初期化します.

        Args:
            config (Dict[str, Any]): 検証対象の設定.
        """
        self.config: Dict[str, Any] = config

    def validate(self) -> None:
        """
        Cannyエッジ検出設定を検証します.

        configにキーが存在する場合にのみ、その値を検証します.

        Raises:
            ProcessorValidationError: 設定が無効な場合.
        """
        if "threshold1" in self.config:
            threshold1 = self.config["threshold1"]
            if not isinstance(threshold1, (int, float)):
                raise ProcessorValidationError(
                    f"Canny 'threshold1' must be a number, got {threshold1}."
                )
            if threshold1 < 0:
                raise ProcessorValidationError(
                    f"Canny 'threshold1' must be non-negative. Got {threshold1}."
                )

        if "threshold2" in self.config:
            threshold2 = self.config["threshold2"]
            if not isinstance(threshold2, (int, float)):
                raise ProcessorValidationError(
                    f"Canny 'threshold2' must be a number, got {threshold2}."
                )
            if threshold2 < 0:
                raise ProcessorValidationError(
                    f"Canny 'threshold2' must be non-negative. Got {threshold2}."
                )

        if "threshold1" in self.config and "threshold2" in self.config:
            threshold1_val = self.config["threshold1"]
            threshold2_val = self.config["threshold2"]
            if (
                isinstance(threshold1_val, (int, float))
                and isinstance(threshold2_val, (int, float))
                and threshold1_val > threshold2_val
            ):
                raise ProcessorValidationError(
                    "Canny 'threshold1' should not be greater than 'threshold2'. "
                    f"Got threshold1={threshold1_val}, threshold2={threshold2_val}."
                )

        if "aperture_size" in self.config:
            aperture_size = self.config["aperture_size"]
            if not isinstance(aperture_size, int):
                raise ProcessorValidationError(
                    f"Canny 'aperture_size' must be an integer, got {aperture_size}."
                )
            if aperture_size not in [3, 5, 7]:
                raise ProcessorValidationError(
                    "Canny 'aperture_size' must be 3, 5, or 7. " f"Got {aperture_size}."
                )

        if "l2_gradient" in self.config:
            l2_gradient = self.config["l2_gradient"]
            if not isinstance(l2_gradient, bool):
                raise ProcessorValidationError(
                    f"Canny 'l2_gradient' must be a boolean, got {l2_gradient}."
                )
