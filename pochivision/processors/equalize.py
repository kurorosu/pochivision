"""ヒストグラム平坦化プロセッサーを提供するモジュール."""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.validators.equalize.equalize import EqualizeInputValidator
from pochivision.utils.image import to_grayscale


@register_processor("equalize")
class EqualizeProcessor(BaseProcessor):
    """
    ヒストグラム平坦化を行うプロセッサー.

    このプロセッサーは、入力画像のヒストグラム平坦化を行います。
    カラー画像の場合は以下の処理方式から選択できます：
    - 'gray': グレースケールに変換してから処理を適用し、結果を再度BGRカラー形式に戻す（デフォルト）
    - 'lab': LAB色空間に変換して輝度（L）チャンネルのみを平坦化
    - 'bgr': BGR各チャンネルを個別に平坦化

    登録名:
        "equalize"

    設定例:
        {
            "equalize": {
                "color_mode": "lab"  # 'gray', 'lab', 'bgr'のいずれか
            }
        }
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        EqualizeProcessorの初期化.

        Args:
            name (str): プロセッサ名.
            config (Optional[Dict[str, Any]], optional): 設定パラメータ. デフォルトはNone.
                - color_mode (str): カラー画像の処理方式 ('gray', 'lab', 'bgr')
        """
        super().__init__(name, config or {})
        self.logger = logging.getLogger(__name__)
        self.validator = EqualizeInputValidator(self.config)
        self.validator.validate_config()
        self.color_mode = self.config.get("color_mode", "gray")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ヒストグラム平坦化処理を実行します.

        Args:
            image (np.ndarray): 入力画像(BGRまたはグレースケール).

        Returns:
            np.ndarray: ヒストグラム平坦化された画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: 処理中にエラーが発生した場合.
        """
        self.validator.validate_image(image)

        try:
            if len(image.shape) == 2:
                return cv2.equalizeHist(image)
            if image.shape[2] == 1:
                return cv2.equalizeHist(image.squeeze(axis=2))

            # カラー画像の処理
            if self.color_mode == "gray":
                gray = to_grayscale(image)
                equalized = cv2.equalizeHist(gray)
                return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

            elif self.color_mode == "lab":
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                equalized_l = cv2.equalizeHist(l)
                equalized_lab = cv2.merge([equalized_l, a, b])
                return cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2BGR)

            elif self.color_mode == "bgr":
                b, g, r = cv2.split(image)
                equalized_b = cv2.equalizeHist(b)
                equalized_g = cv2.equalizeHist(g)
                equalized_r = cv2.equalizeHist(r)
                return cv2.merge([equalized_b, equalized_g, equalized_r])

            else:
                # 未知のモード（バリデータでチェックされるはずなのでここには来ないはず）
                self.logger.warning(
                    f"Unknown color mode: {self.color_mode}, using 'gray'"
                )
                gray = to_grayscale(image)
                equalized = cv2.equalizeHist(gray)
                return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        except cv2.error as e:
            error_msg = f"Error during histogram equalization: {e}"
            self.logger.error(error_msg)
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            self.logger.error(error_msg)
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        ヒストグラム平坦化プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {"color_mode": "gray"}
