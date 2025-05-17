"""CLAHE（適応的ヒストグラム平坦化）プロセッサーを提供するモジュール."""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from processors.validators.clahe.clahe import CLAHEInputValidator
from utils.image import to_grayscale


@register_processor("clahe")
class CLAHEProcessor(BaseProcessor):
    """
    CLAHE（Contrast Limited Adaptive Histogram Equalization）を行うプロセッサー.

    このプロセッサーは、画像のコントラストを局所的に適応的に強調します.
    カラー画像の場合は以下の処理方式から選択できます：
    - 'gray': グレースケールに変換してから処理を適用し、結果を再度BGRカラー形式に戻す（デフォルト）
    - 'lab': LAB色空間に変換して輝度（L）チャンネルのみを平坦化
    - 'bgr': BGR各チャンネルを個別に平坦化

    登録名:
        "clahe"

    設定例:
        {
            "clahe": {
                "color_mode": "lab",
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8]
            }
        }
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        CLAHEProcessorの初期化.

        Args:
            name (str): プロセッサ名.
            config (Optional[Dict[str, Any]], optional): 設定パラメータ. デフォルトはNone.
                - color_mode (str): カラー画像の処理方式 ('gray', 'lab', 'bgr')
                - clip_limit (float): コントラスト制限値
                - tile_grid_size (List[int]): タイルグリッドのサイズ
        """
        super().__init__(name, config or {})
        self.logger = logging.getLogger(__name__)
        self.validator = CLAHEInputValidator(self.config)
        self.validator.validate_config()

        # デフォルト値の設定
        self.color_mode = self.config.get("color_mode", "gray")
        self.clip_limit = float(self.config.get("clip_limit", 2.0))
        self.tile_grid_size = tuple(self.config.get("tile_grid_size", [8, 8]))

        # CLAHEインスタンスの作成
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE処理を実行します.

        Args:
            image (np.ndarray): 入力画像.

        Returns:
            np.ndarray: CLAHE適用後の画像.

        Raises:
            ProcessorRuntimeError: 入力画像の検証に失敗した場合.
        """
        try:
            self.validator.validate_image(image)
        except Exception as e:
            raise ProcessorRuntimeError(f"CLAHE image validation failed: {e}")

        try:
            # 画像がカラーかグレースケールかを判定
            if len(image.shape) > 2 and image.shape[2] > 1:
                # カラー画像の場合
                if self.color_mode == "gray":
                    # グレースケール変換方式
                    gray = to_grayscale(image)
                    equalized = self.clahe.apply(gray)
                    result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                    self.logger.info(
                        "Applied CLAHE to color image using grayscale conversion"
                    )
                elif self.color_mode == "lab":
                    # LAB色空間方式
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    # L（輝度）チャンネルのみ平坦化
                    lab_planes = list(cv2.split(lab))
                    lab_planes[0] = self.clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    # 元の色空間に戻す
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    self.logger.info(
                        "Applied CLAHE to color image using LAB color space"
                    )
                elif self.color_mode == "bgr":
                    # BGR各チャンネル個別方式
                    b, g, r = cv2.split(image)
                    b_eq = self.clahe.apply(b)
                    g_eq = self.clahe.apply(g)
                    r_eq = self.clahe.apply(r)
                    result = cv2.merge([b_eq, g_eq, r_eq])
                    self.logger.info("Applied CLAHE to color image per BGR channel")
                else:
                    # 不明なモードの場合はデフォルト（グレースケール）を使用
                    self.logger.warning(
                        f"Unknown color_mode '{self.color_mode}', "
                        f"falling back to 'gray'"
                    )
                    gray = to_grayscale(image)
                    equalized = self.clahe.apply(gray)
                    result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            else:
                # グレースケール画像の場合、そのまま平坦化
                result = self.clahe.apply(image)
                self.logger.info("Applied CLAHE to grayscale image")

            return result
        except Exception as e:
            self.logger.error("CLAHE processing failed: %s", str(e))
            raise ProcessorRuntimeError(f"CLAHE processing failed: {e}")
