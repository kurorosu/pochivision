"""CLAHE（適応的ヒストグラム平坦化）プロセッサーを提供するモジュール."""

from typing import Any

import cv2
import numpy as np

from pochivision.capturelib.log_manager import LogManager
from pochivision.exceptions import ProcessorRuntimeError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.validators.clahe import CLAHEInputValidator
from pochivision.utils.image import to_grayscale


@register_processor("clahe")
class CLAHEProcessor(BaseProcessor):
    """
    CLAHE（Contrast Limited Adaptive Histogram Equalization）処理を行うプロセッサー.

    このプロセッサーは、入力画像に対して適応的ヒストグラム平坦化を適用します。
    カラー画像の場合は以下の処理方式から選択できます：
    - 'gray': グレースケールに変換してから処理を適用し、結果を再度BGRカラー形式に戻す（デフォルト）
    - 'lab': LAB色空間に変換して輝度（L）チャンネルのみを平坦化
    - 'bgr': BGR各チャンネルを個別に平坦化

    Note:
        OpenCV の CLAHE オブジェクトは `__init__` 時に `cv2.createCLAHE()` で生成してキャッシュしている.
        他の多くのプロセッサは `process()` 呼び出しごとに OpenCV 関数を呼ぶため属性の書き換えで
        パラメータ変更が反映されるが, 本プロセッサはキャッシュ済みオブジェクトが使われるため
        `clip_limit` / `tile_grid_size` 属性を書き換えても反映されない. 動的な変更には
        `update_params()` を呼び出し内部オブジェクトを再生成する必要がある.

    登録名:
        "clahe"

    設定例:
        {
            "clahe": {
                "color_mode": "lab",  # 'gray', 'lab', 'bgr'のいずれか
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8]
            }
        }
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """
        CLAHEProcessorの初期化.

        Args:
            name (str): プロセッサ名.
            config (dict[str, Any] | None, optional): 設定パラメータ. デフォルトはNone.
                - color_mode (str): カラー画像の処理方式 ('gray', 'lab', 'bgr')
                - clip_limit (float): コントラスト制限値
                - tile_grid_size (List[int]): タイルグリッドのサイズ
        """
        super().__init__(name, config or {})
        self.logger = LogManager().get_logger()
        self.validator = CLAHEInputValidator(self.config)

        self.color_mode = self.config.get("color_mode", "gray")
        self.clip_limit = float(self.config.get("clip_limit", 2.0))
        tile_grid_size = self.config.get("tile_grid_size", [8, 8])
        self.tile_grid_size = tuple(tile_grid_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def update_params(
        self,
        clip_limit: float | None = None,
        tile_grid_size: tuple[int, int] | list[int] | None = None,
    ) -> None:
        """CLAHE のパラメータを動的に更新する.

        指定されたパラメータを反映したうえで内部の CLAHE オブジェクトを再生成する.
        省略されたパラメータは現在の値を維持する.

        Args:
            clip_limit (float | None, optional): 新しいコントラスト制限値.
                正の値でなければならない. デフォルトは None (変更しない).
            tile_grid_size (tuple[int, int] | list[int] | None, optional):
                新しいタイルグリッドサイズ. 長さ 2 かつ各要素が正の整数でなければならない.
                デフォルトは None (変更しない).

        Raises:
            ValueError: パラメータが不正な場合.
        """
        if clip_limit is not None:
            if clip_limit <= 0:
                raise ValueError(f"clip_limit must be > 0, got {clip_limit}")
            self.clip_limit = float(clip_limit)

        if tile_grid_size is not None:
            size = tuple(tile_grid_size)
            if len(size) != 2 or not all(isinstance(v, int) and v > 0 for v in size):
                raise ValueError(
                    "tile_grid_size must be a length-2 sequence of positive "
                    f"integers, got {tile_grid_size}"
                )
            self.tile_grid_size = size

        # 設定ディクショナリも同期させる.
        self.config["clip_limit"] = self.clip_limit
        self.config["tile_grid_size"] = list(self.tile_grid_size)

        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE処理を実行します.

        Args:
            image (np.ndarray): 入力画像(BGRまたはグレースケール).

        Returns:
            np.ndarray: CLAHE処理された画像.

        Raises:
            ProcessorValidationError: 入力画像が不正な場合.
            ProcessorRuntimeError: 処理中にエラーが発生した場合.
        """
        self.validator.validate_image(image)

        try:
            # グレースケール画像 (H, W) はそのまま適用.
            if image.ndim == 2:
                return self.clahe.apply(image)
            # shape (H, W, 1) の 1 チャンネル画像は squeeze して適用し,
            # 入力と同じ 3 次元形状で返す.
            if image.ndim == 3 and image.shape[2] == 1:
                clahe_img = self.clahe.apply(image.squeeze(axis=2))
                return clahe_img[:, :, np.newaxis]

            # カラー画像の処理
            if self.color_mode == "gray":
                gray = to_grayscale(image)
                clahe_img = self.clahe.apply(gray)
                return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

            elif self.color_mode == "lab":
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe_l = self.clahe.apply(l)
                clahe_lab = cv2.merge([clahe_l, a, b])
                return cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)

            elif self.color_mode == "bgr":
                b, g, r = cv2.split(image)
                clahe_b = self.clahe.apply(b)
                clahe_g = self.clahe.apply(g)
                clahe_r = self.clahe.apply(r)
                return cv2.merge([clahe_b, clahe_g, clahe_r])

            else:
                # 未知のモード（バリデータでチェックされるはずなのでここには来ないはず）
                self.logger.warning(
                    f"Unknown color mode: {self.color_mode}, using 'gray'"
                )
                gray = to_grayscale(image)
                clahe_img = self.clahe.apply(gray)
                return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

        except cv2.error as e:
            error_msg = f"Error during CLAHE processing: {e}"
            self.logger.error(error_msg)
            raise ProcessorRuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in {self.name}: {e}"
            self.logger.error(error_msg)
            raise ProcessorRuntimeError(error_msg)

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """
        CLAHEプロセッサのデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
        """
        return {"color_mode": "gray", "clip_limit": 2.0, "tile_grid_size": [8, 8]}
