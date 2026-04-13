"""輪郭抽出プロセッサーを定義するモジュール."""

from typing import Any

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError
from pochivision.utils.image import to_grayscale

from .base import BaseProcessor
from .registry import register_processor
from .validators.contour import ContourValidator


def find_contours_compat(
    image: np.ndarray, mode: int, method: int
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """バージョン非依存な ``cv2.findContours`` ラッパー.

    OpenCV 3.x では ``(image, contours, hierarchy)`` の 3 要素,
    OpenCV 4.x では ``(contours, hierarchy)`` の 2 要素を返すため,
    戻り値数の違いを吸収して ``(contours, hierarchy)`` で統一する.

    Args:
        image (np.ndarray): 入力画像 (8bit シングルチャンネル推奨).
        mode (int): 輪郭抽出モード (``cv2.RETR_*``).
        method (int): 輪郭近似方法 (``cv2.CHAIN_APPROX_*``).

    Returns:
        tuple[list[np.ndarray], np.ndarray | None]:
            - 検出された輪郭のリスト.
            - 階層情報 ``(1, N, 4)`` 形状の配列. 輪郭が無い場合は ``None``.
    """
    result = cv2.findContours(image, mode, method)
    # OpenCV 4.x: (contours, hierarchy), OpenCV 3.x: (image, contours, hierarchy)
    if len(result) == 2:
        contours, hierarchy = result
    else:
        _, contours, hierarchy = result
    return list(contours), hierarchy


@register_processor("contour")
class ContourProcessor(BaseProcessor):
    """画像から輪郭を抽出するプロセッサー."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """
        ContourProcessorを初期化.

        Args:
            name (str): プロセッサの名前.
            config (dict[str, Any]): 輪郭抽出の設定辞書.
        """
        super().__init__(name, config)
        self.validator = ContourValidator(self.config)

        default_vals = self.get_default_config()

        # パラメータ解析
        retrieval_mode_str = self.config.get(
            "retrieval_mode", default_vals["retrieval_mode"]
        )
        self._retrieval_mode = self._get_retrieval_mode(retrieval_mode_str)
        self._retrieval_mode_str = retrieval_mode_str

        approx_method_str = self.config.get(
            "approximation_method", default_vals["approximation_method"]
        )
        self._approximation_method = self._get_approximation_method(approx_method_str)
        self._approx_method_str = approx_method_str

        self._min_area = self.config.get("min_area", default_vals["min_area"])
        self._select_mode = self.config.get("select_mode", default_vals["select_mode"])
        self._contour_rank = self.config.get(
            "contour_rank", default_vals["contour_rank"]
        )
        self._outside_color = self.config.get(
            "outside_color", default_vals["outside_color"]
        )
        self._inside_color = self.config.get(
            "inside_color", default_vals["inside_color"]
        )

        # 直近の findContours 結果 (後段や将来の階層フィルタ拡張で参照可能).
        self._last_contours: list[np.ndarray] = []
        self._last_hierarchy: np.ndarray | None = None

    @property
    def last_contours(self) -> list[np.ndarray]:
        """直近の ``process`` 呼び出しで検出された輪郭のリストを返す."""
        return self._last_contours

    @property
    def last_hierarchy(self) -> np.ndarray | None:
        """直近の ``process`` 呼び出しで得られた階層情報を返す.

        Returns:
            np.ndarray | None: ``cv2.findContours`` が返す ``(1, N, 4)`` 形状の
            階層情報. 輪郭が検出されない場合は ``None``.
        """
        return self._last_hierarchy

    @staticmethod
    def _get_retrieval_mode(mode_str: str) -> int:
        """
        文字列で指定された輪郭抽出モードをOpenCVの定数に変換する.

        Args:
            mode_str (str): モード文字列 ('external', 'list', 'ccomp', 'tree', 'floodfill')

        Returns:
            int: OpenCVの輪郭抽出モード定数
        """
        mode_map = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "ccomp": cv2.RETR_CCOMP,
            "tree": cv2.RETR_TREE,
            "floodfill": cv2.RETR_FLOODFILL,
        }
        return int(mode_map.get(mode_str, cv2.RETR_LIST))

    @staticmethod
    def _get_approximation_method(method_str: str) -> int:
        """
        文字列で指定された輪郭近似方法をOpenCVの定数に変換する.

        Args:
            method_str (str): 近似方法文字列 ('none', 'simple', 'tc89_l1', 'tc89_kcos')

        Returns:
            int: OpenCVの輪郭近似方法定数
        """
        method_map = {
            "none": cv2.CHAIN_APPROX_NONE,
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
        }
        return int(method_map.get(method_str, cv2.CHAIN_APPROX_SIMPLE))

    def _select_contour_by_rank(self, contours: list[np.ndarray]) -> np.ndarray | None:
        """
        指定されたランクの輪郭を選択する.

        Args:
            contours (list[np.ndarray]): 輪郭のリスト

        Returns:
            np.ndarray | None: 指定されたランクの輪郭。該当する輪郭がない場合はNone
        """
        if not contours:
            return None

        if self._min_area > 0:
            filtered_contours = [
                c for c in contours if cv2.contourArea(c) >= self._min_area
            ]
        else:
            filtered_contours = contours

        if not filtered_contours:
            return None

        sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

        # 指定されたランク（0から始まる）の輪郭を選択
        if self._contour_rank < len(sorted_contours):
            result: np.ndarray = sorted_contours[self._contour_rank]
            return result
        elif sorted_contours:
            result = sorted_contours[0]  # ランクが範囲外の場合は最大輪郭を返す
            return result
        else:
            return None

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        画像から輪郭を抽出する.

        Args:
            image (np.ndarray): 入力画像 (前段で処理された二値画像が期待される).

        Returns:
            np.ndarray: 輪郭を描画した画像. 入力が二値化されていない場合は元の画像を返す.

        Raises:
            ProcessorRuntimeError: 画像処理中にエラーが発生した場合.
        """
        is_valid, _ = self.validator.validate_image_for_contour(image)

        if not is_valid:
            # 色空間の変換（グレースケール→BGR）が必要な場合は対応
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image.copy()

        try:
            gray_image = to_grayscale(image)

            # 画像がfloat32の場合はuint8に変換
            if gray_image.dtype != np.uint8:
                if np.max(gray_image) <= 1.0 and gray_image.dtype == np.float32:
                    gray_image = (gray_image * 255).astype(np.uint8)
                else:
                    gray_image = gray_image.astype(np.uint8)

            contours, hierarchy = find_contours_compat(
                gray_image, self._retrieval_mode, self._approximation_method
            )

            # 階層情報を後段で参照できるよう保持 (Issue #383).
            self._last_contours = list(contours)
            self._last_hierarchy = hierarchy

            if self._select_mode == "rank":
                # ランクによる選択
                selected_contour = self._select_contour_by_rank(list(contours))
                if selected_contour is not None:
                    contours = [selected_contour]
                else:
                    contours = []
            else:
                # 全ての輪郭を使用（最小面積でフィルタリング）
                if self._min_area > 0:
                    contours = [
                        c for c in contours if cv2.contourArea(c) >= self._min_area
                    ]

            output_image = np.full(
                (image.shape[0], image.shape[1], 3),
                self._outside_color,
                dtype=np.uint8,
            )

            if contours:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
                output_image[mask > 0] = self._inside_color

            return output_image

        except cv2.error as e:
            raise ProcessorRuntimeError(f"Error occurred during contour detection: {e}")

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """
        輪郭抽出プロセッサのデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
        """
        return {
            "retrieval_mode": "list",  # デフォルトはLISTモードで全ての輪郭を検出
            "approximation_method": "simple",
            "min_area": 100,  # 小さすぎる輪郭を除外
            "select_mode": "rank",  # ランクによる選択
            "contour_rank": 0,  # 0=最大面積の輪郭（デフォルト）
            "outside_color": [0, 0, 0],  # 輪郭外側の色 (黒)
            "inside_color": [255, 255, 255],  # 輪郭内側の色 (白)
        }
