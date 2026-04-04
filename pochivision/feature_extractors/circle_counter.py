"""画像内の丸をカウントする特徴量抽出を行うモジュール."""

from typing import Any

import cv2
import numpy as np

from pochivision.capturelib.log_manager import LogManager
from pochivision.exceptions.extractor import ExtractorValidationError
from pochivision.utils.image import to_grayscale

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("circle_counter")
class CircleCounterExtractor(BaseFeatureExtractor):
    """
    画像内の丸（円形オブジェクト）をカウントする特徴量抽出器.

    HoughCircles変換を使用して円形オブジェクトを検出し、
    設定可能な真円度フィルタリングによって精度を向上させます。

    抽出する特徴量:
    - circle_count: 検出された円の総数 [count]
    - small_circle_count: 小サイズ円の数（半径 < 全体の1/3） [count]
    - medium_circle_count: 中サイズ円の数（1/3 <= 半径 < 2/3） [count]
    - large_circle_count: 大サイズ円の数（半径 >= 2/3） [count]
    - circle_density: 画像面積に対する円の密度 [count/pixel2]
    - avg_circle_radius: 平均円半径 [pixel]
    - radius_std: 円半径の標準偏差 [pixel]

    主要パラメータの調整指針:
    - param1 (30-100): 高いほどエッジ検出が厳格になり、ノイズに強くなるが検出漏れが増加
    - param2 (10-50): 低いほど多くの円を検出するが、偽陽性（誤検出）も増加
    - circularity_threshold (0.5-0.9): 高いほど真円に近い形状のみを検出

    使用例と調整例:
    - 高品質画像（低ノイズ）: param1=30-40, param2=20-30
    - 低品質画像（高ノイズ）: param1=60-80, param2=35-45
    - 厳密な円検出: param2=40-50, circularity_threshold=0.8-0.9
    - 多様な形状検出: param2=15-25, circularity_threshold=0.5-0.7
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "circle_count": "count",
        "small_circle_count": "count",
        "medium_circle_count": "count",
        "large_circle_count": "count",
        "circle_density": "count/pixel2",
        "avg_circle_radius": "pixel",
        "radius_std": "pixel",
    }

    def __init__(
        self,
        name: str = "circle_counter",
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        CircleCounterExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "circle_counter".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得
        self.min_radius = self.config["min_radius"]
        self.max_radius = self.config["max_radius"]
        self.min_dist_ratio = self.config["min_dist_ratio"]
        self.param1 = self.config["param1"]
        self.param2 = self.config["param2"]
        self.circularity_threshold = self.config["circularity_threshold"]
        self.blur_kernel_size = self.config["blur_kernel_size"]
        if self.blur_kernel_size > 0 and self.blur_kernel_size % 2 == 0:
            raise ExtractorValidationError(
                f"blur_kernel_size must be an odd number or 0, got {self.blur_kernel_size}"
            )
        self.enable_circularity_filter = self.config["enable_circularity_filter"]

    def extract(self, image: np.ndarray) -> dict[str, float | int]:
        """
        画像から円形オブジェクトをカウントして特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            dict[str, float | int]: 抽出された特徴量の辞書.

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ExtractorValidationError("Input image is empty or None")

        try:
            # float (0-1) 入力を uint8 スケールに変換
            if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            gray = to_grayscale(image)

            if self.blur_kernel_size > 0:
                gray = cv2.GaussianBlur(
                    gray, (self.blur_kernel_size, self.blur_kernel_size), 0
                )

            height, width = gray.shape
            image_area = height * width

            if self.max_radius > 0:
                user_max_radius = self.max_radius
                image_limit = min(height, width) // 2
                max_radius = min(user_max_radius, image_limit)
            else:
                max_radius = min(height, width) // 4

            min_dist = max(1, int(max_radius * self.min_dist_ratio))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min_dist,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=max_radius,
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                if self.enable_circularity_filter:
                    circles = self._filter_by_circularity(gray, circles)
            else:
                circles = np.array([])

            return self._calculate_features(circles, image_area, max_radius)
        except Exception:
            LogManager().get_logger().exception(
                "CircleCounter feature extraction failed"
            )
            raise

    def _filter_by_circularity(
        self, gray: np.ndarray, circles: np.ndarray
    ) -> np.ndarray:
        """
        検出された円を実画像のエッジに基づく真円度でフィルタリングする.

        各候補円の周辺領域 (ROI) でエッジ検出・輪郭抽出を行い,
        実際の輪郭形状の真円度 (4*pi*area / perimeter^2) で評価する.

        Args:
            gray (np.ndarray): グレースケール画像.
            circles (np.ndarray): 検出された円の配列 [[x, y, r], ...].

        Returns:
            np.ndarray: フィルタリング後の円の配列.
        """
        if len(circles) == 0:
            return circles

        filtered_circles = []
        height, width = gray.shape

        for x, y, r in circles:
            # ROI の範囲を計算 (円を囲む矩形 + マージン)
            margin = max(r // 4, 2)
            x1 = max(0, x - r - margin)
            y1 = max(0, y - r - margin)
            x2 = min(width, x + r + margin)
            y2 = min(height, y + r + margin)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # ROI 内でエッジ検出
            edges = cv2.Canny(roi, self.param1 // 2, self.param1)

            # 円領域のマスクを作成し, ROI 外のエッジを除外
            roi_mask = np.zeros_like(edges)
            cx_roi, cy_roi = x - x1, y - y1
            cv2.circle(roi_mask, (cx_roi, cy_roi), r + margin // 2, 255, -1)
            edges = cv2.bitwise_and(edges, roi_mask)

            # エッジから輪郭を抽出
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # 最大輪郭を選択
            contour = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0 and area > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity >= self.circularity_threshold:
                    filtered_circles.append([x, y, r])

        return np.array(filtered_circles) if filtered_circles else np.array([])

    def _calculate_features(
        self, circles: np.ndarray, image_area: int, max_radius: int
    ) -> dict[str, float | int]:
        """
        検出された円から特徴量を計算する.

        Args:
            circles (np.ndarray): 検出された円の配列.
            image_area (int): 画像の総面積.
            max_radius (int): 使用された最大半径.

        Returns:
            dict[str, float | int]: 計算された特徴量.
        """
        results: dict[str, float | int] = {}

        # 基本カウント
        total_count = len(circles)
        results["circle_count"] = total_count

        if total_count == 0:
            # 円が検出されなかった場合
            results["small_circle_count"] = 0
            results["medium_circle_count"] = 0
            results["large_circle_count"] = 0
            results["circle_density"] = 0.0
            results["avg_circle_radius"] = 0.0
            results["radius_std"] = 0.0
        else:
            # 半径の分析
            radii = circles[:, 2]

            # サイズ別カウント（最大半径を基準にした相対的分類）
            small_threshold = max_radius / 3
            medium_threshold = max_radius * 2 / 3

            small_count = int(np.sum(radii < small_threshold))
            medium_count = int(
                np.sum((radii >= small_threshold) & (radii < medium_threshold))
            )
            large_count = int(np.sum(radii >= medium_threshold))

            results["small_circle_count"] = small_count
            results["medium_circle_count"] = medium_count
            results["large_circle_count"] = large_count

            # 密度計算
            results["circle_density"] = (
                float(total_count / image_area) if image_area > 0 else 0.0
            )

            # 半径統計
            results["avg_circle_radius"] = float(np.mean(radii))
            results["radius_std"] = float(np.std(radii))

        return results

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """
        CircleCounterExtractorのデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
        """
        return {
            "min_radius": 5,  # 最小半径（ピクセル）
            "max_radius": 0,  # 最大半径（0の場合は画像サイズの1/4を使用）
            "min_dist_ratio": 0.8,  # 最小距離の係数（max_radius * ratio）
            "param1": 50,  # Canny検出器の高閾値（エッジ検出感度：高いほど厳格、推奨：30-100）
            "param2": 30,  # 蓄積器の閾値（円検出感度：低いほど多く検出、推奨：10-50）
            "circularity_threshold": 0.7,  # 真円度の閾値（0.0-1.0、1.0が完全な円）
            "blur_kernel_size": 5,  # ガウシアンブラーのカーネルサイズ（0で無効化）
            "enable_circularity_filter": True,  # 真円度フィルタリングを有効にするか
        }

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            list[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = CircleCounterExtractor.get_base_feature_names()
        return [
            f"{name}[{CircleCounterExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            list[str]: 基本特徴量名のリスト.
        """
        return [
            "circle_count",
            "small_circle_count",
            "medium_circle_count",
            "large_circle_count",
            "circle_density",
            "avg_circle_radius",
            "radius_std",
        ]

    @staticmethod
    def get_feature_units() -> dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            dict[str, str]: 特徴量名と単位の対応辞書.
        """
        base_names = CircleCounterExtractor.get_base_feature_names()
        return {
            name: CircleCounterExtractor._get_unit_for_feature(name)
            for name in base_names
        }

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名.

        Returns:
            str: 対応する単位.
        """
        return CircleCounterExtractor._FEATURE_UNITS.get(feature_name, "unknown")
