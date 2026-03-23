"""画像内の丸をカウントする特徴量抽出を行うモジュール."""

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

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
        config: Optional[Dict[str, Any]] = None,
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
        self.enable_circularity_filter = self.config["enable_circularity_filter"]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像から円形オブジェクトをカウントして特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        # グレースケール変換
        gray = to_grayscale(image)

        # ガウシアンブラーでノイズ除去
        if self.blur_kernel_size > 0:
            gray = cv2.GaussianBlur(
                gray, (self.blur_kernel_size, self.blur_kernel_size), 0
            )

        # 画像サイズから動的パラメータ計算
        height, width = gray.shape
        image_area = height * width

        # 最大半径の動的調整（画像サイズに応じて）
        # ロジック:
        # 1. max_radius > 0 の場合: ユーザー指定値を使用
        # 2. max_radius = 0 の場合: 画像短辺の1/4を使用（自動計算）
        # 3. ただし、画像短辺の1/2を上限とする（画像全体を占める円を防ぐ）
        if self.max_radius > 0:
            # ユーザー指定値を使用するが、画像サイズの制約を適用
            user_max_radius = self.max_radius
            image_limit = min(height, width) // 2  # 画像短辺の半分が上限
            max_radius = min(user_max_radius, image_limit)
        else:
            # 自動計算: 画像短辺の1/4（一般的に適切なデフォルト値）
            max_radius = min(height, width) // 4

        # 最小距離の計算
        min_dist = max(1, int(max_radius * self.min_dist_ratio))

        # HoughCircles検出
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

        # 検出結果の処理
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # 真円度フィルタリング
            if self.enable_circularity_filter:
                circles = self._filter_by_circularity(gray, circles)
        else:
            circles = np.array([])

        # 特徴量計算
        return self._calculate_features(circles, image_area, max_radius)

    def _filter_by_circularity(
        self, gray: np.ndarray, circles: np.ndarray
    ) -> np.ndarray:
        """
        検出された円を真円度でフィルタリングする.

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
            # 円の輪郭を作成
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 1)

            # 輪郭を取得
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                contour = contours[0]

                # 面積と周囲長の計算
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    # 真円度の計算: 4π × 面積 / 周囲長²
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # 閾値を満たす場合のみ保持
                    if circularity >= self.circularity_threshold:
                        filtered_circles.append([x, y, r])

        return np.array(filtered_circles) if filtered_circles else np.array([])

    def _calculate_features(
        self, circles: np.ndarray, image_area: int, max_radius: int
    ) -> Dict[str, Union[float, int]]:
        """
        検出された円から特徴量を計算する.

        Args:
            circles (np.ndarray): 検出された円の配列.
            image_area (int): 画像の総面積.
            max_radius (int): 使用された最大半径.

        Returns:
            Dict[str, Union[float, int]]: 計算された特徴量.
        """
        results: Dict[str, Union[float, int]] = {}

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
            results["circle_density"] = float(total_count / image_area)

            # 半径統計
            results["avg_circle_radius"] = float(np.mean(radii))
            results["radius_std"] = float(np.std(radii))

        return results

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        CircleCounterExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
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
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
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
