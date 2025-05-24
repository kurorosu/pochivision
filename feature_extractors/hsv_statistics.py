"""HSV統計特徴量抽出を行うモジュール."""

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("hsv")
class HSVStatisticsExtractor(BaseFeatureExtractor):
    """
    画像のHSV統計特徴量を抽出するクラス.

    抽出する特徴量（H、S、Vチャンネルそれぞれ）:
    - mean: HSV平均値
    - median: HSV中央値
    - variance: HSV分散
    - std_dev: HSV標準偏差
    - cv: 変動係数（標準偏差/平均値）

    設定により、RGB値がすべて0のピクセルを計算から除外することができます。
    """

    def __init__(
        self,
        name: str = "hsv_statistics",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        HSVStatisticsExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "hsv_statistics".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.exclude_black_pixels = self.config["exclude_black_pixels"]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からHSV統計特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                H、S、Vチャンネルそれぞれについて:
                - {channel}_mean: 平均値
                - {channel}_median: 中央値
                - {channel}_variance: 分散
                - {channel}_std_dev: 標準偏差
                - {channel}_cv: 変動係数

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Input image must be a 3-channel color image, got shape: {image.shape}"
            )

        # BGR to HSV変換
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        results = {}

        # 黒ピクセル除外の処理
        if self.exclude_black_pixels:
            # 元のBGR画像ですべてのチャンネルが0でないピクセルのマスクを作成
            non_black_mask = np.any(image > 0, axis=2)
        else:
            # すべてのピクセルを使用
            non_black_mask = np.ones(image.shape[:2], dtype=bool)

        # チャンネル名の定義
        channel_names = ["hue", "saturation", "value"]

        # 各チャンネルについて統計値を計算
        for i, channel_name in enumerate(channel_names):
            channel_data = hsv_image[:, :, i]

            # マスクを適用してピクセル値を取得
            if self.exclude_black_pixels:
                pixels = channel_data[non_black_mask].astype(np.float64)
            else:
                pixels = channel_data.flatten().astype(np.float64)

            # 統計値の計算
            if len(pixels) == 0:
                # 有効なピクセルがない場合
                mean_val = 0.0
                median_val = 0.0
                variance_val = 0.0
                std_dev_val = 0.0
                cv_val = float("inf")
            else:
                mean_val = float(np.mean(pixels))
                median_val = float(np.median(pixels))
                variance_val = float(np.var(pixels))
                std_dev_val = float(np.std(pixels))

                # 変動係数の計算（平均値が0の場合は無限大になるため特別処理）
                cv_val = (
                    float(std_dev_val / mean_val) if mean_val != 0 else float("inf")
                )

            # 結果に追加
            results[f"{channel_name}_mean"] = mean_val
            results[f"{channel_name}_median"] = median_val
            results[f"{channel_name}_variance"] = variance_val
            results[f"{channel_name}_std_dev"] = std_dev_val
            results[f"{channel_name}_cv"] = cv_val

        return results

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        HSVStatisticsExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - exclude_black_pixels: RGB値がすべて0のピクセルを除外するかどうか
        """
        return {"exclude_black_pixels": True}

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す.

        Returns:
            list[str]: 特徴量名のリスト.
        """
        feature_names = []
        channels = ["hue", "saturation", "value"]
        stats = ["mean", "median", "variance", "std_dev", "cv"]

        for channel in channels:
            for stat in stats:
                feature_names.append(f"{channel}_{stat}")

        return feature_names
