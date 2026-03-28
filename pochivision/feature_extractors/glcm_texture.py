"""GLCM（Gray-Level Co-occurrence Matrix）テクスチャ特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from pochivision.capturelib.log_manager import LogManager
from pochivision.processors.resize import ResizeProcessor

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("glcm")
class GLCMTextureExtractor(BaseFeatureExtractor):
    """
    画像のGLCM (Gray-Level Co-occurrence Matrix) テクスチャ特徴量を抽出するクラス.

    GLCM は画像のテクスチャ解析に使用される特徴量で,
    指定された距離と角度でのグレーレベルの共起関係を表現する.
    テクスチャの粗さ, 方向性, 規則性などを定量化できる.

    抽出するプロパティ (各距離・角度の組み合わせごとに出力):
    - contrast: コントラスト (局所的な強度変化) [intensity_squared]
    - dissimilarity: 非類似度 (隣接ピクセル間の差異) [intensity]
    - homogeneity: 均質性 (局所的な均一性) [ratio]
    - energy: エネルギー (テクスチャの均一性) [ratio]
    - correlation: 相関 (ピクセル間の線形依存関係) [correlation_coefficient]
    - ASM: Angular Second Moment (エネルギーの二乗) [ratio]

    特徴量名の形式: ``{property}_{distance}_{angle_deg}``
    (例: ``contrast_1_0``, ``energy_2_45``)

    特徴量数 = len(properties) x len(distances) x len(angles).
    デフォルト設定 (6 プロパティ x 3 距離 x 4 角度) では 72 特徴量.

    設定により, 距離, 角度, グレーレベル数, 対称性, 正規化, リサイズ形状などを調整できる.
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "contrast": "intensity_squared",
        "dissimilarity": "intensity",
        "homogeneity": "ratio",
        "energy": "ratio",
        "correlation": "correlation_coefficient",
        "ASM": "ratio",
    }

    def __init__(
        self,
        name: str = "glcm_texture",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        GLCMTextureExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "glcm_texture".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得（デフォルト設定が既にマージされているため直接アクセス）
        self.distances = self.config["distances"]
        self.angles = self._parse_angles(self.config["angles"])
        self.levels = self.config["levels"]
        self.symmetric = self.config["symmetric"]
        self.normed = self.config["normed"]
        self.properties = self.config["properties"]

        # リサイズ形状 (None の場合はリサイズしない)
        resize_shape_config = self.config["resize_shape"]
        self.resize_shape = (
            tuple(resize_shape_config) if resize_shape_config is not None else None
        )

        self.resize_processor = None
        if self.resize_shape is not None:
            resize_config = ResizeProcessor.get_default_config()
            resize_config["width"] = self.resize_shape[1]
            resize_config["height"] = self.resize_shape[0]
            resize_config["preserve_aspect_ratio"] = False
            self.resize_processor = ResizeProcessor(
                name="resize_for_glcm", config=resize_config
            )

    def _parse_angles(self, angles_config: List[Union[int, float]]) -> List[float]:
        """
        角度設定を解析してラジアン値のリストに変換する.

        Args:
            angles_config: 角度設定（度数のリスト）

        Returns:
            List[float]: ラジアン値のリスト

        Raises:
            ValueError: 無効な角度設定の場合
        """
        if not isinstance(angles_config, list):
            raise ValueError("Angles must be a list of numbers")

        if not angles_config:
            raise ValueError("Angles list cannot be empty")

        # すべての要素が数値かチェック
        for angle in angles_config:
            if not isinstance(angle, (int, float)):
                raise ValueError("All angles must be numeric values")

        # 度数をラジアンに変換
        return [np.radians(float(angle)) for angle in angles_config]

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からGLCMテクスチャ特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                各プロパティ、距離、角度の組み合わせについて:
                - {property}_{distance}_{angle_deg}: 特徴量値

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        # 画像の型を適切に変換（OpenCVがサポートしていない型の場合）
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            # サポートされていない型の場合、uint8に変換
            if image.dtype in [np.int32, np.int64]:
                # 整数型の場合、0-255の範囲にクリップしてuint8に変換
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                # その他の型の場合、float32経由でuint8に変換
                image = image.astype(np.float32)
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray_image = image.copy()
        else:
            raise ValueError(f"Input image must be 2D or 3D, got shape: {image.shape}")

        # リサイズ (設定されている場合)
        if self.resize_processor is not None:
            gray_image = self.resize_processor.process(gray_image)

        # uint8 に変換 (NORM_MINMAX はコントラスト情報を破壊するため使用しない)
        if not np.issubdtype(gray_image.dtype, np.integer):
            gray_image = np.clip(gray_image * 255, 0, 255).astype(np.uint8)
        else:
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

        # levels < 256 の場合は整数除算で量子化
        if self.levels < 256:
            gray_image = (gray_image // (256 // self.levels)).astype(np.uint8)

        results = {}

        try:
            glcm = graycomatrix(
                gray_image,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=self.symmetric,
                normed=self.normed,
            )

            # 各プロパティについて特徴量を計算
            for prop in self.properties:
                try:
                    # プロパティ値を計算
                    prop_values = graycoprops(glcm, prop)

                    # 距離と角度の組み合わせごとに結果を格納
                    for d_idx, distance in enumerate(self.distances):
                        for a_idx, angle in enumerate(self.angles):
                            # 角度をラジアンから度に変換
                            angle_deg = int(np.degrees(angle))

                            # 特徴量名を生成
                            feature_name = f"{prop}_{distance}_{angle_deg}"

                            # 特徴量値を取得
                            feature_value = float(prop_values[d_idx, a_idx])

                            # NaN/Inf は未定義を示す (例: 均一画像の correlation)
                            if np.isnan(feature_value) or np.isinf(feature_value):
                                LogManager().get_logger().warning(
                                    f"GLCM {feature_name} is {feature_value}, "
                                    "replacing with NaN"
                                )
                                feature_value = float("nan")

                            results[feature_name] = feature_value

                except Exception:
                    LogManager().get_logger().exception(
                        f"GLCM property '{prop}' computation failed"
                    )
                    raise

        except Exception:
            LogManager().get_logger().exception("GLCM feature extraction failed")
            raise

        return results

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        GLCMTextureExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - distances: ピクセル間距離のリスト
                - angles: 角度のリスト（度数）
                - levels: グレーレベル数
                - symmetric: 対称性を考慮するかどうか
                - normed: 正規化するかどうか
                - properties: 計算するプロパティのリスト
        """
        return {
            "distances": [1, 2, 3],  # 1, 2, 3ピクセル距離
            "angles": [0, 45, 90, 135],  # 標準的な4方向（度数）
            "levels": 256,  # 256グレーレベル
            "symmetric": True,  # 対称性を考慮
            "normed": True,  # 正規化
            "properties": [
                "contrast",  # コントラスト
                "dissimilarity",  # 非類似度
                "homogeneity",  # 均質性
                "energy",  # エネルギー
                "correlation",  # 相関
                "ASM",  # Angular Second Moment
            ],
            "resize_shape": None,  # リサイズ形状 (None=リサイズしない)
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = GLCMTextureExtractor.get_base_feature_names()
        return [
            f"{name}[{GLCMTextureExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        # デフォルト設定を使用して特徴量名を生成
        default_config = GLCMTextureExtractor.get_default_config()

        # 一時的な抽出器を作成して角度を解析
        temp_extractor = GLCMTextureExtractor()
        angles = temp_extractor.angles

        feature_names = []
        for prop in default_config["properties"]:
            for distance in default_config["distances"]:
                for angle in angles:
                    angle_deg = int(np.degrees(angle))
                    feature_names.append(f"{prop}_{distance}_{angle_deg}")

        return feature_names

    @staticmethod
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
        """
        # 基本特徴量名を取得
        base_names = GLCMTextureExtractor.get_base_feature_names()

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = GLCMTextureExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "contrast_1_0"）

        Returns:
            str: 対応する単位
        """
        # 特徴量名からプロパティ部分を抽出
        parts = feature_name.split("_")
        if len(parts) >= 1:
            prop = parts[0]
            return GLCMTextureExtractor._FEATURE_UNITS.get(prop, "unknown")
        return "unknown"
