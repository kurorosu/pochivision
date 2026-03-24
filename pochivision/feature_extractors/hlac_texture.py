"""HLAC（Higher-order Local Auto-Correlation）テクスチャ特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from scipy.signal import convolve2d

from pochivision.processors.binarization import OtsuBinarizationProcessor
from pochivision.processors.resize import ResizeProcessor

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("hlac")
class HLACTextureExtractor(BaseFeatureExtractor):
    """
    画像のHLAC（Higher-order Local Auto-Correlation）テクスチャ特徴量を抽出するクラス.

    HLACは画像のテクスチャ解析に使用される特徴量で、
    局所的なピクセルパターンの自己相関を計算することで、
    スケール不変性や回転不変性を持つテクスチャ記述子を生成します。
    高次の空間的相関パターンを定量化できます。

    抽出する特徴量:
    - 標準HLAC: 45次元の特徴量（0次、1次、2次自己相関） [correlation_coefficient]
    - 回転不変HLAC: 11次元の特徴量（回転に対して不変） [correlation_coefficient]
    - スケール不変性: 複数スケールでの特徴抽出 [correlation_coefficient]
    - 正規化: 特徴量の正規化による明度変化への頑健性 [normalized_correlation]

    設定により、次数、回転不変性、正規化、スケール係数などを調整できます。
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "hlac_feature": "correlation_coefficient",  # 基本的なHLAC特徴量
        "normalized_hlac_feature": "normalized_correlation",  # 正規化されたHLAC特徴量
    }

    def __init__(
        self,
        name: str = "hlac_texture",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        HLACTextureExtractorのコンストラクタ.

        Args:
            name (str): 特徴量抽出器名. デフォルトは "hlac_texture".
            config (dict, optional): 設定パラメータ. デフォルトは空の辞書.
        """
        super().__init__(name, config or {})

        # 設定パラメータの取得
        self.order = self.config["order"]  # 自己相関の次数
        self.rotate_invariant = self.config["rotate_invariant"]  # 回転不変性
        self.normalize = self.config["normalize"]  # 正規化フラグ
        self.scales = self.config["scales"]  # マルチスケール係数

        # リサイズ形状（Noneの場合はリサイズしない）
        resize_shape_config = self.config["resize_shape"]
        self.resize_shape = (
            tuple(resize_shape_config) if resize_shape_config is not None else None
        )

        # ResizeProcessorの準備（リサイズが必要な場合のみ）
        self.resize_processor = None
        if self.resize_shape is not None:
            resize_config = ResizeProcessor.get_default_config()
            resize_config["width"] = self.resize_shape[1]
            resize_config["height"] = self.resize_shape[0]
            # 特徴量抽出では正確なサイズ合わせが必要なため、アスペクト比保持を無効化
            resize_config["preserve_aspect_ratio"] = False

            self.resize_processor = ResizeProcessor(
                name="resize_for_hlac", config=resize_config
            )

        # 大津法二値化プロセッサの準備
        self.otsu_processor = OtsuBinarizationProcessor(name="otsu_for_hlac", config={})

        # HLACカーネルの事前生成
        self.kernels = self._generate_hlac_kernels()

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からHLACテクスチャ特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（BGR形式）.

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書.
                - hlac_feature_{i}: 各HLAC特徴量（i=0,1,2,...）

        Raises:
            ValueError: 画像が空の場合や無効な形状の場合.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        try:
            # 画像の型を適切に変換
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
                raise ValueError(
                    f"Input image must be 2D or 3D, got shape: {image.shape}"
                )

            # 画像を指定サイズにリサイズ（スケール統一のため）
            if self.resize_processor is not None:
                # リサイズプロセッサーを使用してリサイズ
                gray_image = self.resize_processor.process(gray_image)

            features = self._extract_hlac_features(gray_image)

            # 結果の辞書作成（ゼロパディングで正しい順序を保証）
            results = {}
            for i, feature_value in enumerate(features):
                results[f"hlac_feature_{i:02d}"] = float(feature_value)

            return results

        except Exception:
            # エラーが発生した場合、デフォルト値で埋める
            return self._get_default_results()

    def _generate_hlac_kernels(self) -> List[np.ndarray]:
        """
        HLACのパターンカーネルを生成する.

        Returns:
            List[np.ndarray]: HLACカーネルのリスト.
        """
        patterns = []

        # 0次: 中心画素
        patterns.append(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))

        # 1次: 8方向の隣接画素
        offsets = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

        for dx, dy in offsets:
            k = np.zeros((3, 3), dtype=np.uint8)
            k[1, 1] = 1  # 中心画素
            k[1 + dy, 1 + dx] = 1  # 隣接画素
            patterns.append(k)

        # 2次: 2つの隣接画素の組み合わせ
        if self.order >= 2:
            for i in range(len(offsets)):
                for j in range(i, len(offsets)):
                    dx1, dy1 = offsets[i]
                    dx2, dy2 = offsets[j]
                    k = np.zeros((3, 3), dtype=np.uint8)
                    k[1, 1] = 1  # 中心画素
                    k[1 + dy1, 1 + dx1] = 1  # 第1隣接画素
                    k[1 + dy2, 1 + dx2] = 1  # 第2隣接画素
                    patterns.append(k)

        # 回転不変性の処理
        if self.rotate_invariant:
            unique_patterns: List[np.ndarray] = []
            for pattern in patterns:
                # 4方向の回転を生成
                rotations = [np.rot90(pattern, k) for k in range(4)]
                # 既存のパターンと重複していないかチェック
                is_unique = True
                for rotation in rotations:
                    for existing in unique_patterns:
                        if np.array_equal(rotation, existing):
                            is_unique = False
                            break
                    if not is_unique:
                        break

                if is_unique:
                    # 最小の回転パターンを代表として選択
                    min_pattern = min(rotations, key=lambda x: x.tobytes())
                    unique_patterns.append(min_pattern)

            return unique_patterns
        else:
            return patterns

    def _extract_hlac_features(self, image: np.ndarray) -> np.ndarray:
        """
        スケール不変・正規化HLAC特徴量を抽出する.

        Args:
            image (np.ndarray): グレースケール画像.

        Returns:
            np.ndarray: HLAC特徴量ベクトル.
        """
        total_features = np.zeros(len(self.kernels), dtype=np.float32)

        # マルチスケール処理
        for scale in self.scales:
            # スケールに応じて画像をリサイズ
            if scale != 1.0:
                new_height = int(image.shape[0] * scale)
                new_width = int(image.shape[1] * scale)
                if new_height > 0 and new_width > 0:
                    # ResizeProcessorを使用してリサイズ
                    resize_config = ResizeProcessor.get_default_config()
                    resize_config["width"] = new_width
                    resize_config["height"] = new_height
                    resize_config["preserve_aspect_ratio"] = False

                    scale_resize_processor = ResizeProcessor(
                        name=f"resize_scale_{scale}", config=resize_config
                    )
                    scaled_image = scale_resize_processor.process(image)
                else:
                    continue  # スケールが小さすぎる場合はスキップ
            else:
                scaled_image = image

            # 大津法二値化プロセッサを使用
            binary_image = self.otsu_processor.process(scaled_image)
            # HLACでは0と1の二値画像が必要なので255を1に正規化
            binary_image = (binary_image / 255).astype(np.uint8)

            # パディング（境界処理）
            padded_image = np.pad(binary_image, pad_width=1, mode="constant")

            # 各カーネルに対して畳み込み処理
            for i, kernel in enumerate(self.kernels):
                # 畳み込み計算
                conv_result = convolve2d(padded_image, kernel[::-1, ::-1], mode="valid")
                # パターンマッチング（カーネルの合計値と一致する画素数をカウント）
                total_features[i] += np.sum(conv_result == kernel.sum())

        # 正規化処理
        if self.normalize:
            total_sum = np.sum(total_features)
            if total_sum > 0:
                total_features = total_features / total_sum

        return total_features

    def _get_default_results(self) -> Dict[str, float]:
        """
        エラー時のデフォルト結果を返す.

        Returns:
            Dict[str, float]: デフォルト特徴量の辞書.
        """
        num_features = (
            len(self.kernels)
            if hasattr(self, "kernels")
            else (11 if self.rotate_invariant else 45)
        )
        return {f"hlac_feature_{i:02d}": 0.0 for i in range(num_features)}

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        HLACTextureExtractorのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
                - order: 自己相関の次数（1または2）
                - rotate_invariant: 回転不変性の有効/無効
                - normalize: 特徴量の正規化の有効/無効
                - scales: マルチスケール処理のスケール係数リスト
                - resize_shape: リサイズ形状（None=リサイズしない）
        """
        return {
            "order": 2,
            "rotate_invariant": False,
            "normalize": True,
            "scales": [1.0, 0.75, 0.5],
            "resize_shape": None,
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する特徴量名のリストを返す（単位付き）.

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = HLACTextureExtractor.get_base_feature_names()
        return [
            f"{name}[{HLACTextureExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names() -> List[str]:
        """
        この特徴量抽出器が出力する基本特徴量名のリストを返す（単位なし）.

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        # デフォルト設定での特徴量名を返す
        default_config = HLACTextureExtractor.get_default_config()

        # 回転不変性に応じて特徴量数を決定
        if default_config["rotate_invariant"]:
            num_features = 11  # 回転不変HLAC
        else:
            if default_config["order"] == 1:
                num_features = 9  # 0次(1) + 1次(8)
            else:  # order == 2
                num_features = 45  # 0次(1) + 1次(8) + 2次(36)

        return [f"hlac_feature_{i:02d}" for i in range(num_features)]

    @staticmethod
    def get_feature_units() -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
        """
        # 基本特徴量名を取得
        base_names = HLACTextureExtractor.get_base_feature_names()

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = HLACTextureExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "hlac_feature_00"）

        Returns:
            str: 対応する単位
        """
        # HLAC特徴量は基本的にcorrelation_coefficient
        # 正規化が有効な場合はnormalized_correlationになるが、
        # 特徴量名からは判別できないため、基本単位を返す
        if feature_name.startswith("hlac_feature"):
            return HLACTextureExtractor._FEATURE_UNITS["hlac_feature"]
        return "unknown"
