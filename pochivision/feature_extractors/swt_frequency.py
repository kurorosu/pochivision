"""SWT（Stationary Wavelet Transform）周波数変換特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pywt

from pochivision.utils.image import to_grayscale

from .base import BaseFeatureExtractor
from .registry import register_feature_extractor


@register_feature_extractor("swt")
class SWTFrequencyExtractor(BaseFeatureExtractor):
    """
    画像のSWT（Stationary Wavelet Transform）周波数変換特徴量を抽出するクラス.

    SWTは画像の周波数成分を解析し、マルチスケールでの特徴を抽出します。
    通常のDWTとは異なり、SWTはダウンサンプリングを行わないため、
    平行移動不変性を持ち、より詳細な周波数解析が可能です。

    抽出する特徴量:
    - mean_ll: 低周波成分（LL）の平均値 [coefficient]
    - mean_lh: 水平高周波成分（LH）の平均値 [coefficient]
    - mean_hl: 垂直高周波成分（HL）の平均値 [coefficient]
    - mean_hh: 対角高周波成分（HH）の平均値 [coefficient]
    - energy_ll: 低周波成分（LL）のエネルギー [coefficient_squared]
    - energy_lh: 水平高周波成分（LH）のエネルギー [coefficient_squared]
    - energy_hl: 垂直高周波成分（HL）のエネルギー [coefficient_squared]
    - energy_hh: 対角高周波成分（HH）のエネルギー [coefficient_squared]
    - energy_ratio_h: 水平方向エネルギー比 [ratio]
    - energy_ratio_v: 垂直方向エネルギー比 [ratio]
    - energy_ratio_d: 対角方向エネルギー比 [ratio]
    - total_energy: 全エネルギー [coefficient_squared]
    - entropy_ll: 低周波成分のエントロピー [bits]
    - entropy_lh: 水平高周波成分のエントロピー [bits]
    - entropy_hl: 垂直高周波成分のエントロピー [bits]
    - entropy_hh: 対角高周波成分のエントロピー [bits]
    - std_ll: 低周波成分の標準偏差 [coefficient]
    - std_lh: 水平高周波成分の標準偏差 [coefficient]
    - std_hl: 垂直高周波成分の標準偏差 [coefficient]
    - std_hh: 対角高周波成分の標準偏差 [coefficient]

    マルチスケール解析を行う場合、各レベルの特徴量が追加されます。
    マルチスケールでない場合は、最高レベル（最も詳細な分解レベル）の特徴量のみが抽出されます。
    """

    # 特徴量の単位定義
    _FEATURE_UNITS = {
        "mean": "coefficient",
        "energy": "coefficient_squared",
        "energy_ratio": "ratio",
        "total_energy": "coefficient_squared",
        "entropy": "bits",
        "std": "coefficient",
    }

    def _compute_mean(self, coeffs: np.ndarray) -> float:
        """
        係数の平均値を計算する.

        Args:
            coeffs (np.ndarray): ウェーブレット係数

        Returns:
            float: 平均値
        """
        return float(np.mean(coeffs))

    def _compute_energy(self, coeffs: np.ndarray) -> float:
        """
        係数のエネルギーを計算する.

        Args:
            coeffs (np.ndarray): ウェーブレット係数

        Returns:
            float: エネルギー値
        """
        return float(np.sum(coeffs**2))

    def _compute_entropy(self, coeffs: np.ndarray) -> float:
        """
        係数のシャノンエントロピーを計算する.

        Args:
            coeffs (np.ndarray): ウェーブレット係数

        Returns:
            float: エントロピー値
        """
        # 係数を正規化して確率分布を作成
        flattened = coeffs.ravel()

        # 値の範囲を取得
        min_val, max_val = flattened.min(), flattened.max()

        # 範囲が0の場合（すべて同じ値）はエントロピー0
        if max_val == min_val:
            return 0.0

        # ヒストグラムを計算（256ビンで正規化された範囲）
        hist, _ = np.histogram(
            flattened, bins=256, range=(min_val, max_val), density=False
        )

        # 確率に変換
        prob = hist / np.sum(hist)

        # ゼロ確率を除去（log(0)を回避）
        prob = prob[prob > 0]

        # シャノンエントロピーを計算: -Σ p * log2(p)
        return float(-np.sum(prob * np.log2(prob)))

    def _compute_std(self, coeffs: np.ndarray) -> float:
        """
        係数の標準偏差を計算する.

        Args:
            coeffs (np.ndarray): ウェーブレット係数

        Returns:
            float: 標準偏差
        """
        return float(np.std(coeffs))

    def _extract_single_level_features(
        self, coeffs: tuple, level: int = 0
    ) -> Dict[str, float]:
        """
        単一レベルのSWT特徴量を抽出する.

        Args:
            coeffs (tuple): SWT係数のタプル (cA, (cH, cV, cD))
            level (int): 分解レベル（マルチスケール時のプレフィックス用）

        Returns:
            Dict[str, float]: 抽出された特徴量の辞書
        """
        features = {}

        # 係数を取得
        cA, (cH, cV, cD) = coeffs

        # 各成分の基本統計量を計算
        components = {
            "ll": cA,  # 低周波成分（Low-Low）
            "lh": cH,  # 水平高周波成分（Low-High）
            "hl": cV,  # 垂直高周波成分（High-Low）
            "hh": cD,  # 対角高周波成分（High-High）
        }

        # レベルプレフィックスを設定
        prefix = f"L{level}_" if level > 0 else ""

        # 各成分の特徴量を計算
        for comp_name, comp_data in components.items():
            # 平均値
            features[f"{prefix}mean_{comp_name}"] = self._compute_mean(comp_data)
            # エネルギー
            features[f"{prefix}energy_{comp_name}"] = self._compute_energy(comp_data)
            # エントロピー
            features[f"{prefix}entropy_{comp_name}"] = self._compute_entropy(comp_data)
            # 標準偏差
            features[f"{prefix}std_{comp_name}"] = self._compute_std(comp_data)

        # エネルギー比を計算
        total_high_energy = (
            features[f"{prefix}energy_lh"]
            + features[f"{prefix}energy_hl"]
            + features[f"{prefix}energy_hh"]
        )

        if total_high_energy > 0:
            features[f"{prefix}energy_ratio_h"] = (
                features[f"{prefix}energy_lh"] / total_high_energy
            )
            features[f"{prefix}energy_ratio_v"] = (
                features[f"{prefix}energy_hl"] / total_high_energy
            )
            features[f"{prefix}energy_ratio_d"] = (
                features[f"{prefix}energy_hh"] / total_high_energy
            )
        else:
            features[f"{prefix}energy_ratio_h"] = 0.0
            features[f"{prefix}energy_ratio_v"] = 0.0
            features[f"{prefix}energy_ratio_d"] = 0.0

        # 全エネルギー
        features[f"{prefix}total_energy"] = (
            features[f"{prefix}energy_ll"] + total_high_energy
        )

        return features

    def _adjust_image_size_for_swt(self, image: np.ndarray) -> np.ndarray:
        """
        SWT変換のために画像サイズを調整する.

        SWTは各軸のデータ長が偶数である必要があるため、
        奇数サイズの場合は1ピクセル追加して偶数サイズにする。

        Args:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: サイズ調整された画像
        """
        height, width = image.shape[:2]

        # 調整が必要かチェック
        height_adjustment = 1 if height % 2 == 1 else 0
        width_adjustment = 1 if width % 2 == 1 else 0

        if height_adjustment == 0 and width_adjustment == 0:
            # 調整不要
            return image

        # 新しいサイズを計算
        new_height = height + height_adjustment
        new_width = width + width_adjustment

        # パディングで1ピクセル追加（端を複製）
        if len(image.shape) == 2:
            # グレースケール画像
            adjusted_image = np.zeros((new_height, new_width), dtype=image.dtype)
            adjusted_image[:height, :width] = image

            # 右端を複製（幅が奇数の場合）
            if width_adjustment > 0:
                adjusted_image[:height, width:] = image[:, -1:]

            # 下端を複製（高さが奇数の場合）
            if height_adjustment > 0:
                adjusted_image[height:, :width] = image[-1:, :]

            # 右下角を複製（両方とも奇数の場合）
            if height_adjustment > 0 and width_adjustment > 0:
                adjusted_image[height:, width:] = image[-1, -1]

        else:
            # カラー画像
            adjusted_image = np.zeros(
                (new_height, new_width, image.shape[2]), dtype=image.dtype
            )
            adjusted_image[:height, :width, :] = image

            # 右端を複製（幅が奇数の場合）
            if width_adjustment > 0:
                adjusted_image[:height, width:, :] = image[:, -1:, :]

            # 下端を複製（高さが奇数の場合）
            if height_adjustment > 0:
                adjusted_image[height:, :width, :] = image[-1:, :, :]

            # 右下角を複製（両方とも奇数の場合）
            if height_adjustment > 0 and width_adjustment > 0:
                adjusted_image[height:, width:, :] = image[-1:, -1:, :]

        return adjusted_image

    def extract(self, image: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        画像からSWT周波数変換特徴量を抽出する.

        Args:
            image (np.ndarray): 入力画像（グレースケールまたはRGB）

        Returns:
            Dict[str, Union[float, int]]: 抽出された特徴量の辞書

        Raises:
            ValueError: 画像の次元が不正な場合
            RuntimeError: SWT変換に失敗した場合
        """
        try:
            # 画像をグレースケールに変換（既にグレースケールの場合はそのまま）
            gray_image = to_grayscale(image)

            # SWT変換のために画像サイズを調整（奇数サイズを偶数サイズに）
            gray_image = self._adjust_image_size_for_swt(gray_image)

            # 画像を0-1の範囲に正規化
            if gray_image.max() > 1.0:
                gray_image = gray_image.astype(np.float32) / 255.0
            else:
                gray_image = gray_image.astype(np.float32)

            # 設定値を使用してSWT変換を実行
            max_level = self.config.get("max_level", 1)
            coeffs = pywt.swt2(
                gray_image,
                wavelet=self.config.get("wavelet", "db1"),
                level=max_level,
            )

            features = {}

            if self.config.get("multiscale", True):
                # マルチスケール解析：各レベルの特徴量を抽出
                for level_idx, level_coeffs in enumerate(coeffs, start=1):
                    level_features = self._extract_single_level_features(
                        level_coeffs, level=level_idx
                    )
                    features.update(level_features)
            else:
                # 単一スケール解析：最高レベル（最も詳細な分解レベル）のみ
                highest_level_coeffs = coeffs[-1]
                features = self._extract_single_level_features(highest_level_coeffs)

            return features

        except Exception as e:
            # より詳細なエラーメッセージを提供
            max_level = self.config.get("max_level", 1)

            error_msg = f"SWT特徴量抽出中にエラーが発生しました: {str(e)}"
            error_msg += f" (画像サイズ: {image.shape})"
            if "gray_image" in locals():
                error_msg += f" (調整後サイズ: {gray_image.shape})"
            error_msg += f" (設定レベル: {max_level})"

            raise RuntimeError(error_msg) from e

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        デフォルト設定を取得する.

        Returns:
            Dict[str, Any]: デフォルト設定の辞書
        """
        return {
            "wavelet": "db1",
            "max_level": 1,
            "multiscale": True,
        }

    @staticmethod
    def get_feature_names(config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        抽出される特徴量名のリストを取得する（単位付き）.

        Args:
            config (Optional[Dict[str, Any]]): 設定辞書。Noneの場合はデフォルト設定を使用。

        Returns:
            List[str]: 特徴量名のリスト（単位付き）.
        """
        base_names = SWTFrequencyExtractor.get_base_feature_names(config)
        return [
            f"{name}[{SWTFrequencyExtractor._get_unit_for_feature(name)}]"
            for name in base_names
        ]

    @staticmethod
    def get_base_feature_names(config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        抽出される基本特徴量名のリストを取得する（単位なし）.

        Args:
            config (Optional[Dict[str, Any]]): 設定辞書。Noneの場合はデフォルト設定を使用。

        Returns:
            List[str]: 基本特徴量名のリスト.
        """
        # 設定を取得（引数で指定されない場合はデフォルト設定を使用）
        if config is None:
            config = SWTFrequencyExtractor.get_default_config()
        else:
            # デフォルト設定とマージ
            default_config = SWTFrequencyExtractor.get_default_config()
            merged_config = default_config.copy()
            merged_config.update(config)
            config = merged_config

        # _extract_single_level_features()と同じ順序で特徴量名を生成
        components = ["ll", "lh", "hl", "hh"]
        stats = ["mean", "energy", "entropy", "std"]

        base_features = []

        # 各成分の基本統計量
        for comp in components:
            for stat in stats:
                base_features.append(f"{stat}_{comp}")

        # エネルギー比と全エネルギー
        base_features.extend(
            ["energy_ratio_h", "energy_ratio_v", "energy_ratio_d", "total_energy"]
        )

        # マルチスケール解析の場合、各レベルの特徴量を追加
        if config["multiscale"]:
            max_level = config["max_level"]
            all_features = []
            for level in range(1, max_level + 1):
                for feature in base_features:
                    all_features.append(f"L{level}_{feature}")
            return all_features
        else:
            return base_features

    @staticmethod
    def get_feature_units(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        特徴量の単位辞書を返す.

        Args:
            config (Optional[Dict[str, Any]]): 設定辞書。Noneの場合はデフォルト設定を使用。

        Returns:
            Dict[str, str]: 特徴量名と単位の対応辞書.
        """
        # 基本特徴量名を取得（設定を渡す）
        base_names = SWTFrequencyExtractor.get_base_feature_names(config)

        # 各特徴量名に対応する単位を生成
        units = {}
        for name in base_names:
            units[name] = SWTFrequencyExtractor._get_unit_for_feature(name)

        return units

    @staticmethod
    def _get_unit_for_feature(feature_name: str) -> str:
        """
        特徴量名から対応する単位を取得する.

        Args:
            feature_name (str): 特徴量名（例: "mean_ll", "L1_energy_hh"）

        Returns:
            str: 対応する単位
        """
        # マルチレベルプレフィックスを除去（L1_, L2_など）
        clean_name = feature_name
        if clean_name.startswith("L") and "_" in clean_name:
            parts = clean_name.split("_", 1)
            if len(parts) > 1 and parts[0][1:].isdigit():
                clean_name = parts[1]

        # 特徴量タイプを判定
        if clean_name.startswith("mean_"):
            return SWTFrequencyExtractor._FEATURE_UNITS["mean"]
        elif clean_name.startswith("energy_ratio_"):
            return SWTFrequencyExtractor._FEATURE_UNITS["energy_ratio"]
        elif clean_name.startswith("energy_") or clean_name == "total_energy":
            return SWTFrequencyExtractor._FEATURE_UNITS["energy"]
        elif clean_name.startswith("entropy_"):
            return SWTFrequencyExtractor._FEATURE_UNITS["entropy"]
        elif clean_name.startswith("std_"):
            return SWTFrequencyExtractor._FEATURE_UNITS["std"]
        else:
            return "unknown"
