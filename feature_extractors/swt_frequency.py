"""SWT（Stationary Wavelet Transform）周波数変換特徴量抽出を行うモジュール."""

from typing import Any, Dict, List, Union

import numpy as np
import pywt

from utils.image import to_grayscale

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
    - mean_ll: 低周波成分（LL）の平均値
    - mean_lh: 水平高周波成分（LH）の平均値
    - mean_hl: 垂直高周波成分（HL）の平均値
    - mean_hh: 対角高周波成分（HH）の平均値
    - energy_ll: 低周波成分（LL）のエネルギー
    - energy_lh: 水平高周波成分（LH）のエネルギー
    - energy_hl: 垂直高周波成分（HL）のエネルギー
    - energy_hh: 対角高周波成分（HH）のエネルギー
    - energy_ratio_h: 水平方向エネルギー比
    - energy_ratio_v: 垂直方向エネルギー比
    - energy_ratio_d: 対角方向エネルギー比
    - total_energy: 全エネルギー
    - entropy_ll: 低周波成分のエントロピー
    - entropy_lh: 水平高周波成分のエントロピー
    - entropy_hl: 垂直高周波成分のエントロピー
    - entropy_hh: 対角高周波成分のエントロピー
    - std_ll: 低周波成分の標準偏差
    - std_lh: 水平高周波成分の標準偏差
    - std_hl: 垂直高周波成分の標準偏差
    - std_hh: 対角高周波成分の標準偏差

    マルチスケール解析を行う場合、各レベルの特徴量が追加されます。
    マルチスケールでない場合は、最高レベル（最も詳細な分解レベル）の特徴量のみが抽出されます。
    """

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

            # 画像を0-1の範囲に正規化
            if gray_image.max() > 1.0:
                gray_image = gray_image.astype(np.float32) / 255.0
            else:
                gray_image = gray_image.astype(np.float32)

            # SWT変換を実行
            coeffs = pywt.swt2(
                gray_image,
                wavelet=self.config.get("wavelet", "db1"),
                level=self.config.get("max_level", 1),
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
            raise RuntimeError(
                f"SWT特徴量抽出中にエラーが発生しました: {str(e)}"
            ) from e

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
    def get_feature_names() -> List[str]:
        """
        抽出される特徴量名のリストを取得する.

        Returns:
            List[str]: 特徴量名のリスト
        """
        base_features = [
            "mean_ll",
            "mean_lh",
            "mean_hl",
            "mean_hh",
            "energy_ll",
            "energy_lh",
            "energy_hl",
            "energy_hh",
            "energy_ratio_h",
            "energy_ratio_v",
            "energy_ratio_d",
            "total_energy",
            "entropy_ll",
            "entropy_lh",
            "entropy_hl",
            "entropy_hh",
            "std_ll",
            "std_lh",
            "std_hl",
            "std_hh",
        ]
        return base_features
