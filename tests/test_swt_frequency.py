"""SWT周波数変換特徴量抽出器のテストモジュール."""

import numpy as np
import pytest  # noqa: F401

from feature_extractors.swt_frequency import SWTFrequencyExtractor


class TestSWTFrequencyExtractor:
    """SWTFrequencyExtractorのテストクラス."""

    def test_init_default(self):
        """デフォルト設定でのインスタンス化をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        assert extractor.name == "swt_frequency"
        assert extractor.config["wavelet"] == "db1"
        assert extractor.config["max_level"] == 1
        assert extractor.config["multiscale"] is True

    def test_init_custom_config(self):
        """カスタム設定でのインスタンス化をテストする."""
        config = {
            "wavelet": "db4",
            "max_level": 2,
            "multiscale": False,
        }
        extractor = SWTFrequencyExtractor("swt_frequency", config)
        assert extractor.config["wavelet"] == "db4"
        assert extractor.config["max_level"] == 2
        assert extractor.config["multiscale"] is False

    def test_extract_grayscale_image(self):
        """グレースケール画像からの特徴量抽出をテストする."""
        # テスト用のグレースケール画像を作成
        image = np.random.rand(64, 64).astype(np.float32)

        extractor = SWTFrequencyExtractor("swt_frequency", {})
        features = extractor.extract(image)

        # 基本的な特徴量が存在することを確認
        expected_features = [
            "L1_mean_ll",
            "L1_mean_lh",
            "L1_mean_hl",
            "L1_mean_hh",
            "L1_energy_ll",
            "L1_energy_lh",
            "L1_energy_hl",
            "L1_energy_hh",
            "L1_energy_ratio_h",
            "L1_energy_ratio_v",
            "L1_energy_ratio_d",
            "L1_total_energy",
            "L1_entropy_ll",
            "L1_entropy_lh",
            "L1_entropy_hl",
            "L1_entropy_hh",
            "L1_std_ll",
            "L1_std_lh",
            "L1_std_hl",
            "L1_std_hh",
        ]

        for feature_name in expected_features:
            assert feature_name in features
            assert isinstance(features[feature_name], (int, float))
            assert not np.isnan(features[feature_name])

    def test_extract_rgb_image(self):
        """RGB画像からの特徴量抽出をテストする."""
        # テスト用のRGB画像を作成
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        extractor = SWTFrequencyExtractor("swt_frequency", {})
        features = extractor.extract(image)

        # 特徴量が正常に抽出されることを確認
        assert len(features) > 0
        assert all(isinstance(v, (int, float)) for v in features.values())

    def test_extract_single_scale(self):
        """単一スケール解析をテストする."""
        image = np.random.rand(64, 64).astype(np.float32)

        config = {"multiscale": False}
        extractor = SWTFrequencyExtractor("swt_frequency", config)
        features = extractor.extract(image)

        # 単一スケールの場合、レベルプレフィックスがないことを確認
        expected_features = [
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

        for feature_name in expected_features:
            assert feature_name in features

    def test_extract_multiscale(self):
        """マルチスケール解析をテストする."""
        image = np.random.rand(64, 64).astype(np.float32)

        config = {"max_level": 2, "multiscale": True}
        extractor = SWTFrequencyExtractor("swt_frequency", config)
        features = extractor.extract(image)

        # マルチスケールの場合、各レベルの特徴量が存在することを確認
        level1_features = [k for k in features.keys() if k.startswith("L1_")]
        level2_features = [k for k in features.keys() if k.startswith("L2_")]

        assert len(level1_features) > 0
        assert len(level2_features) > 0

    def test_compute_mean(self):
        """平均値計算をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean_val = extractor._compute_mean(coeffs)

        expected_mean = 2.5
        assert abs(mean_val - expected_mean) < 1e-6

    def test_compute_energy(self):
        """エネルギー計算をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        energy = extractor._compute_energy(coeffs)

        expected_energy = 1.0 + 4.0 + 9.0 + 16.0  # 1^2 + 2^2 + 3^2 + 4^2
        assert abs(energy - expected_energy) < 1e-6

    def test_compute_std(self):
        """標準偏差計算をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        std_val = extractor._compute_std(coeffs)

        expected_std = np.std(coeffs)
        assert abs(std_val - expected_std) < 1e-6

    def test_compute_entropy(self):
        """エントロピー計算をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        # 均一分布のテストデータ
        coeffs = np.linspace(0.0, 1.0, 256).reshape(16, 16)
        entropy = extractor._compute_entropy(coeffs)

        # エントロピーは正の値であることを確認
        assert entropy >= 0.0
        assert not np.isnan(entropy)

    def test_compute_entropy_constant_values(self):
        """定数値でのエントロピー計算をテストする."""
        extractor = SWTFrequencyExtractor("swt_frequency", {})
        # すべて同じ値の配列
        coeffs = np.ones((16, 16))
        entropy = extractor._compute_entropy(coeffs)

        # すべて同じ値の場合、エントロピーは0
        assert entropy == 0.0

    def test_get_default_config(self):
        """デフォルト設定の取得をテストする."""
        config = SWTFrequencyExtractor.get_default_config()

        assert config["wavelet"] == "db1"
        assert config["max_level"] == 1
        assert config["multiscale"] is True

    def test_get_feature_names(self):
        """特徴量名リストの取得をテストする."""
        feature_names = SWTFrequencyExtractor.get_feature_names()

        expected_names = [
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

        assert feature_names == expected_names

    def test_different_wavelets(self):
        """異なるウェーブレットでの動作をテストする."""
        image = np.random.rand(64, 64).astype(np.float32)

        wavelets = ["db1", "db4", "haar"]

        for wavelet in wavelets:
            config = {"wavelet": wavelet}
            extractor = SWTFrequencyExtractor("swt_frequency", config)
            features = extractor.extract(image)

            # 各ウェーブレットで特徴量が正常に抽出されることを確認
            assert len(features) > 0
            assert all(isinstance(v, (int, float)) for v in features.values())
            assert all(not np.isnan(v) for v in features.values())

    def test_grayscale_input_compatibility(self):
        """グレースケール入力の互換性をテストする."""
        # 2次元グレースケール画像
        gray_2d = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        # 3次元1チャンネル画像
        gray_3d = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)

        extractor = SWTFrequencyExtractor("swt_frequency", {})

        # 両方とも正常に処理されることを確認
        features_2d = extractor.extract(gray_2d)
        features_3d = extractor.extract(gray_3d)

        assert len(features_2d) > 0
        assert len(features_3d) > 0
        assert all(isinstance(v, (int, float)) for v in features_2d.values())
        assert all(isinstance(v, (int, float)) for v in features_3d.values())
