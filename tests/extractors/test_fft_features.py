"""FFT周波数領域特徴量抽出器のテストモジュール."""

import numpy as np
import pytest

from pochivision.feature_extractors.fft_frequency import FFTFrequencyExtractor


class TestFFTFrequencyExtractor:
    """FFTFrequencyExtractorのテストクラス."""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理."""
        self.extractor = FFTFrequencyExtractor()

    def test_init_default(self):
        """デフォルト設定でのインスタンス化をテスト."""
        extractor = FFTFrequencyExtractor()
        assert extractor.name == "fft_frequency"
        assert extractor.frequency_bands == [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]
        assert extractor.high_low_threshold == 0.2
        assert extractor.directional_tolerance == 10.0
        assert extractor.peak_threshold_ratio == 0.1
        assert extractor.mm_per_pixel is None

    def test_init_custom_config(self):
        """カスタム設定でのインスタンス化をテスト."""
        config = {
            "frequency_bands": [[0.0, 0.2], [0.2, 0.4], [0.4, 0.5]],
            "high_low_threshold": 0.3,
            "directional_tolerance": 15.0,
            "peak_threshold_ratio": 0.2,
            "mm_per_pixel": 0.05,
        }
        extractor = FFTFrequencyExtractor(config=config)
        assert extractor.frequency_bands == [[0.0, 0.2], [0.2, 0.4], [0.4, 0.5]]
        assert extractor.high_low_threshold == 0.3
        assert extractor.directional_tolerance == 15.0
        assert extractor.peak_threshold_ratio == 0.2
        assert extractor.mm_per_pixel == 0.05

    def test_extract_grayscale_image(self):
        """グレースケール画像からの特徴量抽出をテスト."""
        # テスト用のグレースケール画像を作成（チェッカーボードパターン）
        image = np.zeros((64, 64), dtype=np.uint8)
        image[::8, ::8] = 255  # チェッカーボードパターン

        features = self.extractor.extract(image)

        # 期待される特徴量名が含まれているかチェック
        expected_features = [
            "high_low_ratio",
            "spectral_std",
            "horizontal_energy",
            "vertical_energy",
            "num_peaks",
            "max_peak_amp",
            "spectral_centroid",
            "spectral_entropy",
            "horizontal_entropy",
            "vertical_entropy",
            "band_1_0.00_0.10",
            "band_2_0.10_0.30",
            "band_3_0.30_0.50",
            "band_1_0.00_0.10_entropy",
            "band_2_0.10_0.30_entropy",
            "band_3_0.30_0.50_entropy",
        ]

        for feature_name in expected_features:
            assert feature_name in features
            assert isinstance(features[feature_name], (int, float))
            assert not np.isnan(features[feature_name])
            assert not np.isinf(features[feature_name])

    def test_extract_color_image(self):
        """カラー画像からの特徴量抽出をテスト（BGR→グレースケール変換）."""
        # テスト用のカラー画像を作成
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        features = self.extractor.extract(image)

        # 基本的な特徴量が抽出されているかチェック
        assert "high_low_ratio" in features
        assert "spectral_std" in features
        assert "horizontal_energy" in features
        assert "vertical_energy" in features

    def test_extract_uniform_image(self):
        """均一な画像からの特徴量抽出をテスト."""
        # 均一な画像（すべて同じ値）
        image = np.full((64, 64), 128, dtype=np.uint8)

        features = self.extractor.extract(image)

        # 均一な画像では多くの特徴量が0または特定の値になる
        assert features["high_low_ratio"] >= 0
        assert features["spectral_std"] >= 0
        assert features["num_peaks"] >= 0

    def test_extract_with_mm_per_pixel(self):
        """mm_per_pixel設定での特徴量抽出をテスト."""
        config = {"mm_per_pixel": 0.05}
        extractor = FFTFrequencyExtractor(config=config)

        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        features = extractor.extract(image)

        # スケールが適用された特徴量をチェック
        assert "spectral_std" in features
        assert "spectral_centroid" in features
        assert features["spectral_std"] >= 0
        assert features["spectral_centroid"] >= 0

    def test_extract_empty_image(self):
        """空の画像での例外処理をテスト."""
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Input image is empty or None"):
            self.extractor.extract(empty_image)

    def test_extract_none_image(self):
        """None画像での例外処理をテスト."""
        with pytest.raises(ValueError, match="Input image is empty or None"):
            self.extractor.extract(None)

    def test_extract_invalid_shape(self):
        """無効な形状の画像での例外処理をテスト."""
        # 1次元配列
        invalid_image = np.array([1, 2, 3, 4])

        with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
            self.extractor.extract(invalid_image)

    def test_extract_different_dtypes(self):
        """異なるデータ型の画像での特徴量抽出をテスト."""
        base_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        # float32
        float_image = base_image.astype(np.float32) / 255.0
        features_float = self.extractor.extract(float_image)
        assert "high_low_ratio" in features_float

        # int32
        int_image = base_image.astype(np.int32)
        features_int = self.extractor.extract(int_image)
        assert "high_low_ratio" in features_int

    def test_compute_band_energy(self):
        """周波数帯域エネルギー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        bands = [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]

        energies = self.extractor._compute_band_energy(image, bands)

        assert len(energies) == 3
        assert "band_1_0.00_0.10" in energies
        assert "band_2_0.10_0.30" in energies
        assert "band_3_0.30_0.50" in energies

        # エネルギーの合計は1に近い（正規化されているため）
        total_energy = sum(energies.values())
        assert 0.9 <= total_energy <= 1.1

    def test_compute_spectral_centroid(self):
        """スペクトル重心計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        centroid = self.extractor._compute_spectral_centroid(image)

        assert isinstance(centroid, float)
        assert 0 <= centroid <= 0.5  # 正規化された周波数範囲

    def test_compute_high_low_freq_ratio(self):
        """高周波/低周波比計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        ratio = self.extractor._compute_high_low_freq_ratio(image, 0.2)

        assert isinstance(ratio, float)
        assert ratio >= 0

    def test_compute_spectral_std(self):
        """スペクトル標準偏差計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        std = self.extractor._compute_spectral_std(image)

        assert isinstance(std, float)
        assert std >= 0

    def test_compute_directional_energy(self):
        """方向性エネルギー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        # 水平方向
        horizontal = self.extractor._compute_directional_energy(image, 0.0, 10.0)
        assert isinstance(horizontal, float)
        assert 0 <= horizontal <= 1

        # 垂直方向
        vertical = self.extractor._compute_directional_energy(image, 90.0, 10.0)
        assert isinstance(vertical, float)
        assert 0 <= vertical <= 1

    def test_compute_spectral_peaks(self):
        """スペクトルピーク計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        num_peaks, max_peak = self.extractor._compute_spectral_peaks(image, 0.1)

        assert isinstance(num_peaks, int)
        assert isinstance(max_peak, float)
        assert num_peaks >= 0
        assert max_peak >= 0

    def test_compute_spectral_entropy(self):
        """スペクトラルエントロピー計算の内部メソッドをテスト."""
        # テスト用のスペクトラム振幅を作成
        magnitude = np.random.rand(32, 32) * 100

        entropy = self.extractor._compute_spectral_entropy(magnitude)

        assert isinstance(entropy, float)
        assert entropy >= 0  # エントロピーは非負

        # 均一な分布では高いエントロピー
        uniform_magnitude = np.ones((32, 32))
        uniform_entropy = self.extractor._compute_spectral_entropy(uniform_magnitude)
        assert uniform_entropy > 0

        # ゼロ配列では0エントロピー
        zero_magnitude = np.zeros((32, 32))
        zero_entropy = self.extractor._compute_spectral_entropy(zero_magnitude)
        assert zero_entropy == 0.0

    def test_compute_directional_entropy(self):
        """方向性エントロピー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        # 水平方向エントロピー
        horizontal_entropy = self.extractor._compute_directional_entropy(
            image, 0.0, 10.0
        )
        assert isinstance(horizontal_entropy, float)
        assert horizontal_entropy >= 0

        # 垂直方向エントロピー
        vertical_entropy = self.extractor._compute_directional_entropy(
            image, 90.0, 10.0
        )
        assert isinstance(vertical_entropy, float)
        assert vertical_entropy >= 0

    def test_compute_band_entropy(self):
        """周波数帯域エントロピー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        bands = [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]

        entropies = self.extractor._compute_band_entropy(image, bands)

        assert len(entropies) == 3
        assert "band_1_0.00_0.10_entropy" in entropies
        assert "band_2_0.10_0.30_entropy" in entropies
        assert "band_3_0.30_0.50_entropy" in entropies

        # すべてのエントロピーが非負であることを確認
        for entropy in entropies.values():
            assert isinstance(entropy, float)
            assert entropy >= 0

    def test_get_default_config(self):
        """デフォルト設定の取得をテスト."""
        config = FFTFrequencyExtractor.get_default_config()

        expected_keys = [
            "frequency_bands",
            "high_low_threshold",
            "directional_tolerance",
            "peak_threshold_ratio",
            "mm_per_pixel",
        ]

        for key in expected_keys:
            assert key in config

        assert config["frequency_bands"] == [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]
        assert config["high_low_threshold"] == 0.2
        assert config["directional_tolerance"] == 10.0
        assert config["peak_threshold_ratio"] == 0.1
        assert config["mm_per_pixel"] is None

    def test_get_feature_names(self):
        """特徴量名リストの取得をテスト."""
        feature_names = FFTFrequencyExtractor.get_feature_names()

        expected_features = [
            "high_low_ratio[ratio]",
            "spectral_std[cycle/mm_or_cycle/pixel]",
            "horizontal_energy[ratio]",
            "vertical_energy[ratio]",
            "num_peaks[count]",
            "max_peak_amp[amplitude]",
            "spectral_centroid[cycle/mm_or_cycle/pixel]",
            "spectral_entropy[bits]",
            "horizontal_entropy[bits]",
            "vertical_entropy[bits]",
            "band_1_0.00_0.10[ratio]",
            "band_2_0.10_0.30[ratio]",
            "band_3_0.30_0.50[ratio]",
            "band_1_0.00_0.10_entropy[bits]",
            "band_2_0.10_0.30_entropy[bits]",
            "band_3_0.30_0.50_entropy[bits]",
        ]

        assert len(feature_names) == len(expected_features)
        for feature_name in expected_features:
            assert feature_name in feature_names

    def test_feature_names_and_units(self):
        """特徴量名と単位の包括的なテスト."""
        print("\n=== FFT特徴量名・単位テスト ===")

        # 基本特徴量名の確認
        base_names = FFTFrequencyExtractor.get_base_feature_names()
        expected_base_names = [
            "high_low_ratio",
            "spectral_std",
            "horizontal_energy",
            "vertical_energy",
            "num_peaks",
            "max_peak_amp",
            "spectral_centroid",
            "spectral_entropy",
            "horizontal_entropy",
            "vertical_entropy",
            "band_1_0.00_0.10",
            "band_2_0.10_0.30",
            "band_3_0.30_0.50",
            "band_1_0.00_0.10_entropy",
            "band_2_0.10_0.30_entropy",
            "band_3_0.30_0.50_entropy",
        ]
        print(f"基本特徴量名: {base_names}")
        assert (
            base_names == expected_base_names
        ), f"Expected {expected_base_names}, got {base_names}"

        # 単位付き特徴量名の確認
        unit_names = FFTFrequencyExtractor.get_feature_names()
        expected_unit_names = [
            "high_low_ratio[ratio]",
            "spectral_std[cycle/mm_or_cycle/pixel]",
            "horizontal_energy[ratio]",
            "vertical_energy[ratio]",
            "num_peaks[count]",
            "max_peak_amp[amplitude]",
            "spectral_centroid[cycle/mm_or_cycle/pixel]",
            "spectral_entropy[bits]",
            "horizontal_entropy[bits]",
            "vertical_entropy[bits]",
            "band_1_0.00_0.10[ratio]",
            "band_2_0.10_0.30[ratio]",
            "band_3_0.30_0.50[ratio]",
            "band_1_0.00_0.10_entropy[bits]",
            "band_2_0.10_0.30_entropy[bits]",
            "band_3_0.30_0.50_entropy[bits]",
        ]
        print(f"単位付き特徴量名: {unit_names}")
        assert (
            unit_names == expected_unit_names
        ), f"Expected {expected_unit_names}, got {unit_names}"

        # 単位辞書の確認
        units = FFTFrequencyExtractor.get_feature_units()
        expected_units = {
            "high_low_ratio": "ratio",
            "spectral_std": "cycle/mm_or_cycle/pixel",
            "horizontal_energy": "ratio",
            "vertical_energy": "ratio",
            "num_peaks": "count",
            "max_peak_amp": "amplitude",
            "spectral_centroid": "cycle/mm_or_cycle/pixel",
            "spectral_entropy": "bits",
            "horizontal_entropy": "bits",
            "vertical_entropy": "bits",
            "band_1_0.00_0.10": "ratio",
            "band_2_0.10_0.30": "ratio",
            "band_3_0.30_0.50": "ratio",
            "band_1_0.00_0.10_entropy": "bits",
            "band_2_0.10_0.30_entropy": "bits",
            "band_3_0.30_0.50_entropy": "bits",
        }
        print(f"特徴量単位辞書: {units}")
        assert units == expected_units, f"Expected {expected_units}, got {units}"

        # 抽出結果と特徴量名の整合性確認
        test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        features = self.extractor.extract(test_image)

        # 抽出された特徴量のキーが基本特徴量名と一致することを確認
        feature_keys = list(features.keys())
        print(f"抽出された特徴量のキー: {feature_keys}")
        assert set(feature_keys) == set(
            base_names
        ), f"Feature keys {feature_keys} don't match base names {base_names}"

        print("FFT特徴量名・単位テスト: 成功")

    def test_feature_consistency(self):
        """同じ画像に対して一貫した結果が得られることをテスト."""
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        features1 = self.extractor.extract(image)
        features2 = self.extractor.extract(image)

        # 同じ画像なので同じ結果が得られるはず
        for key in features1:
            assert abs(features1[key] - features2[key]) < 1e-10

    def test_error_handling_in_extract(self):
        """extract メソッドでのエラーハンドリングをテスト."""
        # 正常な画像で一度テスト
        normal_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        features = self.extractor.extract(normal_image)

        # すべての特徴量が数値であることを確認
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
