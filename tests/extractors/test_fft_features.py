"""FFT周波数領域特徴量抽出器のテストモジュール."""

import numpy as np
import pytest

from feature_extractors.fft_frequency import FFTFrequencyExtractor


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
            "band_1_0.00_0.10",
            "band_2_0.10_0.30",
            "band_3_0.30_0.50",
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
            "high_low_ratio",
            "spectral_std",
            "horizontal_energy",
            "vertical_energy",
            "num_peaks",
            "max_peak_amp",
            "spectral_centroid",
            "band_1_0.00_0.10",
            "band_2_0.10_0.30",
            "band_3_0.30_0.50",
        ]

        assert len(feature_names) == len(expected_features)
        for feature_name in expected_features:
            assert feature_name in feature_names

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
