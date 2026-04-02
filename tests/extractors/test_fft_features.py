"""FFT周波数領域特徴量抽出器のテストモジュール."""

import numpy as np
import pytest

from pochivision.exceptions.extractor import ExtractorValidationError
from pochivision.feature_extractors.fft_frequency import FFTFrequencyExtractor
from tests.extractors.conftest import DummyImages


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

        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image is empty or None"
        ):
            self.extractor.extract(empty_image)

    def test_extract_none_image(self):
        """None画像での例外処理をテスト."""
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image is empty or None"
        ):
            self.extractor.extract(None)

    def test_extract_invalid_shape(self):
        """無効な形状の画像での例外処理をテスト."""
        # 1次元配列
        invalid_image = np.array([1, 2, 3, 4])

        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image must be 2D or 3D"
        ):
            self.extractor.extract(invalid_image)

    def test_extract_too_small_image(self):
        """極小画像で ValueError が発生することを確認."""
        for size in [(1, 1), (2, 2), (3, 3), (3, 64), (64, 3)]:
            small_image = np.ones(size, dtype=np.uint8) * 128
            with pytest.raises(
                (ValueError, ExtractorValidationError), match="Image too small for FFT"
            ):
                self.extractor.extract(small_image)

    def test_extract_minimum_size_works(self):
        """最小サイズ (4x4) の画像が正常に処理されることを確認."""
        min_image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        features = self.extractor.extract(min_image)
        assert "high_low_ratio" in features

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
        magnitude, power_spectrum, freq_norm, angle_map = (
            self.extractor._compute_fft_data(image)
        )

        energies = self.extractor._compute_band_energy(power_spectrum, freq_norm, bands)

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
        magnitude, _, freq_norm, _ = self.extractor._compute_fft_data(image)

        centroid = self.extractor._compute_spectral_centroid(magnitude, freq_norm)

        assert isinstance(centroid, float)
        assert 0 <= centroid <= 0.5  # 正規化された周波数範囲

    def test_compute_high_low_freq_ratio(self):
        """高周波/低周波比計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        _, power_spectrum, freq_norm, _ = self.extractor._compute_fft_data(image)

        ratio = self.extractor._compute_high_low_freq_ratio(
            power_spectrum, freq_norm, 0.2
        )

        assert isinstance(ratio, float)
        assert ratio >= 0

    def test_compute_spectral_std(self):
        """スペクトル標準偏差計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        magnitude, _, freq_norm, _ = self.extractor._compute_fft_data(image)

        std = self.extractor._compute_spectral_std(magnitude, freq_norm)

        assert isinstance(std, float)
        assert std >= 0

    def test_compute_directional_energy(self):
        """方向性エネルギー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        _, power_spectrum, _, angle_map = self.extractor._compute_fft_data(image)

        # 水平方向
        horizontal = self.extractor._compute_directional_energy(
            power_spectrum, angle_map, 0.0, 10.0
        )
        assert isinstance(horizontal, float)
        assert 0 <= horizontal <= 1

        # 垂直方向
        vertical = self.extractor._compute_directional_energy(
            power_spectrum, angle_map, 90.0, 10.0
        )
        assert isinstance(vertical, float)
        assert 0 <= vertical <= 1

    def test_compute_spectral_peaks(self):
        """スペクトルピーク計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        magnitude, _, _, _ = self.extractor._compute_fft_data(image)

        num_peaks, max_peak = self.extractor._compute_spectral_peaks(magnitude, 0.1)

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
        magnitude, _, _, angle_map = self.extractor._compute_fft_data(image)

        # 水平方向エントロピー
        horizontal_entropy = self.extractor._compute_directional_entropy(
            magnitude, angle_map, 0.0, 10.0
        )
        assert isinstance(horizontal_entropy, float)
        assert horizontal_entropy >= 0

        # 垂直方向エントロピー
        vertical_entropy = self.extractor._compute_directional_entropy(
            magnitude, angle_map, 90.0, 10.0
        )
        assert isinstance(vertical_entropy, float)
        assert vertical_entropy >= 0

    def test_compute_band_entropy(self):
        """周波数帯域エントロピー計算の内部メソッドをテスト."""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        bands = [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]
        magnitude, _, freq_norm, _ = self.extractor._compute_fft_data(image)

        entropies = self.extractor._compute_band_entropy(magnitude, freq_norm, bands)

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
            "spectral_entropy[normalized]",
            "horizontal_entropy[normalized]",
            "vertical_entropy[normalized]",
            "band_1_0.00_0.10[ratio]",
            "band_2_0.10_0.30[ratio]",
            "band_3_0.30_0.50[ratio]",
            "band_1_0.00_0.10_entropy[normalized]",
            "band_2_0.10_0.30_entropy[normalized]",
            "band_3_0.30_0.50_entropy[normalized]",
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
            "spectral_entropy[normalized]",
            "horizontal_entropy[normalized]",
            "vertical_entropy[normalized]",
            "band_1_0.00_0.10[ratio]",
            "band_2_0.10_0.30[ratio]",
            "band_3_0.30_0.50[ratio]",
            "band_1_0.00_0.10_entropy[normalized]",
            "band_2_0.10_0.30_entropy[normalized]",
            "band_3_0.30_0.50_entropy[normalized]",
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
            "spectral_entropy": "normalized",
            "horizontal_entropy": "normalized",
            "vertical_entropy": "normalized",
            "band_1_0.00_0.10": "ratio",
            "band_2_0.10_0.30": "ratio",
            "band_3_0.30_0.50": "ratio",
            "band_1_0.00_0.10_entropy": "normalized",
            "band_2_0.10_0.30_entropy": "normalized",
            "band_3_0.30_0.50_entropy": "normalized",
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

    # --- band_energy ---

    def test_gradient_energy_concentrated_in_low_band(self):
        """グラデーション画像はエネルギーの 95% 以上が低帯域に集中."""
        features = self.extractor.extract(DummyImages.gradient())
        assert features["band_1_0.00_0.10"] > 0.95

    def test_stripe_energy_majority_in_mid_band(self):
        """4px ストライプは中帯域にエネルギーの 50% 以上が集中."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["band_2_0.10_0.30"] > 0.5

    def test_stripe_low_band_not_dominant(self):
        """ストライプの低帯域エネルギーは 50% 未満 (DC 除外の効果)."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["band_1_0.00_0.10"] < 0.5

    def test_band_energies_sum_to_one_square(self):
        """正方形画像の帯域エネルギー合計が ~1.0."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        features = self.extractor.extract(image)
        total = (
            features["band_1_0.00_0.10"]
            + features["band_2_0.10_0.30"]
            + features["band_3_0.30_0.50"]
        )
        assert 0.95 <= total <= 1.05, f"got {total}"

    def test_band_energies_sum_to_one_nonsquare(self):
        """非正方形画像でも帯域エネルギー合計が ~1.0."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (32, 96), dtype=np.uint8)
        features = self.extractor.extract(image)
        total = (
            features["band_1_0.00_0.10"]
            + features["band_2_0.10_0.30"]
            + features["band_3_0.30_0.50"]
        )
        assert 0.95 <= total <= 1.05, f"got {total}"

    # --- high_low_ratio ---

    def test_gradient_high_low_ratio_near_zero(self):
        """グラデーション画像の high_low_ratio は ~0 (低周波のみ)."""
        features = self.extractor.extract(DummyImages.gradient())
        assert features["high_low_ratio"] < 0.01

    def test_stripe_high_low_ratio_above_one(self):
        """ストライプ画像の high_low_ratio は > 1 (高周波 > 低周波)."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["high_low_ratio"] > 1.0

    # --- spectral_std ---

    def test_spectral_std_uniform_is_near_zero(self):
        """単色画像のスペクトル標準偏差はランダム画像より十分小さい."""
        features_uniform = self.extractor.extract(DummyImages.uniform())
        np.random.seed(42)
        features_random = self.extractor.extract(
            np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        )
        assert features_uniform["spectral_std"] < features_random["spectral_std"]

    def test_spectral_std_random_is_positive(self):
        """ランダム画像のスペクトル標準偏差は > 0."""
        np.random.seed(42)
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        features = self.extractor.extract(image)
        assert features["spectral_std"] > 0.01

    # --- directional_energy ---
    # 水平ストライプは垂直方向の周波数成分を持つ (FFT の性質).
    # DC 除外後は方向性エネルギーがパターンを直接反映する.

    def test_h_stripe_vertical_energy_above_70pct(self):
        """水平ストライプの vertical_energy は 70% 以上."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["vertical_energy"] > 0.7

    def test_h_stripe_vertical_greater_than_horizontal(self):
        """水平ストライプは vertical_energy > horizontal_energy."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["vertical_energy"] > features["horizontal_energy"]

    def test_v_stripe_horizontal_greater_than_vertical(self):
        """垂直ストライプは horizontal_energy > vertical_energy."""
        features = self.extractor.extract(DummyImages.v_stripe())
        assert features["horizontal_energy"] > features["vertical_energy"]

    # --- spectral_centroid ---

    def test_gradient_centroid_below_005(self):
        """グラデーション画像のスペクトル重心は < 0.05 (低周波に集中)."""
        features = self.extractor.extract(DummyImages.gradient())
        assert features["spectral_centroid"] < 0.05

    def test_stripe_centroid_near_025(self):
        """ストライプ画像のスペクトル重心は 0.1-0.3 付近."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert 0.1 < features["spectral_centroid"] < 0.3

    # --- num_peaks / max_peak_amp ---

    def test_num_peaks_positive_for_stripe(self):
        """ストライプ画像のピーク数は 1 以上."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["num_peaks"] >= 1

    def test_max_peak_amp_positive(self):
        """max_peak_amp は非ゼロ画像で正の値."""
        features = self.extractor.extract(DummyImages.h_stripe())
        assert features["max_peak_amp"] > 0

    # --- spectral_entropy ---

    def test_spectral_entropy_random_above_08(self):
        """ランダム画像の正規化エントロピーは > 0.8 (高エントロピー)."""
        np.random.seed(42)
        features = self.extractor.extract(
            np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        )
        assert features["spectral_entropy"] > 0.8

    def test_spectral_entropy_stripe_below_random(self):
        """ストライプ画像のエントロピーはランダムより低い."""
        np.random.seed(42)
        features_random = self.extractor.extract(
            np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        )
        features_stripe = self.extractor.extract(DummyImages.h_stripe())
        assert features_stripe["spectral_entropy"] < features_random["spectral_entropy"]

    # --- directional_entropy ---

    def test_directional_entropy_h_stripe_more_vertical(self):
        """水平ストライプは垂直方向のエントロピーが垂直ストライプより大きい."""
        features_h = self.extractor.extract(DummyImages.h_stripe())
        features_v = self.extractor.extract(DummyImages.v_stripe())
        assert features_h["vertical_entropy"] > features_v["vertical_entropy"]

    # --- band_entropy ---

    def test_band_entropy_all_positive_for_random(self):
        """ランダム画像は全帯域でエントロピーが正."""
        np.random.seed(42)
        features = self.extractor.extract(
            np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        )
        assert features["band_1_0.00_0.10_entropy"] > 0
        assert features["band_2_0.10_0.30_entropy"] > 0
        assert features["band_3_0.30_0.50_entropy"] > 0
