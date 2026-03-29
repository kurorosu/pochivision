"""LBPテクスチャ特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from pochivision.feature_extractors import LBPTextureExtractor, get_feature_extractor


def test_lbp_texture_basic():
    """LBPテクスチャ特徴量抽出の基本テスト."""
    print("=== LBPテクスチャ特徴量抽出基本テスト ===")

    # テスト画像の作成（チェッカーボードパターン）
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i // 8 + j // 8) % 2 == 0:
                test_image[i : i + 8, j : j + 8, :] = 255

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = LBPTextureExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.6f}")

    # 基本的な特徴量が含まれているかチェック
    expected_features = [
        "lbp_mean",
        "lbp_std",
        "lbp_skewness",
        "lbp_kurtosis",
        "lbp_entropy",
        "lbp_energy",
    ]
    for feature in expected_features:
        assert feature in features, f"特徴量 {feature} が見つかりません"

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("lbp", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for name, value in features2.items():
        print(f"  {name}: {value:.6f}")

    # 結果が一致することを確認
    for key in features:
        assert (
            abs(features[key] - features2[key]) < 1e-10
        ), f"特徴量 {key} の値が一致しません"


def test_lbp_different_parameters():
    """異なるパラメータでのLBP特徴量抽出テスト."""
    print("\n=== 異なるパラメータでのLBPテスト ===")

    # テスト画像（ランダムテクスチャ）
    np.random.seed(42)
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # 1. デフォルト設定（P=8, R=1, uniform）
    print("\n--- デフォルト設定 (P=8, R=1, uniform) ---")
    extractor_default = LBPTextureExtractor()
    features_default = extractor_default.extract(test_image)
    for name, value in features_default.items():
        print(f"  {name}: {value:.6f}")

    # 2. 異なる近傍点数（P=16）
    print("\n--- P=16での設定 ---")
    config_p16 = {"P": 16, "R": 2}
    extractor_p16 = get_feature_extractor("lbp", config_p16)
    features_p16 = extractor_p16.extract(test_image)
    for name, value in features_p16.items():
        print(f"  {name}: {value:.6f}")

    # 3. 通常のLBP（non-uniform）
    print("\n--- 通常のLBP (method=default) ---")
    config_default_method = {"method": "default", "P": 8, "R": 1}
    extractor_default_method = get_feature_extractor("lbp", config_default_method)
    features_default_method = extractor_default_method.extract(test_image)
    for name, value in features_default_method.items():
        print(f"  {name}: {value:.6f}")

    # 4. 回転不変LBP
    print("\n--- 回転不変LBP (method=ror) ---")
    config_ror = {"method": "ror", "P": 8, "R": 1}
    extractor_ror = get_feature_extractor("lbp", config_ror)
    features_ror = extractor_ror.extract(test_image)
    for name, value in features_ror.items():
        print(f"  {name}: {value:.6f}")


def test_lbp_with_histogram():
    """ヒストグラム含有オプションのテスト."""
    print("\n=== ヒストグラム含有オプションテスト ===")

    # テスト画像
    test_image = np.full((64, 64, 3), 128, dtype=np.uint8)
    # 一部にパターンを追加
    test_image[20:40, 20:40, :] = 200

    # 1. ヒストグラムなし（デフォルト）
    print("\n--- ヒストグラムなし ---")
    extractor_no_hist = LBPTextureExtractor()
    features_no_hist = extractor_no_hist.extract(test_image)
    print(f"特徴量数: {len(features_no_hist)}")
    for name, value in features_no_hist.items():
        print(f"  {name}: {value:.6f}")

    # 2. ヒストグラムあり
    print("\n--- ヒストグラムあり ---")
    config_with_hist = {"include_histogram": True}
    extractor_with_hist = get_feature_extractor("lbp", config_with_hist)
    features_with_hist = extractor_with_hist.extract(test_image)
    print(f"特徴量数: {len(features_with_hist)}")

    # 統計量のみ表示
    stat_features = {
        k: v for k, v in features_with_hist.items() if not k.startswith("lbp_bin_")
    }
    print("統計量:")
    for name, value in stat_features.items():
        print(f"  {name}: {value:.6f}")

    # ヒストグラムビンの数を確認
    hist_features = {
        k: v for k, v in features_with_hist.items() if k.startswith("lbp_bin_")
    }
    print(f"ヒストグラムビン数: {len(hist_features)}")
    print("ヒストグラム（最初の10ビン）:")
    for i, (name, value) in enumerate(list(hist_features.items())[:10]):
        print(f"  {name}: {value:.6f}")


def test_lbp_resize_options():
    """リサイズオプションのテスト."""
    print("\n=== リサイズオプションテスト ===")

    # 異なるサイズのテスト画像
    test_image_small = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    test_image_large = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # 1. デフォルトリサイズ（128x128）
    print("\n--- デフォルトリサイズ (128x128) ---")
    extractor_default = LBPTextureExtractor()

    features_small = extractor_default.extract(test_image_small)
    features_large = extractor_default.extract(test_image_large)

    print("小画像 (32x32) -> リサイズ後:")
    for name, value in features_small.items():
        print(f"  {name}: {value:.6f}")

    print("大画像 (256x256) -> リサイズ後:")
    for name, value in features_large.items():
        print(f"  {name}: {value:.6f}")

    # 2. カスタムリサイズ（64x64）
    print("\n--- カスタムリサイズ (64x64) ---")
    config_custom_resize = {"resize_shape": [64, 64]}
    extractor_custom = get_feature_extractor("lbp", config_custom_resize)
    features_custom = extractor_custom.extract(test_image_large)

    print("大画像 (256x256) -> 64x64リサイズ後:")
    for name, value in features_custom.items():
        print(f"  {name}: {value:.6f}")

    # 3. リサイズなし
    print("\n--- リサイズなし ---")
    config_no_resize = {"resize_shape": None}
    extractor_no_resize = get_feature_extractor("lbp", config_no_resize)
    features_no_resize = extractor_no_resize.extract(test_image_small)

    print("小画像 (32x32) -> リサイズなし:")
    for name, value in features_no_resize.items():
        print(f"  {name}: {value:.6f}")


def test_lbp_grayscale_input():
    """グレースケール画像入力のテスト."""
    print("\n=== グレースケール画像入力テスト ===")

    # グレースケールテスト画像
    gray_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    extractor = LBPTextureExtractor()
    features = extractor.extract(gray_image)

    print("グレースケール画像の特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.6f}")

    # カラー画像と比較
    color_image = np.stack([gray_image, gray_image, gray_image], axis=2)
    features_color = extractor.extract(color_image)

    print("\n同じ内容のカラー画像の特徴量:")
    for name, value in features_color.items():
        print(f"  {name}: {value:.6f}")

    # 結果がほぼ同じであることを確認
    for key in features:
        diff = abs(features[key] - features_color[key])
        assert (
            diff < 1e-6
        ), f"グレースケールとカラーで特徴量 {key} の値が大きく異なります: {diff}"


def test_lbp_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = LBPTextureExtractor()

    # 1. 単色画像
    print("\n--- 単色画像 ---")
    uniform_image = np.full((64, 64, 3), 128, dtype=np.uint8)
    features_uniform = extractor.extract(uniform_image)
    print("単色画像の特徴量:")
    for name, value in features_uniform.items():
        print(f"  {name}: {value:.6f}")

    # 2. 白画像
    print("\n--- 白画像 ---")
    white_image = np.full((64, 64, 3), 255, dtype=np.uint8)
    features_white = extractor.extract(white_image)
    print("白画像の特徴量:")
    for name, value in features_white.items():
        print(f"  {name}: {value:.6f}")

    # 3. 黒画像
    print("\n--- 黒画像 ---")
    black_image = np.zeros((64, 64, 3), dtype=np.uint8)
    features_black = extractor.extract(black_image)
    print("黒画像の特徴量:")
    for name, value in features_black.items():
        print(f"  {name}: {value:.6f}")

    # 4. 最小サイズ画像
    print("\n--- 最小サイズ画像 ---")
    tiny_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    features_tiny = extractor.extract(tiny_image)
    print("最小サイズ画像の特徴量:")
    for name, value in features_tiny.items():
        print(f"  {name}: {value:.6f}")


def test_lbp_error_handling():
    """エラーハンドリングのテスト."""
    extractor = LBPTextureExtractor()

    # 空の画像で ValueError
    with pytest.raises(ValueError, match="empty or None"):
        extractor.extract(np.array([]))

    # None画像で ValueError
    with pytest.raises(ValueError, match="empty or None"):
        extractor.extract(None)


def test_lbp_feature_names_and_config():
    """特徴量名とデフォルト設定のテスト."""
    print("\n=== 特徴量名とデフォルト設定テスト ===")

    # 特徴量名の取得（単位付き）
    feature_names = LBPTextureExtractor.get_feature_names()
    print(f"特徴量名リスト（単位付き）: {feature_names}")
    print(f"特徴量数: {len(feature_names)}")

    # 特徴量名の取得（単位なし）
    feature_names_no_units = LBPTextureExtractor.get_base_feature_names()
    print(f"特徴量名リスト（単位なし）: {feature_names_no_units}")

    # 特徴量の単位辞書の取得
    feature_units = LBPTextureExtractor.get_feature_units()
    print(f"特徴量の単位辞書: {feature_units}")

    # 個別の特徴量の単位取得テスト
    test_features = ["lbp_mean", "lbp_entropy", "lbp_energy", "nonexistent_feature"]
    for feature in test_features:
        unit = LBPTextureExtractor._get_unit_for_feature(feature)
        print(f"特徴量 '{feature}' の単位: {unit}")

    # 単位付きと単位なしの特徴量名の数が一致することを確認
    assert len(feature_names) == len(
        feature_names_no_units
    ), "単位付きと単位なしの特徴量名の数が一致しません"

    # 単位付きの特徴量名が正しい形式であることを確認
    for i, (name_with_unit, name_without_unit) in enumerate(
        zip(feature_names, feature_names_no_units)
    ):
        expected_unit = feature_units.get(name_without_unit, "unknown")
        expected_name_with_unit = f"{name_without_unit}[{expected_unit}]"
        assert (
            name_with_unit == expected_name_with_unit
        ), f"特徴量名 {i}: 期待値 '{expected_name_with_unit}', 実際 '{name_with_unit}'"

    # デフォルト設定の取得
    default_config = LBPTextureExtractor.get_default_config()
    print(f"デフォルト設定: {default_config}")

    # 実際の抽出結果と特徴量名（単位なし）が一致するかチェック
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    extractor = LBPTextureExtractor()
    features = extractor.extract(test_image)

    extracted_names = set(features.keys())
    expected_names = set(feature_names_no_units)

    print(f"抽出された特徴量名: {sorted(extracted_names)}")
    print(f"期待される特徴量名（単位なし）: {sorted(expected_names)}")

    assert (
        extracted_names == expected_names
    ), "抽出された特徴量名と期待される特徴量名（単位なし）が一致しません"
    print("特徴量名の一致を確認しました")

    # 基本的な単位の確認
    expected_units = {
        "lbp_mean": "pattern_index",
        "lbp_std": "pattern_index",
        "lbp_skewness": "dimensionless",
        "lbp_kurtosis": "dimensionless",
        "lbp_entropy": "normalized",
        "lbp_energy": "ratio",
    }

    for feature, expected_unit in expected_units.items():
        actual_unit = LBPTextureExtractor._get_unit_for_feature(feature)
        assert (
            actual_unit == expected_unit
        ), f"特徴量 '{feature}' の単位が期待値と異なります: 期待値 '{expected_unit}', 実際 '{actual_unit}'"

    print("基本特徴量の単位確認が完了しました")


def test_lbp_units_with_histogram():
    """ヒストグラムを含む場合の単位管理機能のテスト."""
    print("\n=== ヒストグラム含有時の単位管理テスト ===")

    # ヒストグラムを含む設定
    config_with_histogram = {"include_histogram": True}
    extractor = LBPTextureExtractor(config=config_with_histogram)

    # テスト画像
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    features = extractor.extract(test_image)

    # 特徴量名の取得（単位付き、インスタンス設定反映）
    feature_names = extractor.get_feature_names_instance()
    print(f"ヒストグラム含有時の特徴量名（単位付き）: {feature_names}")

    # 特徴量名の取得（単位なし、インスタンス設定反映）
    feature_names_no_units = extractor.get_base_feature_names_instance()
    print(f"ヒストグラム含有時の特徴量名（単位なし）: {feature_names_no_units}")

    # 特徴量の単位辞書の取得（インスタンス設定反映）
    feature_units = extractor.get_feature_units_instance()
    print(f"ヒストグラム含有時の特徴量の単位辞書: {feature_units}")

    # ヒストグラムビンの単位確認
    histogram_features = [
        name for name in feature_names_no_units if name.startswith("lbp_bin_")
    ]
    print(f"ヒストグラムビン特徴量: {histogram_features}")

    for bin_feature in histogram_features:
        unit = extractor.get_feature_unit_instance(bin_feature)
        assert (
            unit == "ratio"
        ), f"ヒストグラムビン '{bin_feature}' の単位が 'ratio' ではありません: {unit}"

    # 実際の抽出結果と特徴量名（単位なし）が一致するかチェック
    extracted_names = set(features.keys())
    expected_names = set(feature_names_no_units)

    assert (
        extracted_names == expected_names
    ), "ヒストグラム含有時の抽出された特徴量名と期待される特徴量名が一致しません"

    # 単位付きの特徴量名が正しい形式であることを確認
    for name_with_unit, name_without_unit in zip(feature_names, feature_names_no_units):
        expected_unit = feature_units.get(name_without_unit, "unknown")
        expected_name_with_unit = f"{name_without_unit}[{expected_unit}]"
        assert name_with_unit == expected_name_with_unit, (
            f"ヒストグラム含有時の特徴量名形式が不正: 期待値 '{expected_name_with_unit}', "
            f"実際 '{name_with_unit}'"
        )

    print("ヒストグラム含有時の単位管理機能の確認が完了しました")


def test_lbp_config_merging():
    """設定マージ機能のテスト."""
    print("\n=== 設定マージ機能テスト ===")

    # テスト画像
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # 1. 空の設定でインスタンス化（デフォルト設定のみ）
    extractor1 = LBPTextureExtractor()
    print(f"空設定時のconfig: {extractor1.config}")

    # 2. 部分的な設定でインスタンス化（デフォルト設定とマージ）
    partial_config = {"P": 16, "R": 2}
    extractor2 = LBPTextureExtractor(config=partial_config)
    print(f"部分設定時のconfig: {extractor2.config}")

    # 3. 完全な設定でインスタンス化
    full_config = {
        "P": 12,
        "R": 1.5,
        "method": "ror",
        "resize_shape": [64, 64],
        "include_histogram": True,
    }
    extractor3 = LBPTextureExtractor(config=full_config)
    print(f"完全設定時のconfig: {extractor3.config}")

    # 各設定での特徴量抽出
    features1 = extractor1.extract(test_image)
    features2 = extractor2.extract(test_image)
    features3 = extractor3.extract(test_image)

    print(f"デフォルト設定での特徴量数: {len(features1)}")
    print(f"部分設定での特徴量数: {len(features2)}")
    print(f"完全設定での特徴量数: {len(features3)}")

    # ヒストグラム含有設定での特徴量数の違いを確認
    assert len(features3) > len(
        features1
    ), "ヒストグラム含有設定で特徴量数が増加していません"


if __name__ == "__main__":
    test_lbp_texture_basic()
    test_lbp_different_parameters()
    test_lbp_with_histogram()
    test_lbp_resize_options()
    test_lbp_grayscale_input()
    test_lbp_edge_cases()
    test_lbp_error_handling()
    test_lbp_feature_names_and_config()
    test_lbp_units_with_histogram()
    test_lbp_config_merging()
    print("\n=== 全てのテストが完了しました ===")


from tests.extractors.conftest import DummyImages


class TestLBPBehavior:
    """LBP 特徴量の振る舞いテスト."""

    _CONFIG = {
        "P": 8,
        "R": 1,
        "method": "uniform",
        "resize_shape": None,
        "include_histogram": True,
        "preserve_aspect_ratio": True,
        "aspect_ratio_mode": "width",
    }
    _CONFIG_NO_HIST = {
        "P": 8,
        "R": 1,
        "method": "uniform",
        "resize_shape": None,
        "include_histogram": False,
        "preserve_aspect_ratio": True,
        "aspect_ratio_mode": "width",
    }

    def setup_method(self):
        """テストメソッドごとに extractor を初期化."""
        self.ext = LBPTextureExtractor(config=self._CONFIG)

    # --- 全黒画像 ---

    def test_black_energy_is_one(self):
        """全黒画像の energy は 1.0 (全ピクセルが同一パターン)."""
        f = self.ext.extract(DummyImages.black())
        assert f["lbp_energy"] == 1.0

    def test_black_entropy_is_zero(self):
        """全黒画像の entropy は 0.0."""
        f = self.ext.extract(DummyImages.black())
        assert f["lbp_entropy"] == 0.0

    def test_black_std_is_zero(self):
        """全黒画像の std は 0.0."""
        f = self.ext.extract(DummyImages.black())
        assert f["lbp_std"] == 0.0

    def test_black_bin8_is_one(self):
        """全黒画像は bin_8 (周囲すべて明るい) が 1.0."""
        f = self.ext.extract(DummyImages.black())
        assert f["lbp_bin_8"] == 1.0

    # --- チェッカーボード ---

    def test_checker_bin0_and_bin8_equal(self):
        """チェッカーボードは bin_0 と bin_8 がそれぞれ 0.5."""
        f = self.ext.extract(DummyImages.checker())
        assert abs(f["lbp_bin_0"] - 0.5) < 0.01
        assert abs(f["lbp_bin_8"] - 0.5) < 0.01

    def test_checker_std_is_maximum(self):
        """チェッカーボードの std は大きい (2値分布)."""
        f = self.ext.extract(DummyImages.checker())
        assert f["lbp_std"] > 3.0

    def test_checker_energy_is_half(self):
        """チェッカーボードの energy は ~0.5 (2パターンが均等)."""
        f = self.ext.extract(DummyImages.checker())
        assert abs(f["lbp_energy"] - 0.5) < 0.05

    # --- 水平 = 垂直ストライプ (uniform LBP は回転不変) ---

    def test_h_stripe_equals_v_stripe(self):
        """uniform LBP では水平と垂直ストライプが同じ特徴量."""
        fh = self.ext.extract(DummyImages.h_stripe())
        fv = self.ext.extract(DummyImages.v_stripe())
        for key in fh:
            assert abs(fh[key] - fv[key]) < 1e-6, f"{key}: {fh[key]} vs {fv[key]}"

    # --- ランダム ---

    def test_random_entropy_high(self):
        """ランダム画像の entropy は > 0.7 (多様なパターン)."""
        f = self.ext.extract(DummyImages.random())
        assert f["lbp_entropy"] > 0.7

    def test_random_energy_low(self):
        """ランダム画像の energy は < 0.3 (パターンが分散)."""
        f = self.ext.extract(DummyImages.random())
        assert f["lbp_energy"] < 0.3

    def test_random_non_uniform_bin_high(self):
        """ランダム画像の non-uniform ビン (bin_9) は > 0.1."""
        f = self.ext.extract(DummyImages.random())
        assert f["lbp_bin_9"] > 0.1

    def test_random_std_positive(self):
        """ランダム画像の std は > 1.0."""
        f = self.ext.extract(DummyImages.random())
        assert f["lbp_std"] > 1.0

    # --- ランダム vs 全黒 ---

    def test_random_entropy_higher_than_black(self):
        """ランダムの entropy は全黒より高い."""
        fr = self.ext.extract(DummyImages.random())
        fb = self.ext.extract(DummyImages.black())
        assert fr["lbp_entropy"] > fb["lbp_entropy"]

    def test_random_energy_lower_than_black(self):
        """ランダムの energy は全黒より低い."""
        fr = self.ext.extract(DummyImages.random())
        fb = self.ext.extract(DummyImages.black())
        assert fr["lbp_energy"] < fb["lbp_energy"]

    # --- ヒストグラム正規化 ---

    def test_histogram_sums_to_one(self):
        """ヒストグラムの合計が 1.0."""
        f = self.ext.extract(DummyImages.random())
        total = sum(f[f"lbp_bin_{i}"] for i in range(10))
        assert abs(total - 1.0) < 0.001

    def test_all_bins_non_negative(self):
        """ヒストグラムの全ビンが非負."""
        f = self.ext.extract(DummyImages.random())
        for i in range(10):
            assert f[f"lbp_bin_{i}"] >= 0.0

    # --- 均一画像 ---

    def test_uniform_low_entropy(self):
        """均一画像の entropy は < 0.2."""
        f = self.ext.extract(DummyImages.uniform())
        assert f["lbp_entropy"] < 0.2

    def test_uniform_high_energy(self):
        """均一画像の energy は > 0.8."""
        f = self.ext.extract(DummyImages.uniform())
        assert f["lbp_energy"] > 0.8

    def test_uniform_bin8_dominant(self):
        """均一画像は bin_8 が > 0.9 (ほぼ全ピクセルが同一パターン)."""
        f = self.ext.extract(DummyImages.uniform())
        assert f["lbp_bin_8"] > 0.9

    # --- include_histogram フラグ ---

    def test_histogram_excluded_when_disabled(self):
        """include_histogram=False では bin 特徴量が含まれない."""
        ext_no_hist = LBPTextureExtractor(config=self._CONFIG_NO_HIST)
        f = ext_no_hist.extract(DummyImages.random())
        assert "lbp_bin_0" not in f
        assert "lbp_mean" in f

    def test_histogram_included_when_enabled(self):
        """include_histogram=True では bin 特徴量が含まれる."""
        f = self.ext.extract(DummyImages.random())
        assert "lbp_bin_0" in f
        assert "lbp_bin_9" in f
