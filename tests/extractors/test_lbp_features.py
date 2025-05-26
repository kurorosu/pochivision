"""LBPテクスチャ特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from feature_extractors import LBPTextureExtractor, get_feature_extractor


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
        "lbp_uniformity",
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
    print("\n=== エラーハンドリングテスト ===")

    extractor = LBPTextureExtractor()

    # 1. 空の画像
    print("\n--- 空の画像 ---")
    try:
        empty_image = np.array([])
        features = extractor.extract(empty_image)
        print("空画像でもエラーなく処理されました")
        print("デフォルト特徴量:")
        for name, value in features.items():
            print(f"  {name}: {value}")
    except Exception as e:
        print(f"空画像でエラー: {e}")

    # 2. None画像
    print("\n--- None画像 ---")
    try:
        features = extractor.extract(None)
        print("None画像でもエラーなく処理されました")
    except Exception as e:
        print(f"None画像でエラー: {e}")

    # 3. 不正な次元の画像
    print("\n--- 不正な次元の画像 ---")
    try:
        invalid_image = np.random.randint(0, 256, (64, 64, 3, 2), dtype=np.uint8)
        features = extractor.extract(invalid_image)
        print("4次元画像でもエラーなく処理されました")
        for name, value in features.items():
            print(f"  {name}: {value}")
    except Exception as e:
        print(f"4次元画像でエラー: {e}")


def test_lbp_feature_names_and_config():
    """特徴量名とデフォルト設定のテスト."""
    print("\n=== 特徴量名とデフォルト設定テスト ===")

    # 特徴量名の取得
    feature_names = LBPTextureExtractor.get_feature_names()
    print(f"特徴量名リスト: {feature_names}")
    print(f"特徴量数: {len(feature_names)}")

    # デフォルト設定の取得
    default_config = LBPTextureExtractor.get_default_config()
    print(f"デフォルト設定: {default_config}")

    # 実際の抽出結果と特徴量名が一致するかチェック
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    extractor = LBPTextureExtractor()
    features = extractor.extract(test_image)

    extracted_names = set(features.keys())
    expected_names = set(feature_names)

    print(f"抽出された特徴量名: {sorted(extracted_names)}")
    print(f"期待される特徴量名: {sorted(expected_names)}")

    assert (
        extracted_names == expected_names
    ), "抽出された特徴量名と期待される特徴量名が一致しません"
    print("特徴量名の一致を確認しました")


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
    test_lbp_config_merging()
    print("\n=== 全てのテストが完了しました ===")
