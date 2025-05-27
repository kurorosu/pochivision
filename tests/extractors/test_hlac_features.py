"""HLACテクスチャ特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from feature_extractors import HLACTextureExtractor, get_feature_extractor


def test_hlac_texture_basic():
    """HLACテクスチャ特徴量抽出の基本テスト."""
    print("=== HLACテクスチャ特徴量抽出基本テスト ===")

    # テスト画像の作成（チェッカーボードパターン）
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i // 8 + j // 8) % 2 == 0:
                test_image[i : i + 8, j : j + 8, :] = 255

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = HLACTextureExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.6f}")

    # 基本的な特徴量が含まれているかチェック（45次元）
    assert len(features) == 45, f"特徴量数が45ではありません: {len(features)}"

    # 特徴量名の形式チェック（ゼロパディング対応）
    for i in range(45):
        feature_name = f"hlac_feature_{i:02d}"
        assert feature_name in features, f"特徴量 {feature_name} が見つかりません"

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("hlac", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for name, value in features2.items():
        print(f"  {name}: {value:.6f}")

    # 結果が一致することを確認
    for key in features:
        assert (
            abs(features[key] - features2[key]) < 1e-10
        ), f"特徴量 {key} の値が一致しません"


def test_hlac_rotation_invariant():
    """回転不変HLACテスト."""
    print("\n=== 回転不変HLACテスト ===")

    # テスト画像（ランダムテクスチャ）
    np.random.seed(42)
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # 1. 標準HLAC（45次元）
    print("\n--- 標準HLAC (45次元) ---")
    extractor_standard = HLACTextureExtractor()
    features_standard = extractor_standard.extract(test_image)
    print(f"特徴量数: {len(features_standard)}")
    for i, (name, value) in enumerate(features_standard.items()):
        if i < 5:  # 最初の5つのみ表示
            print(f"  {name}: {value:.6f}")

    # 2. 回転不変HLAC（11次元）
    print("\n--- 回転不変HLAC (11次元) ---")
    config_ri = {"rotate_invariant": True}
    extractor_ri = get_feature_extractor("hlac", config_ri)
    features_ri = extractor_ri.extract(test_image)
    print(f"特徴量数: {len(features_ri)}")
    for name, value in features_ri.items():
        print(f"  {name}: {value:.6f}")

    # 特徴量数の確認
    assert (
        len(features_standard) == 45
    ), f"標準HLACの特徴量数が45ではありません: {len(features_standard)}"
    assert (
        len(features_ri) == 11
    ), f"回転不変HLACの特徴量数が11ではありません: {len(features_ri)}"


def test_hlac_different_orders():
    """異なる次数でのHLACテスト."""
    print("\n=== 異なる次数でのHLACテスト ===")

    # テスト画像
    test_image = np.full((64, 64, 3), 128, dtype=np.uint8)
    # 一部にパターンを追加
    test_image[20:40, 20:40, :] = 200

    # 1. 1次HLAC
    print("\n--- 1次HLAC ---")
    config_order1 = {"order": 1}
    extractor_order1 = get_feature_extractor("hlac", config_order1)
    features_order1 = extractor_order1.extract(test_image)
    print(f"特徴量数: {len(features_order1)}")
    for name, value in features_order1.items():
        print(f"  {name}: {value:.6f}")

    # 2. 2次HLAC（デフォルト）
    print("\n--- 2次HLAC ---")
    config_order2 = {"order": 2}
    extractor_order2 = get_feature_extractor("hlac", config_order2)
    features_order2 = extractor_order2.extract(test_image)
    print(f"特徴量数: {len(features_order2)}")
    for i, (name, value) in enumerate(features_order2.items()):
        if i < 10:  # 最初の10個のみ表示
            print(f"  {name}: {value:.6f}")

    # 特徴量数の確認
    assert (
        len(features_order1) == 9
    ), f"1次HLACの特徴量数が9ではありません: {len(features_order1)}"
    assert (
        len(features_order2) == 45
    ), f"2次HLACの特徴量数が45ではありません: {len(features_order2)}"


def test_hlac_multiscale():
    """マルチスケールHLACテスト."""
    print("\n=== マルチスケールHLACテスト ===")

    # テスト画像
    test_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    # 1. デフォルトスケール [1.0, 0.75, 0.5]
    print("\n--- デフォルトスケール [1.0, 0.75, 0.5] ---")
    extractor_default = HLACTextureExtractor()
    features_default = extractor_default.extract(test_image)
    for i, (name, value) in enumerate(features_default.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 2. 単一スケール [1.0]
    print("\n--- 単一スケール [1.0] ---")
    config_single = {"scales": [1.0]}
    extractor_single = get_feature_extractor("hlac", config_single)
    features_single = extractor_single.extract(test_image)
    for i, (name, value) in enumerate(features_single.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 3. より多くのスケール
    print("\n--- 多スケール [1.0, 0.8, 0.6, 0.4] ---")
    config_multi = {"scales": [1.0, 0.8, 0.6, 0.4]}
    extractor_multi = get_feature_extractor("hlac", config_multi)
    features_multi = extractor_multi.extract(test_image)
    for i, (name, value) in enumerate(features_multi.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")


def test_hlac_normalization():
    """正規化オプションのテスト."""
    print("\n=== 正規化オプションテスト ===")

    # テスト画像
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # 1. 正規化あり（デフォルト）
    print("\n--- 正規化あり ---")
    extractor_norm = HLACTextureExtractor()
    features_norm = extractor_norm.extract(test_image)

    # 正規化されているかチェック（合計が1に近い）
    total_norm = sum(features_norm.values())
    print(f"特徴量の合計: {total_norm:.6f}")
    for i, (name, value) in enumerate(features_norm.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 2. 正規化なし
    print("\n--- 正規化なし ---")
    config_no_norm = {"normalize": False}
    extractor_no_norm = get_feature_extractor("hlac", config_no_norm)
    features_no_norm = extractor_no_norm.extract(test_image)

    total_no_norm = sum(features_no_norm.values())
    print(f"特徴量の合計: {total_no_norm:.6f}")
    for i, (name, value) in enumerate(features_no_norm.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 正規化の効果を確認
    assert (
        abs(total_norm - 1.0) < 0.01
    ), f"正規化後の合計が1に近くありません: {total_norm}"
    assert total_no_norm > total_norm, "正規化なしの方が値が大きいはずです"


def test_hlac_resize_options():
    """リサイズオプションのテスト."""
    print("\n=== リサイズオプションテスト ===")

    # 異なるサイズのテスト画像
    test_image_small = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    test_image_large = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # 1. リサイズなし（デフォルト）
    print("\n--- リサイズなし ---")
    extractor_default = HLACTextureExtractor()

    features_small = extractor_default.extract(test_image_small)
    features_large = extractor_default.extract(test_image_large)

    print("小画像 (32x32):")
    for i, (name, value) in enumerate(features_small.items()):
        if i < 3:
            print(f"  {name}: {value:.6f}")

    print("大画像 (256x256):")
    for i, (name, value) in enumerate(features_large.items()):
        if i < 3:
            print(f"  {name}: {value:.6f}")

    # 2. カスタムリサイズ（64x64）
    print("\n--- カスタムリサイズ (64x64) ---")
    config_custom_resize = {"resize_shape": [64, 64]}
    extractor_custom = get_feature_extractor("hlac", config_custom_resize)
    features_custom = extractor_custom.extract(test_image_large)

    print("大画像 (256x256) -> 64x64リサイズ後:")
    for i, (name, value) in enumerate(features_custom.items()):
        if i < 3:
            print(f"  {name}: {value:.6f}")


def test_hlac_grayscale_input():
    """グレースケール画像入力のテスト."""
    print("\n=== グレースケール画像入力テスト ===")

    # グレースケールテスト画像
    gray_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    extractor = HLACTextureExtractor()
    features = extractor.extract(gray_image)

    print("グレースケール画像の特徴量:")
    for i, (name, value) in enumerate(features.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # カラー画像と比較
    color_image = np.stack([gray_image, gray_image, gray_image], axis=2)
    features_color = extractor.extract(color_image)

    print("同等のカラー画像の特徴量:")
    for i, (name, value) in enumerate(features_color.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 結果が一致することを確認
    for key in features:
        assert (
            abs(features[key] - features_color[key]) < 1e-10
        ), f"グレースケールとカラーで特徴量 {key} の値が一致しません"


def test_hlac_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = HLACTextureExtractor()

    # 1. 単色画像
    print("\n--- 単色画像 ---")
    solid_image = np.full((64, 64, 3), 128, dtype=np.uint8)
    features_solid = extractor.extract(solid_image)
    print("単色画像の特徴量:")
    for i, (name, value) in enumerate(features_solid.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 2. 白黒画像
    print("\n--- 白黒画像 ---")
    bw_image = np.zeros((64, 64, 3), dtype=np.uint8)
    bw_image[:32, :, :] = 255  # 上半分を白に
    features_bw = extractor.extract(bw_image)
    print("白黒画像の特徴量:")
    for i, (name, value) in enumerate(features_bw.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")

    # 3. 最小サイズ画像
    print("\n--- 最小サイズ画像 ---")
    tiny_image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    features_tiny = extractor.extract(tiny_image)
    print("最小サイズ画像の特徴量:")
    for i, (name, value) in enumerate(features_tiny.items()):
        if i < 5:
            print(f"  {name}: {value:.6f}")


def test_hlac_error_handling():
    """エラーハンドリングのテスト."""
    print("\n=== エラーハンドリングテスト ===")

    extractor = HLACTextureExtractor()

    # 1. 空の画像
    print("\n--- 空の画像 ---")
    try:
        empty_image = np.array([])
        features = extractor.extract(empty_image)
        print("空の画像でもエラーが発生せず、デフォルト値が返されました")
        assert len(features) == 45, "デフォルト特徴量数が正しくありません"
    except ValueError as e:
        print(f"期待通りValueErrorが発生: {e}")

    # 2. None画像
    print("\n--- None画像 ---")
    try:
        features = extractor.extract(None)
        print("None画像でもエラーが発生せず、デフォルト値が返されました")
        assert len(features) == 45, "デフォルト特徴量数が正しくありません"
    except ValueError as e:
        print(f"期待通りValueErrorが発生: {e}")

    # 3. 不正な形状の画像
    print("\n--- 不正な形状の画像 ---")
    try:
        invalid_image = np.random.randint(0, 256, (64, 64, 64, 3), dtype=np.uint8)
        features = extractor.extract(invalid_image)
        print("不正な形状でもエラーが発生せず、デフォルト値が返されました")
        assert len(features) == 45, "デフォルト特徴量数が正しくありません"
    except ValueError as e:
        print(f"期待通りValueErrorが発生: {e}")


def test_hlac_feature_names_and_config():
    """特徴量名とコンフィグのテスト."""
    print("\n=== 特徴量名とコンフィグテスト ===")

    # デフォルト設定の確認
    default_config = HLACTextureExtractor.get_default_config()
    print("デフォルト設定:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")

    # 特徴量名の確認
    feature_names = HLACTextureExtractor.get_feature_names()
    print(f"\n特徴量名（数: {len(feature_names)}）:")
    for i, name in enumerate(feature_names):
        if i < 10:  # 最初の10個のみ表示
            print(f"  {name}")

    # 実際の抽出結果と特徴量名が一致するかチェック
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    extractor = HLACTextureExtractor()
    features = extractor.extract(test_image)

    assert len(features) == len(feature_names), "特徴量数と特徴量名の数が一致しません"
    for name in feature_names:
        assert name in features, f"特徴量名 {name} が実際の特徴量に含まれていません"


def test_hlac_config_merging():
    """設定のマージテスト."""
    print("\n=== 設定マージテスト ===")

    # カスタム設定
    custom_config = {
        "order": 1,
        "rotate_invariant": True,
        "normalize": False,
    }

    extractor = HLACTextureExtractor(config=custom_config)

    print("マージされた設定:")
    for key, value in extractor.config.items():
        print(f"  {key}: {value}")

    # カスタム設定が反映されているかチェック
    assert extractor.config["order"] == 1, "order設定が反映されていません"
    assert (
        extractor.config["rotate_invariant"] is True
    ), "rotate_invariant設定が反映されていません"
    assert extractor.config["normalize"] is False, "normalize設定が反映されていません"

    # デフォルト値が保持されているかチェック
    assert "scales" in extractor.config, "デフォルトのscales設定が失われています"


if __name__ == "__main__":
    test_hlac_texture_basic()
    test_hlac_rotation_invariant()
    test_hlac_different_orders()
    test_hlac_multiscale()
    test_hlac_normalization()
    test_hlac_resize_options()
    test_hlac_grayscale_input()
    test_hlac_edge_cases()
    test_hlac_error_handling()
    test_hlac_feature_names_and_config()
    test_hlac_config_merging()
    print("\n=== すべてのテストが完了しました ===")
