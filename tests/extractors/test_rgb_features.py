"""RGB統計特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from feature_extractors import RGBStatisticsExtractor, get_feature_extractor


def test_rgb_statistics_basic():
    """RGB統計特徴量抽出の基本テスト."""
    print("=== RGB統計特徴量抽出基本テスト ===")

    # テスト画像の作成（赤、緑、青のグラデーション画像）
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 赤チャンネルは横方向グラデーション
    for i in range(100):
        test_image[:, i, 0] = int(i * 2.55)  # 0-255
    # 緑チャンネルは縦方向グラデーション
    for i in range(100):
        test_image[i, :, 1] = int(i * 2.55)  # 0-255
    # 青チャンネルは対角線グラデーション
    for i in range(100):
        for j in range(100):
            test_image[i, j, 2] = int((i + j) * 1.275)  # 0-255

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = RGBStatisticsExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    channels = ["red", "green", "blue"]
    stats = ["mean", "median", "variance", "std_dev", "cv"]

    for channel in channels:
        print(f"\n{channel.upper()}チャンネル:")
        for stat in stats:
            key = f"{channel}_{stat}"
            value = features.get(key, "N/A")
            if isinstance(value, (int, float)) and value != float("inf"):
                print(f"  {stat}: {value:.3f}")
            else:
                print(f"  {stat}: {value}")

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("rgb", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for channel in channels:
        print(f"\n{channel.upper()}チャンネル:")
        for stat in stats:
            key = f"{channel}_{stat}"
            value = features2.get(key, "N/A")
            if isinstance(value, (int, float)) and value != float("inf"):
                print(f"  {stat}: {value:.3f}")
            else:
                print(f"  {stat}: {value}")


def test_black_pixel_exclusion():
    """黒ピクセル除外機能のテスト."""
    print("\n=== 黒ピクセル除外機能テスト ===")

    # 1. 黒ピクセル除外有効の場合
    print("--- 黒ピクセル除外有効 ---")
    extractor_exclude = RGBStatisticsExtractor(config={"exclude_black_pixels": True})

    # 黒い背景に色付きの矩形を配置
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 中央に色付きの領域を作成
    test_image[25:75, 25:75, 0] = 200  # 赤
    test_image[25:75, 25:75, 1] = 150  # 緑
    test_image[25:75, 25:75, 2] = 100  # 青

    features_exclude = extractor_exclude.extract(test_image)
    print("黒ピクセル除外有効時の特徴量:")
    for channel in ["red", "green", "blue"]:
        mean_key = f"{channel}_mean"
        print(f"  {channel}_mean: {features_exclude[mean_key]:.1f}")

    # 2. 黒ピクセル除外無効の場合
    print("\n--- 黒ピクセル除外無効 ---")
    extractor_include = RGBStatisticsExtractor(config={"exclude_black_pixels": False})

    features_include = extractor_include.extract(test_image)
    print("黒ピクセル除外無効時の特徴量:")
    for channel in ["red", "green", "blue"]:
        mean_key = f"{channel}_mean"
        print(f"  {channel}_mean: {features_include[mean_key]:.1f}")

    # 3. 全て黒の画像でのテスト
    print("\n--- 全て黒の画像 ---")
    black_image = np.zeros((50, 50, 3), dtype=np.uint8)

    features_black_exclude = extractor_exclude.extract(black_image)
    print("黒画像（除外有効）:")
    for channel in ["red", "green", "blue"]:
        mean_key = f"{channel}_mean"
        print(f"  {channel}_mean: {features_black_exclude[mean_key]}")

    features_black_include = extractor_include.extract(black_image)
    print("黒画像（除外無効）:")
    for channel in ["red", "green", "blue"]:
        mean_key = f"{channel}_mean"
        print(f"  {channel}_mean: {features_black_include[mean_key]:.1f}")


def test_pure_color_images():
    """純色画像でのテスト."""
    print("\n=== 純色画像テスト ===")

    extractor = RGBStatisticsExtractor()

    # 純赤画像
    red_image = np.zeros((50, 50, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # 赤チャンネルのみ

    features_red = extractor.extract(red_image)
    print("純赤画像の特徴量:")
    print(f"  red_mean: {features_red['red_mean']:.1f}")
    print(f"  green_mean: {features_red['green_mean']:.1f}")
    print(f"  blue_mean: {features_red['blue_mean']:.1f}")

    # 純緑画像
    green_image = np.zeros((50, 50, 3), dtype=np.uint8)
    green_image[:, :, 1] = 255  # 緑チャンネルのみ

    features_green = extractor.extract(green_image)
    print("\n純緑画像の特徴量:")
    print(f"  red_mean: {features_green['red_mean']:.1f}")
    print(f"  green_mean: {features_green['green_mean']:.1f}")
    print(f"  blue_mean: {features_green['blue_mean']:.1f}")

    # 純青画像
    blue_image = np.zeros((50, 50, 3), dtype=np.uint8)
    blue_image[:, :, 2] = 255  # 青チャンネルのみ

    features_blue = extractor.extract(blue_image)
    print("\n純青画像の特徴量:")
    print(f"  red_mean: {features_blue['red_mean']:.1f}")
    print(f"  green_mean: {features_blue['green_mean']:.1f}")
    print(f"  blue_mean: {features_blue['blue_mean']:.1f}")

    # 白画像
    white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
    features_white = extractor.extract(white_image)
    print("\n白画像の特徴量:")
    print(f"  red_mean: {features_white['red_mean']:.1f}")
    print(f"  green_mean: {features_white['green_mean']:.1f}")
    print(f"  blue_mean: {features_white['blue_mean']:.1f}")


def test_statistical_properties():
    """統計的特性のテスト."""
    print("\n=== 統計的特性テスト ===")

    extractor = RGBStatisticsExtractor()

    # 単色画像（分散=0のケース）
    uniform_image = np.full((50, 50, 3), [128, 64, 192], dtype=np.uint8)
    features_uniform = extractor.extract(uniform_image)

    print("単色画像の統計特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"\n{channel.upper()}チャンネル:")
        print(f"  mean: {features_uniform[f'{channel}_mean']:.1f}")
        print(f"  variance: {features_uniform[f'{channel}_variance']:.3f}")
        print(f"  std_dev: {features_uniform[f'{channel}_std_dev']:.3f}")
        print(f"  cv: {features_uniform[f'{channel}_cv']}")

    # ランダム画像
    np.random.seed(42)  # 再現性のため
    random_image = np.random.randint(1, 256, (50, 50, 3), dtype=np.uint8)
    features_random = extractor.extract(random_image)

    print("\nランダム画像の統計特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"\n{channel.upper()}チャンネル:")
        print(f"  mean: {features_random[f'{channel}_mean']:.1f}")
        print(f"  median: {features_random[f'{channel}_median']:.1f}")
        print(f"  variance: {features_random[f'{channel}_variance']:.1f}")
        print(f"  std_dev: {features_random[f'{channel}_std_dev']:.1f}")
        print(f"  cv: {features_random[f'{channel}_cv']:.3f}")


def test_exclude_black_pixels_option():
    """exclude_black_pixelsオプションのテスト."""
    print("\n=== exclude_black_pixelsオプションテスト ===")

    # テスト画像の作成（一部に黒ピクセルを含む）
    test_image = np.zeros((50, 50, 3), dtype=np.uint8)
    # 中央部分に非ゼロ値を設定
    test_image[10:40, 10:40, 0] = 100  # 30x30の領域に赤=100
    test_image[10:40, 10:40, 1] = 150  # 緑=150
    test_image[10:40, 10:40, 2] = 200  # 青=200
    # 一部に異なる色を設定
    test_image[20:30, 20:30, 0] = 50  # 10x10の領域に赤=50
    test_image[20:30, 20:30, 1] = 100  # 緑=100
    test_image[20:30, 20:30, 2] = 255  # 青=255

    print(f"テスト画像: 全ピクセル数={50*50}, 非黒ピクセル数={30*30}")

    # 1. exclude_black_pixels=True（デフォルト）
    extractor_exclude = RGBStatisticsExtractor(config={"exclude_black_pixels": True})
    features_exclude = extractor_exclude.extract(test_image)
    print("\nexclude_black_pixels=True（黒ピクセル除外）:")
    for channel in ["red", "green", "blue"]:
        for stat in ["mean", "median", "variance", "std_dev", "cv"]:
            key = f"{channel}_{stat}"
            value = features_exclude[key]
            if isinstance(value, float) and value != float("inf"):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # 2. exclude_black_pixels=False
    extractor_include = RGBStatisticsExtractor(config={"exclude_black_pixels": False})
    features_include = extractor_include.extract(test_image)
    print("\nexclude_black_pixels=False（黒ピクセル含む）:")
    for channel in ["red", "green", "blue"]:
        for stat in ["mean", "median", "variance", "std_dev", "cv"]:
            key = f"{channel}_{stat}"
            value = features_include[key]
            if isinstance(value, float) and value != float("inf"):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # 3. 結果の比較
    print("\n--- 結果比較 ---")
    for channel in ["red", "green", "blue"]:
        print(f"{channel.upper()}チャンネル:")
        print(
            f"  平均値: 除外={features_exclude[f'{channel}_mean']:.3f}, "
            f"含む={features_include[f'{channel}_mean']:.3f}"
        )
        print(
            f"  分散: 除外={features_exclude[f'{channel}_variance']:.3f}, "
            f"含む={features_include[f'{channel}_variance']:.3f}"
        )


def test_grayscale_support():
    """グレースケール画像対応のテスト."""
    print("\n=== グレースケール画像対応テスト ===")

    extractor = RGBStatisticsExtractor()

    # 2次元グレースケール画像
    gray_2d = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
    features_2d = extractor.extract(gray_2d)
    print("2次元グレースケール画像の特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"  {channel}_mean: {features_2d[f'{channel}_mean']:.1f}")

    # グレースケールなので全チャンネルが同じ値になるはず
    assert abs(features_2d["red_mean"] - features_2d["green_mean"]) < 0.1
    assert abs(features_2d["green_mean"] - features_2d["blue_mean"]) < 0.1
    print("  ✓ 全チャンネルが同じ値になっています")


def test_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = RGBStatisticsExtractor()

    # 単色画像（分散=0のケース）
    uniform_image = np.full((50, 50, 3), [128, 64, 192], dtype=np.uint8)
    features = extractor.extract(uniform_image)
    print("単色画像の特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"  {channel}_variance: {features[f'{channel}_variance']:.3f}")
        print(f"  {channel}_std_dev: {features[f'{channel}_std_dev']:.3f}")
        print(f"  {channel}_cv: {features[f'{channel}_cv']}")

    # 黒画像（平均=0のケース、すべてのピクセルが除外される）
    black_image = np.zeros((50, 50, 3), dtype=np.uint8)
    features = extractor.extract(black_image)
    print("\n黒画像の特徴量（全て除外）:")
    for channel in ["red", "green", "blue"]:
        for stat in ["mean", "median", "variance", "std_dev", "cv"]:
            key = f"{channel}_{stat}"
            print(f"  {key}: {features[key]}")


def test_default_config_merging():
    """デフォルト設定のマージ機能のテスト."""
    print("\n=== デフォルト設定マージテスト ===")

    # テスト画像
    test_image = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)

    # 1. 空の設定でインスタンス化（デフォルト設定のみ）
    extractor1 = RGBStatisticsExtractor()
    print(f"空設定時のconfig: {extractor1.config}")

    # 2. 部分的な設定でインスタンス化（デフォルト設定とマージ）
    partial_config = {"exclude_black_pixels": False}
    extractor2 = RGBStatisticsExtractor(config=partial_config)
    print(f"部分設定時のconfig: {extractor2.config}")

    # 3. 各設定での実際の動作確認
    print("\n--- 実際の処理結果 ---")

    features1 = extractor1.extract(test_image)
    print(f"デフォルト設定(exclude_black=True): red_mean={features1['red_mean']:.1f}")

    features2 = extractor2.extract(test_image)
    print(f"カスタム設定(exclude_black=False): red_mean={features2['red_mean']:.1f}")


def test_feature_names_and_units():
    """特徴量名と単位のテスト."""
    print("\n=== 特徴量名・単位テスト ===")

    # 基本特徴量名の確認
    base_names = RGBStatisticsExtractor.get_base_feature_names()
    expected_base_names = [
        "red_mean",
        "red_median",
        "red_variance",
        "red_std_dev",
        "red_cv",
        "green_mean",
        "green_median",
        "green_variance",
        "green_std_dev",
        "green_cv",
        "blue_mean",
        "blue_median",
        "blue_variance",
        "blue_std_dev",
        "blue_cv",
    ]
    print(f"基本特徴量名: {base_names}")
    assert (
        base_names == expected_base_names
    ), f"Expected {expected_base_names}, got {base_names}"

    # 単位付き特徴量名の確認
    unit_names = RGBStatisticsExtractor.get_feature_names()
    expected_unit_names = [
        "red_mean[0-255]",
        "red_median[0-255]",
        "red_variance[0-255_squared]",
        "red_std_dev[0-255]",
        "red_cv[ratio]",
        "green_mean[0-255]",
        "green_median[0-255]",
        "green_variance[0-255_squared]",
        "green_std_dev[0-255]",
        "green_cv[ratio]",
        "blue_mean[0-255]",
        "blue_median[0-255]",
        "blue_variance[0-255_squared]",
        "blue_std_dev[0-255]",
        "blue_cv[ratio]",
    ]
    print(f"単位付き特徴量名: {unit_names}")
    assert (
        unit_names == expected_unit_names
    ), f"Expected {expected_unit_names}, got {unit_names}"

    # 単位辞書の確認
    units = RGBStatisticsExtractor.get_feature_units()
    print(f"特徴量単位辞書: {units}")

    # 各特徴量の単位が正しいことを確認
    for name in base_names:
        channel, stat = name.split("_", 1)
        expected_unit = RGBStatisticsExtractor._FEATURE_UNITS[stat]
        assert (
            units[name] == expected_unit
        ), f"Unit mismatch for {name}: expected {expected_unit}, got {units[name]}"

    # 抽出結果と特徴量名の整合性確認
    extractor = RGBStatisticsExtractor()
    test_image = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)
    features = extractor.extract(test_image)

    # 抽出された特徴量のキーが基本特徴量名と一致することを確認
    feature_keys = list(features.keys())
    print(f"抽出された特徴量のキー: {feature_keys}")
    assert set(feature_keys) == set(
        base_names
    ), f"Feature keys {feature_keys} don't match base names {base_names}"

    print("特徴量名・単位テスト: 成功")


def test_unit_for_feature_method():
    """_get_unit_for_feature()メソッドのテスト."""
    print("\n=== 単位取得メソッドテスト ===")

    # 正常なケース
    test_cases = [
        ("red_mean", "0-255"),
        ("green_median", "0-255"),
        ("blue_variance", "0-255_squared"),
        ("red_std_dev", "0-255"),
        ("green_cv", "ratio"),
    ]

    for feature_name, expected_unit in test_cases:
        unit = RGBStatisticsExtractor._get_unit_for_feature(feature_name)
        print(f"  {feature_name} -> {unit}")
        assert unit == expected_unit, f"Expected {expected_unit}, got {unit}"

    # 異常なケース
    invalid_cases = ["invalid", "red", "mean", ""]
    for invalid_name in invalid_cases:
        unit = RGBStatisticsExtractor._get_unit_for_feature(invalid_name)
        print(f"  {invalid_name} -> {unit}")
        assert unit == "unknown", f"Expected 'unknown', got {unit}"

    print("単位取得メソッドテスト: 成功")


if __name__ == "__main__":
    test_rgb_statistics_basic()
    test_black_pixel_exclusion()
    test_pure_color_images()
    test_statistical_properties()
    test_exclude_black_pixels_option()
    test_grayscale_support()
    test_edge_cases()
    test_default_config_merging()
    test_feature_names_and_units()
    test_unit_for_feature_method()
    print("\n=== RGB統計特徴量抽出テスト完了 ===")
