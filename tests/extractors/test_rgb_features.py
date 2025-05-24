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
        mean_key = f"{channel}_mean"
        print(f"  {channel}_mean: {features2[mean_key]:.3f}")

    # 特徴量名の取得テスト
    print(f"\n特徴量名リスト: {RGBStatisticsExtractor.get_feature_names()}")
    print(f"デフォルト設定: {RGBStatisticsExtractor.get_default_config()}")


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


def test_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = RGBStatisticsExtractor()

    # 1. 極小画像
    tiny_image = np.array([[[255, 128, 64]]], dtype=np.uint8)  # 1x1x3画像
    features_tiny = extractor.extract(tiny_image)
    print("1x1画像の特徴量:")
    print(f"  red_mean: {features_tiny['red_mean']:.1f}")
    print(f"  red_median: {features_tiny['red_median']:.1f}")
    print(f"  red_variance: {features_tiny['red_variance']:.3f}")

    # 2. 線形画像（1次元的）
    line_image = np.zeros((1, 100, 3), dtype=np.uint8)
    for i in range(100):
        line_image[0, i, :] = [i * 2, 255 - i * 2, 128]
    features_line = extractor.extract(line_image)
    print("\n線形画像（1x100）の特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"  {channel}_mean: {features_line[f'{channel}_mean']:.1f}")

    # 3. 2値画像（0と255のみ）
    binary_image = np.random.choice([0, 255], size=(50, 50, 3), p=[0.5, 0.5]).astype(
        np.uint8
    )
    features_binary = extractor.extract(binary_image)
    print("\n2値画像の特徴量:")
    for channel in ["red", "green", "blue"]:
        print(f"  {channel}_mean: {features_binary[f'{channel}_mean']:.1f}")
        print(f"  {channel}_std_dev: {features_binary[f'{channel}_std_dev']:.1f}")


def test_error_cases():
    """エラーケースのテスト."""
    print("\n=== エラーケーステスト ===")

    extractor = RGBStatisticsExtractor()

    # 1. 空の画像
    try:
        empty_image = np.array([])
        extractor.extract(empty_image)
        print("エラー: 空画像で例外が発生しませんでした")
    except ValueError as e:
        print(f"空画像のエラー処理: {e}")

    # 2. Noneの画像
    try:
        extractor.extract(None)
        print("エラー: None画像で例外が発生しませんでした")
    except ValueError as e:
        print(f"None画像のエラー処理: {e}")

    # 3. グレースケール画像（2次元）
    try:
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        extractor.extract(gray_image)
        print("エラー: グレースケール画像で例外が発生しませんでした")
    except ValueError as e:
        print(f"グレースケール画像のエラー処理: {e}")

    # 4. 4チャンネル画像
    try:
        rgba_image = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        extractor.extract(rgba_image)
        print("エラー: 4チャンネル画像で例外が発生しませんでした")
    except ValueError as e:
        print(f"4チャンネル画像のエラー処理: {e}")


def test_config_validation():
    """設定の検証テスト."""
    print("\n=== 設定検証テスト ===")

    # 正常な設定
    valid_configs = [
        {"exclude_black_pixels": True},
        {"exclude_black_pixels": False},
        {},  # デフォルト設定使用
    ]

    for i, config in enumerate(valid_configs):
        try:
            extractor = RGBStatisticsExtractor(config=config)
            print(f"設定{i+1}: {config} - 正常")
            print(f"  実際の設定: {extractor.config}")
        except Exception as e:
            print(f"設定{i+1}: {config} - エラー: {e}")


if __name__ == "__main__":
    test_rgb_statistics_basic()
    test_black_pixel_exclusion()
    test_pure_color_images()
    test_statistical_properties()
    test_edge_cases()
    test_error_cases()
    test_config_validation()
    print("\n=== RGB統計特徴量抽出テスト完了 ===")
