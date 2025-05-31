"""輝度統計特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from feature_extractors import BrightnessStatisticsExtractor, get_feature_extractor


def test_brightness_statistics():
    """輝度統計特徴量抽出のテスト."""
    print("=== 輝度統計特徴量抽出テスト ===")

    # テスト画像の作成（グラデーション画像）
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        test_image[i, :, :] = i * 2  # 0から198までのグラデーション

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = BrightnessStatisticsExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("brightness", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for name, value in features2.items():
        print(f"  {name}: {value:.3f}")

    # 設定付きテスト（LAB色空間のL成分使用）
    print("\n--- LAB色空間L成分でのテスト ---")
    config = {"color_mode": "lab_l"}
    extractor3 = get_feature_extractor("brightness", config)
    features3 = extractor3.extract(test_image)

    print("抽出された特徴量（LAB-L成分）:")
    for name, value in features3.items():
        print(f"  {name}: {value:.3f}")

    # ROI指定テスト
    print("\n--- ROI指定テスト ---")
    config_roi = {"roi": [25, 25, 50, 50]}  # 中央部分のみ
    extractor4 = get_feature_extractor("brightness", config_roi)
    features4 = extractor4.extract(test_image)

    print("抽出された特徴量（ROI指定）:")
    for name, value in features4.items():
        print(f"  {name}: {value:.3f}")

    # 特徴量名の取得テスト
    print(
        f"\n特徴量名リスト（単位付き）: {BrightnessStatisticsExtractor.get_feature_names()}"
    )
    print(
        f"基本特徴量名リスト（単位なし）: {BrightnessStatisticsExtractor.get_base_feature_names()}"
    )
    print(f"特徴量単位辞書: {BrightnessStatisticsExtractor.get_feature_units()}")
    print(f"デフォルト設定: {BrightnessStatisticsExtractor.get_default_config()}")


def test_zero_pixel_exclusion():
    """輝度値0のピクセル除外機能のテスト."""
    print("\n=== 輝度値0除外機能テスト ===")

    extractor = BrightnessStatisticsExtractor()

    # 1. 関心領域のみの画像（背景が黒）
    roi_image = np.zeros((50, 50, 3), dtype=np.uint8)
    # 中央部分に関心領域を作成（25x25の白い正方形）
    roi_image[12:38, 12:38, :] = 200  # 輝度値200の領域

    features = extractor.extract(roi_image)
    print("関心領域のみの画像（背景黒）:")
    print(f"  全ピクセル数: {50*50}")
    print(f"  関心領域ピクセル数: {26*26}")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")

    # 2. マスク画像（0と255の2値画像）
    mask_image = np.zeros((50, 50, 3), dtype=np.uint8)
    mask_image[10:40, 10:40, :] = 255  # 30x30の白い領域

    features_mask = extractor.extract(mask_image)
    print("\nマスク画像（0と255の2値）:")
    for name, value in features_mask.items():
        print(f"  {name}: {value:.3f}")

    # 3. 輝度値0を含む混合画像
    mixed_image = np.zeros((50, 50, 3), dtype=np.uint8)
    # 一部に様々な輝度値を設定
    mixed_image[10:20, 10:20, :] = 50  # 輝度値50の領域
    mixed_image[20:30, 20:30, :] = 100  # 輝度値100の領域
    mixed_image[30:40, 30:40, :] = 150  # 輝度値150の領域
    # 残りは0（背景）

    features_mixed = extractor.extract(mixed_image)
    print("\n混合画像（複数の輝度値+背景0）:")
    for name, value in features_mixed.items():
        print(f"  {name}: {value:.3f}")

    # 4. すべてが0の画像
    zero_image = np.zeros((50, 50, 3), dtype=np.uint8)
    features_zero = extractor.extract(zero_image)
    print("\n全て0の画像:")
    for name, value in features_zero.items():
        print(f"  {name}: {value}")


def test_exclude_zero_pixels_option():
    """exclude_zero_pixelsオプションのテスト."""
    print("\n=== exclude_zero_pixelsオプションテスト ===")

    # テスト画像の作成（一部に0値を含む）
    test_image = np.zeros((50, 50, 3), dtype=np.uint8)
    # 中央部分に非ゼロ値を設定
    test_image[10:40, 10:40, :] = 100  # 30x30の領域に輝度値100
    test_image[20:30, 20:30, :] = 200  # 10x10の領域に輝度値200（重複）

    print(f"テスト画像: 全ピクセル数={50*50}, 非ゼロピクセル数={30*30}")

    # 1. exclude_zero_pixels=True（デフォルト）
    extractor_exclude = BrightnessStatisticsExtractor(
        config={"exclude_zero_pixels": True}
    )
    features_exclude = extractor_exclude.extract(test_image)
    print("\nexclude_zero_pixels=True（ゼロ値除外）:")
    for name, value in features_exclude.items():
        print(f"  {name}: {value:.3f}")

    # 2. exclude_zero_pixels=False
    extractor_include = BrightnessStatisticsExtractor(
        config={"exclude_zero_pixels": False}
    )
    features_include = extractor_include.extract(test_image)
    print("\nexclude_zero_pixels=False（ゼロ値含む）:")
    for name, value in features_include.items():
        print(f"  {name}: {value:.3f}")

    # 3. 結果の比較
    print("\n--- 結果比較 ---")
    print(
        f"平均値: 除外={features_exclude['mean']:.3f}, "
        f"含む={features_include['mean']:.3f}"
    )
    print(
        f"中央値: 除外={features_exclude['median']:.3f}, "
        f"含む={features_include['median']:.3f}"
    )
    print(
        f"分散: 除外={features_exclude['variance']:.3f}, "
        f"含む={features_include['variance']:.3f}"
    )

    # 4. 全てゼロの画像でのテスト
    zero_image = np.zeros((50, 50, 3), dtype=np.uint8)

    features_exclude_zero = extractor_exclude.extract(zero_image)
    features_include_zero = extractor_include.extract(zero_image)

    print("\n--- 全てゼロ画像での比較 ---")
    print("exclude_zero_pixels=True:")
    for name, value in features_exclude_zero.items():
        print(f"  {name}: {value}")

    print("exclude_zero_pixels=False:")
    for name, value in features_include_zero.items():
        print(f"  {name}: {value}")

    # 5. 単一非ゼロピクセルでのテスト
    single_pixel_image = np.zeros((50, 50, 3), dtype=np.uint8)
    single_pixel_image[25, 25, :] = 128  # 中央に1ピクセルだけ非ゼロ

    features_exclude_single = extractor_exclude.extract(single_pixel_image)
    features_include_single = extractor_include.extract(single_pixel_image)

    print("\n--- 単一非ゼロピクセルでの比較 ---")
    print("exclude_zero_pixels=True:")
    for name, value in features_exclude_single.items():
        print(f"  {name}: {value:.3f}")

    print("exclude_zero_pixels=False:")
    for name, value in features_include_single.items():
        print(f"  {name}: {value:.3f}")


def test_default_config_merging():
    """デフォルト設定のマージ機能のテスト."""
    print("\n=== デフォルト設定マージテスト ===")

    # テスト画像
    test_image = np.full((50, 50, 3), 100, dtype=np.uint8)

    # 1. 空の設定でインスタンス化（デフォルト設定のみ）
    extractor1 = BrightnessStatisticsExtractor()
    print(f"空設定時のconfig: {extractor1.config}")

    # 2. 部分的な設定でインスタンス化（デフォルト設定とマージ）
    partial_config = {"color_mode": "hsv_v"}
    extractor2 = BrightnessStatisticsExtractor(config=partial_config)
    print(f"部分設定時のconfig: {extractor2.config}")

    # 3. 完全な設定でインスタンス化（ユーザー設定が優先）
    full_config = {
        "color_mode": "lab_l",
        "roi": [10, 10, 30, 30],
        "exclude_zero_pixels": False,
    }
    extractor3 = BrightnessStatisticsExtractor(config=full_config)
    print(f"完全設定時のconfig: {extractor3.config}")

    # 4. 各設定での実際の動作確認
    print("\n--- 実際の処理結果 ---")

    features1 = extractor1.extract(test_image)
    print(f"デフォルト設定(gray, exclude_zero=True): mean={features1['mean']:.1f}")

    features2 = extractor2.extract(test_image)
    print(f"HSV-V成分(exclude_zero=True): mean={features2['mean']:.1f}")

    features3 = extractor3.extract(test_image)
    print(f"LAB-L成分(ROI, exclude_zero=False): mean={features3['mean']:.1f}")


def test_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = BrightnessStatisticsExtractor()

    # 単色画像（分散=0のケース）
    uniform_image = np.full((50, 50, 3), 128, dtype=np.uint8)
    features = extractor.extract(uniform_image)
    print("単色画像（128）の特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")

    # 黒画像（平均=0のケース、すべてのピクセルが除外される）
    black_image = np.zeros((50, 50, 3), dtype=np.uint8)
    features = extractor.extract(black_image)
    print("\n黒画像の特徴量（全て除外）:")
    for name, value in features.items():
        print(f"  {name}: {value}")

    # グレースケール画像
    gray_image = np.random.randint(1, 256, (50, 50), dtype=np.uint8)  # 1-255の範囲
    features = extractor.extract(gray_image)
    print("\nグレースケール画像の特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")


def test_feature_names_and_units():
    """特徴量名と単位のテスト."""
    print("\n=== 特徴量名・単位テスト ===")

    # 基本特徴量名の確認
    base_names = BrightnessStatisticsExtractor.get_base_feature_names()
    expected_base_names = ["mean", "median", "variance", "std_dev", "cv"]
    print(f"基本特徴量名: {base_names}")
    assert (
        base_names == expected_base_names
    ), f"Expected {expected_base_names}, got {base_names}"

    # 単位付き特徴量名の確認
    unit_names = BrightnessStatisticsExtractor.get_feature_names()
    expected_unit_names = [
        "mean[0-255]",
        "median[0-255]",
        "variance[0-255_squared]",
        "std_dev[0-255]",
        "cv[ratio]",
    ]
    print(f"単位付き特徴量名: {unit_names}")
    assert (
        unit_names == expected_unit_names
    ), f"Expected {expected_unit_names}, got {unit_names}"

    # 単位辞書の確認
    units = BrightnessStatisticsExtractor.get_feature_units()
    expected_units = {
        "mean": "0-255",
        "median": "0-255",
        "variance": "0-255_squared",
        "std_dev": "0-255",
        "cv": "ratio",
    }
    print(f"特徴量単位辞書: {units}")
    assert units == expected_units, f"Expected {expected_units}, got {units}"

    # 抽出結果と特徴量名の整合性確認
    extractor = BrightnessStatisticsExtractor()
    test_image = np.full((50, 50, 3), 100, dtype=np.uint8)
    features = extractor.extract(test_image)

    # 抽出された特徴量のキーが基本特徴量名と一致することを確認
    feature_keys = list(features.keys())
    print(f"抽出された特徴量のキー: {feature_keys}")
    assert set(feature_keys) == set(
        base_names
    ), f"Feature keys {feature_keys} don't match base names {base_names}"

    print("特徴量名・単位テスト: 成功")


if __name__ == "__main__":
    test_brightness_statistics()
    test_zero_pixel_exclusion()
    test_exclude_zero_pixels_option()
    test_default_config_merging()
    test_edge_cases()
    test_feature_names_and_units()
    print("\n=== 輝度統計特徴量抽出テスト完了 ===")
