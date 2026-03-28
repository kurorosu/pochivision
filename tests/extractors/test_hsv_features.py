"""HSV統計特徴量抽出機能のテストスクリプト."""

import cv2
import numpy as np
import pytest  # noqa: F401

from pochivision.feature_extractors import HSVStatisticsExtractor, get_feature_extractor


def test_hsv_statistics():
    """HSV統計特徴量抽出のテスト."""
    print("=== HSV統計特徴量抽出テスト ===")

    # テスト画像の作成（グラデーション画像）
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        test_image[i, :, 0] = i * 2  # Bチャンネル: 0から198
        test_image[i, :, 1] = (i + 50) % 256  # Gチャンネル: 50から255、0から47
        test_image[i, :, 2] = 255 - i * 2  # Rチャンネル: 255から57

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = HSVStatisticsExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("hsv", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for name, value in features2.items():
        print(f"  {name}: {value:.3f}")

    # ROI指定テスト
    print("\n--- ROI指定テスト ---")
    config_roi = {"roi": [25, 25, 50, 50]}  # 中央部分のみ
    extractor3 = get_feature_extractor("hsv", config_roi)
    features3 = extractor3.extract(test_image)

    print("抽出された特徴量（ROI指定）:")
    for name, value in features3.items():
        print(f"  {name}: {value:.3f}")

    # 特徴量名の取得テスト
    print(f"\n基本特徴量名リスト: {HSVStatisticsExtractor.get_base_feature_names()}")
    print(f"単位付き特徴量名リスト: {HSVStatisticsExtractor.get_feature_names()}")
    print(f"デフォルト設定: {HSVStatisticsExtractor.get_default_config()}")


def test_exclude_black_pixels_option():
    """exclude_black_pixelsオプションのテスト."""
    print("\n=== exclude_black_pixelsオプションテスト ===")

    # テスト画像の作成（一部に黒ピクセルを含む）
    test_image = np.zeros((50, 50, 3), dtype=np.uint8)
    # 中央部分に非ゼロ値を設定
    test_image[10:40, 10:40, 0] = 100  # 30x30の領域にB=100
    test_image[10:40, 10:40, 1] = 150  # G=150
    test_image[10:40, 10:40, 2] = 200  # R=200
    # 一部に異なる色を設定
    test_image[20:30, 20:30, 0] = 50  # 10x10の領域にB=50
    test_image[20:30, 20:30, 1] = 100  # G=100
    test_image[20:30, 20:30, 2] = 255  # R=255

    print(f"テスト画像: 全ピクセル数={50*50}, 非黒ピクセル数={30*30}")

    # 1. exclude_black_pixels=True（デフォルト）
    extractor_exclude = HSVStatisticsExtractor(config={"exclude_black_pixels": True})
    features_exclude = extractor_exclude.extract(test_image)
    print("\nexclude_black_pixels=True（黒ピクセル除外）:")
    for name, value in features_exclude.items():
        print(f"  {name}: {value:.3f}")

    # 2. exclude_black_pixels=False
    extractor_include = HSVStatisticsExtractor(config={"exclude_black_pixels": False})
    features_include = extractor_include.extract(test_image)
    print("\nexclude_black_pixels=False（黒ピクセル含む）:")
    for name, value in features_include.items():
        print(f"  {name}: {value:.3f}")

    # 3. 結果の比較
    print("\n--- 結果比較 ---")
    print(
        f"Hue平均値: 除外={features_exclude['hue_mean']:.3f}, "
        f"含む={features_include['hue_mean']:.3f}"
    )
    print(
        f"Saturation平均値: 除外={features_exclude['saturation_mean']:.3f}, "
        f"含む={features_include['saturation_mean']:.3f}"
    )
    print(
        f"Value平均値: 除外={features_exclude['value_mean']:.3f}, "
        f"含む={features_include['value_mean']:.3f}"
    )

    # 4. 全て黒の画像でのテスト
    black_image = np.zeros((50, 50, 3), dtype=np.uint8)

    features_exclude_black = extractor_exclude.extract(black_image)
    features_include_black = extractor_include.extract(black_image)

    print("\n--- 全て黒画像での比較 ---")
    print("exclude_black_pixels=True:")
    for name, value in features_exclude_black.items():
        print(f"  {name}: {value}")

    print("exclude_black_pixels=False:")
    for name, value in features_include_black.items():
        print(f"  {name}: {value}")

    # 5. 単一非黒ピクセルでのテスト
    single_pixel_image = np.zeros((50, 50, 3), dtype=np.uint8)
    single_pixel_image[25, 25, :] = [128, 64, 192]  # 中央に1ピクセルだけ非黒

    features_exclude_single = extractor_exclude.extract(single_pixel_image)
    features_include_single = extractor_include.extract(single_pixel_image)

    print("\n--- 単一非黒ピクセルでの比較 ---")
    print("exclude_black_pixels=True:")
    for name, value in features_exclude_single.items():
        print(f"  {name}: {value:.3f}")

    print("exclude_black_pixels=False:")
    for name, value in features_include_single.items():
        print(f"  {name}: {value:.3f}")


def test_hsv_color_space_properties():
    """HSV色空間の特性テスト."""
    print("\n=== HSV色空間特性テスト ===")

    extractor = HSVStatisticsExtractor()

    # 1. 純色テスト（赤、緑、青）
    colors = {
        "red": [0, 0, 255],
        "green": [0, 255, 0],
        "blue": [255, 0, 0],
        "white": [255, 255, 255],
        "gray": [128, 128, 128],
    }

    for color_name, bgr_value in colors.items():
        color_image = np.full((50, 50, 3), bgr_value, dtype=np.uint8)
        features = extractor.extract(color_image)
        print(f"\n{color_name.upper()}画像 (BGR={bgr_value}):")
        print(f"  Hue: {features['hue_mean']:.1f}")
        print(f"  Saturation: {features['saturation_mean']:.1f}")
        print(f"  Value: {features['value_mean']:.1f}")

    # 2. グラデーション画像（色相変化）
    hue_gradient = np.zeros((50, 180, 3), dtype=np.uint8)
    for i in range(180):
        # HSVで色相のみ変化させる
        hsv_color = np.array([[[i, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        hue_gradient[:, i, :] = bgr_color

    features_hue = extractor.extract(hue_gradient)
    print("\n色相グラデーション画像:")
    print(f"  Hue平均: {features_hue['hue_mean']:.1f}")
    print(f"  Hue分散: {features_hue['hue_variance']:.1f}")


def test_default_config_merging():
    """デフォルト設定のマージ機能のテスト."""
    print("\n=== デフォルト設定マージテスト ===")

    # テスト画像
    test_image = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)

    # 1. 空の設定でインスタンス化（デフォルト設定のみ）
    extractor1 = HSVStatisticsExtractor()
    print(f"空設定時のconfig: {extractor1.config}")

    # 2. 部分的な設定でインスタンス化（デフォルト設定とマージ）
    partial_config = {"roi": [10, 10, 30, 30]}
    extractor2 = HSVStatisticsExtractor(config=partial_config)
    print(f"部分設定時のconfig: {extractor2.config}")

    # 3. 完全な設定でインスタンス化（ユーザー設定が優先）
    full_config = {"exclude_black_pixels": False, "roi": [5, 5, 40, 40]}
    extractor3 = HSVStatisticsExtractor(config=full_config)
    print(f"完全設定時のconfig: {extractor3.config}")

    # 4. 各設定での実際の動作確認
    print("\n--- 実際の処理結果 ---")

    features1 = extractor1.extract(test_image)
    print(f"デフォルト設定(exclude_black=True): hue_mean={features1['hue_mean']:.1f}")

    features2 = extractor2.extract(test_image)
    print(f"ROI設定(exclude_black=True): hue_mean={features2['hue_mean']:.1f}")

    features3 = extractor3.extract(test_image)
    print(f"完全設定(exclude_black=False): hue_mean={features3['hue_mean']:.1f}")


def test_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = HSVStatisticsExtractor()

    # 単色画像（分散=0のケース）
    uniform_image = np.full((50, 50, 3), [64, 128, 192], dtype=np.uint8)
    features = extractor.extract(uniform_image)
    print("単色画像の特徴量:")
    for name, value in features.items():
        if "variance" in name or "std_dev" in name:
            print(f"  {name}: {value:.3f}")

    # 黒画像（すべてのピクセルが除外される）
    black_image = np.zeros((50, 50, 3), dtype=np.uint8)
    features = extractor.extract(black_image)
    print("\n黒画像の特徴量（全て除外）:")
    for name, value in features.items():
        print(f"  {name}: {value}")

    # エラーハンドリングテスト
    print("\n--- エラーハンドリングテスト ---")
    try:
        # 空の画像
        empty_image = np.array([])
        extractor.extract(empty_image)
        print("エラー: 空の画像が受け入れられました")
    except ValueError as e:
        print(f"正常: 空の画像でエラー発生 - {e}")


def test_grayscale_support():
    """グレースケール画像対応のテスト（新機能）."""
    print("\n=== グレースケール画像対応テスト ===")

    extractor = HSVStatisticsExtractor()

    # 1. 2次元グレースケール画像
    gray_2d = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
    features_2d = extractor.extract(gray_2d)
    print("2次元グレースケール画像の特徴量:")
    print(f"  hue_mean: {features_2d['hue_mean']:.1f}")
    print(f"  saturation_mean: {features_2d['saturation_mean']:.1f}")
    print(f"  value_mean: {features_2d['value_mean']:.1f}")
    # グレースケールなので彩度は0になるはず
    assert features_2d["saturation_mean"] < 1.0  # 彩度はほぼ0
    print("  ✓ グレースケール画像の彩度が低い値になっています")

    # 2. 3次元1チャンネル画像
    gray_3d_1ch = gray_2d[:, :, np.newaxis]  # (50, 50, 1)
    features_3d_1ch = extractor.extract(gray_3d_1ch)
    print("\n3次元1チャンネル画像の特徴量:")
    print(f"  hue_mean: {features_3d_1ch['hue_mean']:.1f}")
    print(f"  saturation_mean: {features_3d_1ch['saturation_mean']:.1f}")
    print(f"  value_mean: {features_3d_1ch['value_mean']:.1f}")
    # 2次元と同じ結果になるはず
    assert abs(features_2d["value_mean"] - features_3d_1ch["value_mean"]) < 0.1
    print("  ✓ 2次元グレースケールと同じ結果です")

    # 3. 4チャンネル画像（BGRA）
    rgba_image = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
    features_rgba = extractor.extract(rgba_image)
    print("\n4チャンネル画像（BGRA）の特徴量:")
    print(f"  hue_mean: {features_rgba['hue_mean']:.1f}")
    print(f"  saturation_mean: {features_rgba['saturation_mean']:.1f}")
    print(f"  value_mean: {features_rgba['value_mean']:.1f}")
    print("  ✓ 4チャンネル画像も正常に処理されました")

    # 4. グレースケール値の一貫性テスト
    gray_value = 128
    uniform_gray_2d = np.full((30, 30), gray_value, dtype=np.uint8)
    uniform_gray_3d = np.full((30, 30, 1), gray_value, dtype=np.uint8)
    uniform_gray_bgr = np.full((30, 30, 3), gray_value, dtype=np.uint8)

    features_gray_2d = extractor.extract(uniform_gray_2d)
    features_gray_3d = extractor.extract(uniform_gray_3d)
    features_gray_bgr = extractor.extract(uniform_gray_bgr)

    print(f"\n一様グレースケール値({gray_value})の一貫性テスト:")
    print(
        f" 2次元: V={features_gray_2d['value_mean']:.1f}, "
        f" S={features_gray_2d['saturation_mean']:.1f}"
    )
    print(
        f" 3次元1ch: V={features_gray_3d['value_mean']:.1f}, "
        f" S={features_gray_3d['saturation_mean']:.1f}"
    )
    print(
        f" 3次元3ch: V={features_gray_bgr['value_mean']:.1f}, "
        f" S={features_gray_bgr['saturation_mean']:.1f}"
    )

    # すべて同じ値になるはず
    assert abs(features_gray_2d["value_mean"] - gray_value) < 0.1
    assert abs(features_gray_3d["value_mean"] - gray_value) < 0.1
    assert abs(features_gray_bgr["value_mean"] - gray_value) < 0.1
    # 彩度はすべて0に近いはず
    assert features_gray_2d["saturation_mean"] < 1.0
    assert features_gray_3d["saturation_mean"] < 1.0
    assert features_gray_bgr["saturation_mean"] < 1.0
    print("  ✓ すべての形状で一貫した結果が得られました")


def test_feature_names_and_units():
    """特徴量名と単位のテスト."""
    print("\n=== 特徴量名・単位テスト ===")

    # 基本特徴量名の確認
    base_names = HSVStatisticsExtractor.get_base_feature_names()
    expected_base_names = []
    channels = ["hue", "saturation", "value"]
    stats = ["mean", "median", "variance", "std_dev", "cv"]
    for channel in channels:
        for stat in stats:
            expected_base_names.append(f"{channel}_{stat}")

    print(f"基本特徴量名の数: {len(base_names)}")
    print(f"基本特徴量名の例: {base_names[:5]}")
    assert (
        base_names == expected_base_names
    ), f"Expected {expected_base_names}, got {base_names}"

    # 単位付き特徴量名の確認
    unit_names = HSVStatisticsExtractor.get_feature_names()
    expected_unit_names = [
        "hue_mean[hue_0_179]",
        "hue_median[hue_0_179]",
        "hue_variance[hue_0_179_squared]",
        "hue_std_dev[hue_0_179]",
        "hue_cv[ratio]",
        "saturation_mean[intensity]",
        "saturation_median[intensity]",
        "saturation_variance[squared_intensity]",
        "saturation_std_dev[intensity]",
        "saturation_cv[ratio]",
        "value_mean[intensity]",
        "value_median[intensity]",
        "value_variance[squared_intensity]",
        "value_std_dev[intensity]",
        "value_cv[ratio]",
    ]
    print(f"単位付き特徴量名の数: {len(unit_names)}")
    print(f"単位付き特徴量名の例: {unit_names[:5]}")
    assert (
        unit_names == expected_unit_names
    ), f"Expected {expected_unit_names}, got {unit_names}"

    # 単位辞書の確認
    units = HSVStatisticsExtractor.get_feature_units()
    expected_units = {
        "hue_mean": "hue_0_179",
        "hue_median": "hue_0_179",
        "hue_variance": "hue_0_179_squared",
        "hue_std_dev": "hue_0_179",
        "hue_cv": "ratio",
        "saturation_mean": "intensity",
        "saturation_median": "intensity",
        "saturation_variance": "squared_intensity",
        "saturation_std_dev": "intensity",
        "saturation_cv": "ratio",
        "value_mean": "intensity",
        "value_median": "intensity",
        "value_variance": "squared_intensity",
        "value_std_dev": "intensity",
        "value_cv": "ratio",
    }
    print(f"特徴量単位辞書のサイズ: {len(units)}")
    print(f"特徴量単位辞書の例: {dict(list(units.items())[:5])}")
    assert units == expected_units, f"Expected {expected_units}, got {units}"

    # 抽出結果と特徴量名の整合性確認
    extractor = HSVStatisticsExtractor()
    test_image = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)
    features = extractor.extract(test_image)

    # 抽出された特徴量のキーが基本特徴量名と一致することを確認
    feature_keys = list(features.keys())
    print(f"抽出された特徴量のキー数: {len(feature_keys)}")
    assert set(feature_keys) == set(
        base_names
    ), f"Feature keys {feature_keys} don't match base names {base_names}"

    print("特徴量名・単位テスト: 成功")


def test_unit_for_feature_method():
    """_get_unit_for_feature()メソッドのテスト."""
    print("\n=== 単位取得メソッドテスト ===")

    # 正常なHSV特徴量名
    test_cases = [
        ("hue_mean", "hue_0_179"),
        ("hue_variance", "hue_0_179_squared"),
        ("saturation_mean", "intensity"),
        ("saturation_variance", "squared_intensity"),
        ("value_cv", "ratio"),
    ]

    for feature_name, expected_unit in test_cases:
        unit = HSVStatisticsExtractor._get_unit_for_feature(feature_name)
        print(f"{feature_name} -> {unit}")
        assert (
            unit == expected_unit
        ), f"Expected {expected_unit}, got {unit} for {feature_name}"

    # 無効な特徴量名
    invalid_cases = ["invalid_feature", "rgb_mean", ""]
    for invalid_name in invalid_cases:
        unit = HSVStatisticsExtractor._get_unit_for_feature(invalid_name)
        print(f"{invalid_name} -> {unit}")
        assert unit == "unknown", f"Expected 'unknown', got {unit} for {invalid_name}"

    print("単位取得メソッドテスト: 成功")


def test_hue_circular_statistics():
    """Hue チャンネルの循環統計が正しく計算されることをテスト."""
    extractor = HSVStatisticsExtractor()

    # Hue が 0/180 境界付近の赤ピクセルで構成される画像を作成
    # HSV で H=5 と H=175 はどちらも赤付近
    hsv_image = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv_image[:50, :, :] = [5, 255, 255]  # H=5 (赤)
    hsv_image[50:, :, :] = [175, 255, 255]  # H=175 (赤)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    features = extractor.extract(bgr_image)

    # 循環平均は 0 付近 (赤) であるべき, 90 (緑) ではない
    hue_mean = features["hue_mean"]
    assert (
        hue_mean < 10 or hue_mean > 170
    ), f"Hue mean should be near 0/180 (red), got {hue_mean}"

    # 循環標準偏差は小さいはず (両方とも赤付近)
    hue_std = features["hue_std_dev"]
    assert hue_std < 30, f"Hue std_dev should be small for similar hues, got {hue_std}"


def test_hue_circular_vs_linear_difference():
    """循環統計と線形統計で結果が異なるケースをテスト."""
    extractor = HSVStatisticsExtractor()

    # 全ピクセルが同じ Hue の場合, 循環統計でも分散は 0
    hsv_uniform = np.zeros((50, 50, 3), dtype=np.uint8)
    hsv_uniform[:, :, :] = [90, 200, 200]  # H=90 (緑)
    bgr_uniform = cv2.cvtColor(hsv_uniform, cv2.COLOR_HSV2BGR)

    features = extractor.extract(bgr_uniform)
    assert features["hue_variance"] < 0.1, "Uniform hue should have ~0 variance"
    assert features["hue_std_dev"] < 0.1, "Uniform hue should have ~0 std_dev"


if __name__ == "__main__":
    test_hsv_statistics()
    test_exclude_black_pixels_option()
    test_hsv_color_space_properties()
    test_default_config_merging()
    test_edge_cases()
    test_grayscale_support()
    test_feature_names_and_units()
    test_unit_for_feature_method()
    print("\n=== HSV統計特徴量抽出テスト完了 ===")
