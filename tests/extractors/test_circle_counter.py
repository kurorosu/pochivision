"""円カウント特徴量抽出機能のテストスクリプト."""

import cv2
import numpy as np
import pytest  # noqa: F401

from feature_extractors import CircleCounterExtractor, get_feature_extractor


def test_circle_counter_basic():
    """円カウント特徴量抽出の基本テスト."""
    print("=== 円カウント特徴量抽出基本テスト ===")

    # テスト画像の作成（白背景に黒い円）
    test_image = np.full((200, 200, 3), 255, dtype=np.uint8)

    # 複数の円を描画
    cv2.circle(test_image, (50, 50), 20, (0, 0, 0), -1)  # 小さい円
    cv2.circle(test_image, (150, 50), 30, (0, 0, 0), -1)  # 中くらいの円
    cv2.circle(test_image, (100, 150), 40, (0, 0, 0), -1)  # 大きい円

    print(f"テスト画像サイズ: {test_image.shape}")

    # 直接インスタンス化でのテスト
    extractor = CircleCounterExtractor()
    features = extractor.extract(test_image)

    print("抽出された特徴量:")
    for key, value in features.items():
        if isinstance(value, float):
            if key == "circle_density":
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # レジストリ経由でのテスト
    print("\n--- レジストリ経由でのテスト ---")
    extractor2 = get_feature_extractor("circle_counter", {})
    features2 = extractor2.extract(test_image)

    print("抽出された特徴量（レジストリ経由）:")
    for key, value in features2.items():
        if isinstance(value, float):
            if key == "circle_density":
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def test_no_circles():
    """円が存在しない画像のテスト."""
    print("\n=== 円なし画像テスト ===")

    extractor = CircleCounterExtractor()

    # 1. 無地の画像
    plain_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    features_plain = extractor.extract(plain_image)

    print("無地画像の特徴量:")
    for key, value in features_plain.items():
        print(f"  {key}: {value}")

    # 2. ノイズ画像
    np.random.seed(42)
    noise_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    features_noise = extractor.extract(noise_image)

    print("\nノイズ画像の特徴量:")
    for key, value in features_noise.items():
        print(f"  {key}: {value}")

    # 3. 直線のみの画像
    lines_image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.line(lines_image, (10, 10), (90, 90), (0, 0, 0), 2)
    cv2.line(lines_image, (10, 90), (90, 10), (0, 0, 0), 2)
    cv2.rectangle(lines_image, (30, 30), (70, 70), (0, 0, 0), 2)

    features_lines = extractor.extract(lines_image)

    print("\n直線・矩形画像の特徴量:")
    for key, value in features_lines.items():
        print(f"  {key}: {value}")


def test_perfect_circles():
    """完全な円の検出テスト."""
    print("\n=== 完全な円検出テスト ===")

    extractor = CircleCounterExtractor()

    # 白背景に黒い完全な円
    test_image = np.full((200, 200, 3), 255, dtype=np.uint8)

    # 異なるサイズの円を描画
    circles_data = [
        ((60, 60), 15),  # 小さい円
        ((140, 60), 25),  # 中くらいの円
        ((60, 140), 35),  # 大きい円
        ((140, 140), 45),  # 最大の円
    ]

    for center, radius in circles_data:
        cv2.circle(test_image, center, radius, (0, 0, 0), -1)

    features = extractor.extract(test_image)

    print(f"描画した円の数: {len(circles_data)}")
    print("検出結果:")
    for key, value in features.items():
        if isinstance(value, float):
            if key == "circle_density":
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def test_ellipses_vs_circles():
    """楕円と円の区別テスト."""
    print("\n=== 楕円vs円区別テスト ===")

    # 真円度フィルタ有効
    extractor_filter = CircleCounterExtractor(
        config={"enable_circularity_filter": True, "circularity_threshold": 0.8}
    )

    # 真円度フィルタ無効
    extractor_no_filter = CircleCounterExtractor(
        config={"enable_circularity_filter": False}
    )

    # テスト画像作成（円と楕円を含む）
    test_image = np.full((200, 200, 3), 255, dtype=np.uint8)

    # 完全な円
    cv2.circle(test_image, (60, 60), 25, (0, 0, 0), -1)

    # 楕円（真円度が低い）
    cv2.ellipse(test_image, (140, 60), (40, 20), 0, 0, 360, (0, 0, 0), -1)
    cv2.ellipse(test_image, (60, 140), (20, 35), 45, 0, 360, (0, 0, 0), -1)

    print("真円度フィルタ有効:")
    features_filter = extractor_filter.extract(test_image)
    for key, value in features_filter.items():
        if "count" in key:
            print(f"  {key}: {value}")

    print("\n真円度フィルタ無効:")
    features_no_filter = extractor_no_filter.extract(test_image)
    for key, value in features_no_filter.items():
        if "count" in key:
            print(f"  {key}: {value}")


def test_parameter_variations():
    """パラメータ変更のテスト."""
    print("\n=== パラメータ変更テスト ===")

    # テスト画像作成
    test_image = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 15, (0, 0, 0), -1)
    cv2.circle(test_image, (100, 50), 25, (0, 0, 0), -1)
    cv2.circle(test_image, (75, 100), 20, (0, 0, 0), -1)

    # 1. 感度の高い設定（より多くの円を検出）
    print("--- 高感度設定 ---")
    extractor_sensitive = CircleCounterExtractor(
        config={
            "param2": 20,  # より低い閾値
            "min_radius": 3,  # より小さい最小半径
            "circularity_threshold": 0.5,  # より緩い真円度
        }
    )
    features_sensitive = extractor_sensitive.extract(test_image)
    print(f"検出円数: {features_sensitive['circle_count']}")

    # 2. 感度の低い設定（より厳格な検出）
    print("\n--- 低感度設定 ---")
    extractor_strict = CircleCounterExtractor(
        config={
            "param2": 50,  # より高い閾値
            "min_radius": 10,  # より大きい最小半径
            "circularity_threshold": 0.9,  # より厳格な真円度
        }
    )
    features_strict = extractor_strict.extract(test_image)
    print(f"検出円数: {features_strict['circle_count']}")

    # 3. デフォルト設定
    print("\n--- デフォルト設定 ---")
    extractor_default = CircleCounterExtractor()
    features_default = extractor_default.extract(test_image)
    print(f"検出円数: {features_default['circle_count']}")


def test_size_classification():
    """サイズ分類のテスト."""
    print("\n=== サイズ分類テスト ===")

    extractor = CircleCounterExtractor()

    # 異なるサイズの円を描画
    test_image = np.full((300, 300, 3), 255, dtype=np.uint8)

    # 小、中、大の円を描画
    # 最大半径が75（画像サイズ300の1/4）として、
    # 小: <25, 中: 25-50, 大: >=50
    circles = [
        ((75, 75), 10),  # 小 (10 < 25)
        ((225, 75), 15),  # 小 (15 < 25)
        ((75, 150), 30),  # 中 (25 <= 30 < 50)
        ((225, 150), 40),  # 中 (25 <= 40 < 50)
        ((75, 225), 55),  # 大 (55 >= 50)
        ((225, 225), 65),  # 大 (65 >= 50)
    ]

    for center, radius in circles:
        cv2.circle(test_image, center, radius, (0, 0, 0), -1)

    features = extractor.extract(test_image)

    print("描画した円: 小*2, 中*2, 大*2")
    print("検出結果:")
    print(f"  total_count: {features['circle_count']}")
    print(f"  small_count: {features['small_circle_count']}")
    print(f"  medium_count: {features['medium_circle_count']}")
    print(f"  large_count: {features['large_circle_count']}")


def test_density_calculation():
    """密度計算のテスト."""
    print("\n=== 密度計算テスト ===")

    extractor = CircleCounterExtractor()

    # 小さい画像に少数の円
    small_image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(small_image, (50, 50), 15, (0, 0, 0), -1)

    features_small = extractor.extract(small_image)
    density_small = features_small["circle_density"]

    print(f"小画像 (100x100, 1円): density = {density_small:.2e}")

    # 大きい画像に多数の円
    large_image = np.full((200, 200, 3), 255, dtype=np.uint8)
    for x in range(50, 200, 50):
        for y in range(50, 200, 50):
            cv2.circle(large_image, (x, y), 15, (0, 0, 0), -1)

    features_large = extractor.extract(large_image)
    density_large = features_large["circle_density"]

    print(f"大画像 (200x200, 16円): density = {density_large:.2e}")


def test_blur_effect():
    """ブラー効果のテスト."""
    print("\n=== ブラー効果テスト ===")

    # ノイジーな画像を作成
    test_image = np.full((150, 150, 3), 255, dtype=np.uint8)

    # 円を描画
    cv2.circle(test_image, (75, 75), 25, (0, 0, 0), -1)

    # ノイズを追加
    np.random.seed(42)
    noise = np.random.randint(-30, 31, test_image.shape, dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ブラーなし
    extractor_no_blur = CircleCounterExtractor(config={"blur_kernel_size": 0})
    features_no_blur = extractor_no_blur.extract(test_image)

    # ブラーあり
    extractor_blur = CircleCounterExtractor(config={"blur_kernel_size": 5})
    features_blur = extractor_blur.extract(test_image)

    print(f"ブラーなし - 検出円数: {features_no_blur['circle_count']}")
    print(f"ブラーあり - 検出円数: {features_blur['circle_count']}")


def test_edge_cases():
    """エッジケースのテスト."""
    print("\n=== エッジケーステスト ===")

    extractor = CircleCounterExtractor()

    # 1. 空の画像
    try:
        empty_image = np.array([])
        extractor.extract(empty_image)
    except ValueError as e:
        print(f"空画像エラー: {e}")

    # 2. Noneの画像
    try:
        extractor.extract(None)
    except ValueError as e:
        print(f"None画像エラー: {e}")

    # 3. 1次元画像
    try:
        one_dim_image = np.array([1, 2, 3])
        features = extractor.extract(one_dim_image)
        print(f"1次元画像: circle_count = {features['circle_count']}")
    except Exception as e:
        print(f"1次元画像エラー: {e}")

    # 4. 非常に小さい画像
    tiny_image = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.circle(tiny_image, (5, 5), 3, (0, 0, 0), -1)
    features_tiny = extractor.extract(tiny_image)
    print(f"極小画像 (10x10): circle_count = {features_tiny['circle_count']}")

    # 5. 非常に大きい画像（リソースを考慮して適度なサイズ）
    large_image = np.full((500, 500, 3), 255, dtype=np.uint8)
    cv2.circle(large_image, (250, 250), 50, (0, 0, 0), -1)
    features_large = extractor.extract(large_image)
    print(f"大画像 (500x500): circle_count = {features_large['circle_count']}")


def test_default_config_merging():
    """デフォルト設定のマージテスト."""
    print("\n=== デフォルト設定マージテスト ===")

    # 1. デフォルト設定
    extractor_default = CircleCounterExtractor()
    print("デフォルト設定:")
    for key, value in extractor_default.config.items():
        print(f"  {key}: {value}")

    # 2. 部分的な設定オーバーライド
    custom_config = {
        "min_radius": 10,
        "param2": 25,
    }
    extractor_custom = CircleCounterExtractor(config=custom_config)
    print(f"\nカスタム設定 {custom_config}:")
    for key, value in extractor_custom.config.items():
        print(f"  {key}: {value}")


def test_feature_names_and_units():
    """特徴量名と単位のテスト."""
    print("\n=== 特徴量名・単位テスト ===")

    # 基本特徴量名
    base_names = CircleCounterExtractor.get_base_feature_names()
    print("基本特徴量名:")
    for name in base_names:
        print(f"  {name}")

    # 単位付き特徴量名
    feature_names = CircleCounterExtractor.get_feature_names()
    print("\n単位付き特徴量名:")
    for name in feature_names:
        print(f"  {name}")

    # 特徴量の単位辞書
    units = CircleCounterExtractor.get_feature_units()
    print("\n特徴量単位辞書:")
    for name, unit in units.items():
        print(f"  {name}: {unit}")

    # 実際の抽出結果と一致確認
    extractor = CircleCounterExtractor()
    test_image = np.full((100, 100, 3), 255, dtype=np.uint8)
    features = extractor.extract(test_image)

    print("\n実際の抽出特徴量名:")
    actual_names = list(features.keys())
    for name in actual_names:
        print(f"  {name}")

    # 一致確認
    missing_in_actual = set(base_names) - set(actual_names)
    missing_in_base = set(actual_names) - set(base_names)

    if missing_in_actual:
        print(f"\n基本名にあるが実際にない: {missing_in_actual}")
    if missing_in_base:
        print(f"\n実際にあるが基本名にない: {missing_in_base}")
    if not missing_in_actual and not missing_in_base:
        print("\n✓ 特徴量名の一致確認: OK")


def test_unit_for_feature_method():
    """特徴量の単位取得メソッドのテスト."""
    print("\n=== 単位取得メソッドテスト ===")

    # 各特徴量の単位をテスト
    test_features = [
        "circle_count",
        "small_circle_count",
        "medium_circle_count",
        "large_circle_count",
        "circle_density",
        "avg_circle_radius",
        "radius_std",
        "unknown_feature",  # 存在しない特徴量
    ]

    for feature in test_features:
        unit = CircleCounterExtractor._get_unit_for_feature(feature)
        print(f"  {feature}: {unit}")


if __name__ == "__main__":
    # 基本テスト実行
    test_circle_counter_basic()
    test_no_circles()
    test_perfect_circles()
    test_ellipses_vs_circles()
    test_parameter_variations()
    test_size_classification()
    test_density_calculation()
    test_blur_effect()
    test_edge_cases()
    test_default_config_merging()
    test_feature_names_and_units()
    test_unit_for_feature_method()

    print("\n=== すべてのテスト完了 ===")
