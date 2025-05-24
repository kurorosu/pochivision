"""特徴量抽出機能のテストスクリプト."""

import numpy as np

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
    print(f"\n特徴量名リスト: {BrightnessStatisticsExtractor.get_feature_names()}")
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
    full_config = {"color_mode": "lab_l", "roi": [10, 10, 30, 30]}
    extractor3 = BrightnessStatisticsExtractor(config=full_config)
    print(f"完全設定時のconfig: {extractor3.config}")

    # 4. 各設定での実際の動作確認
    print("\n--- 実際の処理結果 ---")

    features1 = extractor1.extract(test_image)
    print(f"デフォルト設定(gray): mean={features1['mean']:.1f}")

    features2 = extractor2.extract(test_image)
    print(f"HSV-V成分: mean={features2['mean']:.1f}")

    features3 = extractor3.extract(test_image)
    print(f"LAB-L成分(ROI): mean={features3['mean']:.1f}")


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


if __name__ == "__main__":
    test_brightness_statistics()
    test_zero_pixel_exclusion()
    test_default_config_merging()
    test_edge_cases()
    print("\n=== テスト完了 ===")
