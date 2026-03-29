"""HLACテクスチャ特徴量抽出機能のテストスクリプト."""

import numpy as np
import pytest  # noqa: F401

from pochivision.feature_extractors import HLACTextureExtractor, get_feature_extractor


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

    # 基本的な特徴量が含まれているかチェック（37次元）
    assert len(features) == 37, f"特徴量数が37ではありません: {len(features)}"

    # 特徴量名の形式チェック（ゼロパディング対応）
    for i in range(37):
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

    # 1. 標準HLAC（37次元）
    print("\n--- 標準HLAC (37次元) ---")
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
        len(features_standard) == 37
    ), f"標準HLACの特徴量数が37ではありません: {len(features_standard)}"
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
        len(features_order2) == 37
    ), f"2次HLACの特徴量数が37ではありません: {len(features_order2)}"


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
    extractor = HLACTextureExtractor()

    # 空の画像で ValueError
    with pytest.raises(ValueError, match="empty or None"):
        extractor.extract(np.array([]))

    # None画像で ValueError
    with pytest.raises(ValueError, match="empty or None"):
        extractor.extract(None)


def test_hlac_feature_names_and_config():
    """特徴量名とコンフィグのテスト."""
    print("\n=== 特徴量名とコンフィグテスト ===")

    # デフォルト設定の確認
    default_config = HLACTextureExtractor.get_default_config()
    print("デフォルト設定:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")

    # 基本特徴量名の確認
    base_feature_names = HLACTextureExtractor.get_base_feature_names()
    print(f"\n基本特徴量名（数: {len(base_feature_names)}）:")
    for i, name in enumerate(base_feature_names):
        if i < 10:  # 最初の10個のみ表示
            print(f"  {name}")

    # 単位付き特徴量名の確認
    unit_feature_names = HLACTextureExtractor.get_feature_names()
    print(f"\n単位付き特徴量名（数: {len(unit_feature_names)}）:")
    for i, name in enumerate(unit_feature_names):
        if i < 10:  # 最初の10個のみ表示
            print(f"  {name}")

    # 実際の抽出結果と基本特徴量名が一致するかチェック
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    extractor = HLACTextureExtractor()
    features = extractor.extract(test_image)

    assert len(features) == len(
        base_feature_names
    ), "特徴量数と基本特徴量名の数が一致しません"
    for name in base_feature_names:
        assert name in features, f"基本特徴量名 {name} が実際の特徴量に含まれていません"


def test_hlac_config_merging():
    """設定のマージ機能のテスト."""
    print("\n=== 設定マージテスト ===")

    # テスト画像
    test_image = np.full((64, 64, 3), 100, dtype=np.uint8)

    # 1. 空の設定（デフォルト設定のみ）
    extractor1 = HLACTextureExtractor()
    print(f"デフォルト設定: {extractor1.config}")

    # 2. 部分的な設定（デフォルト設定とマージ）
    partial_config = {"order": 1, "normalize": False}
    extractor2 = HLACTextureExtractor(config=partial_config)
    print(f"部分設定: {extractor2.config}")

    # 3. 完全な設定（ユーザー設定が優先）
    full_config = {
        "order": 1,
        "rotate_invariant": True,
        "normalize": False,
        "scales": [1.0],
        "resize_shape": [32, 32],
    }
    extractor3 = HLACTextureExtractor(config=full_config)
    print(f"完全設定: {extractor3.config}")

    # 各設定での実際の動作確認
    features1 = extractor1.extract(test_image)
    features2 = extractor2.extract(test_image)
    features3 = extractor3.extract(test_image)

    print(f"デフォルト設定での特徴量数: {len(features1)}")
    print(f"部分設定での特徴量数: {len(features2)}")
    print(f"完全設定での特徴量数: {len(features3)}")


def test_feature_names_and_units():
    """特徴量名と単位のテスト."""
    print("\n=== 特徴量名・単位テスト ===")

    # 基本特徴量名の確認
    base_names = HLACTextureExtractor.get_base_feature_names()
    expected_base_names = [f"hlac_feature_{i:02d}" for i in range(37)]
    print(f"基本特徴量名の数: {len(base_names)}")
    print(f"基本特徴量名の例: {base_names[:5]}")
    assert (
        base_names == expected_base_names
    ), f"Expected {expected_base_names[:5]}..., got {base_names[:5]}..."

    # 単位付き特徴量名の確認
    unit_names = HLACTextureExtractor.get_feature_names()
    expected_unit_names = [
        f"hlac_feature_{i:02d}[correlation_coefficient]" for i in range(37)
    ]
    print(f"単位付き特徴量名の数: {len(unit_names)}")
    print(f"単位付き特徴量名の例: {unit_names[:5]}")
    assert (
        unit_names == expected_unit_names
    ), f"Expected {expected_unit_names[:5]}..., got {unit_names[:5]}..."

    # 単位辞書の確認
    units = HLACTextureExtractor.get_feature_units()
    expected_units = {
        f"hlac_feature_{i:02d}": "correlation_coefficient" for i in range(37)
    }
    print(f"特徴量単位辞書のサイズ: {len(units)}")
    print(f"特徴量単位辞書の例: {dict(list(units.items())[:5])}")
    assert units == expected_units, f"Expected {expected_units}, got {units}"

    # 抽出結果と特徴量名の整合性確認
    extractor = HLACTextureExtractor()
    test_image = np.full((64, 64, 3), 100, dtype=np.uint8)
    features = extractor.extract(test_image)

    # 抽出された特徴量のキーが基本特徴量名と一致することを確認
    feature_keys = list(features.keys())
    print(f"抽出された特徴量のキー数: {len(feature_keys)}")
    assert set(feature_keys) == set(
        base_names
    ), f"Feature keys {feature_keys[:5]}... don't match base names {base_names[:5]}..."

    print("特徴量名・単位テスト: 成功")


def test_unit_for_feature_method():
    """_get_unit_for_feature()メソッドのテスト."""
    print("\n=== 単位取得メソッドテスト ===")

    # 正常なHLAC特徴量名
    test_cases = [
        ("hlac_feature_00", "correlation_coefficient"),
        ("hlac_feature_01", "correlation_coefficient"),
        ("hlac_feature_44", "correlation_coefficient"),
    ]

    for feature_name, expected_unit in test_cases:
        unit = HLACTextureExtractor._get_unit_for_feature(feature_name)
        print(f"{feature_name} -> {unit}")
        assert (
            unit == expected_unit
        ), f"Expected {expected_unit}, got {unit} for {feature_name}"

    # 無効な特徴量名
    invalid_cases = ["invalid_feature", "other_feature_00", ""]
    for invalid_name in invalid_cases:
        unit = HLACTextureExtractor._get_unit_for_feature(invalid_name)
        print(f"{invalid_name} -> {unit}")
        assert unit == "unknown", f"Expected 'unknown', got {unit} for {invalid_name}"

    print("単位取得メソッドテスト: 成功")


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
    test_feature_names_and_units()
    test_unit_for_feature_method()
    print("\n=== HLACテクスチャ特徴量抽出テスト完了 ===")


class TestHLACBehavior:
    """HLAC 特徴量の振る舞いテスト."""

    _CONFIG_RAW = {
        "order": 2,
        "rotate_invariant": False,
        "normalize": False,
        "scales": [1.0],
        "resize_shape": None,
        "binarization_method": "adaptive",
        "adaptive_block_size": 11,
        "adaptive_c": 2,
        "preserve_aspect_ratio": True,
        "aspect_ratio_mode": "width",
    }
    _CONFIG_NORM = {
        "order": 2,
        "rotate_invariant": False,
        "normalize": True,
        "scales": [1.0],
        "resize_shape": None,
        "binarization_method": "adaptive",
        "adaptive_block_size": 11,
        "adaptive_c": 2,
        "preserve_aspect_ratio": True,
        "aspect_ratio_mode": "width",
    }

    @staticmethod
    def _make_white(size: int = 32) -> np.ndarray:
        return np.full((size, size), 255, dtype=np.uint8)

    @staticmethod
    def _make_black(size: int = 32) -> np.ndarray:
        return np.zeros((size, size), dtype=np.uint8)

    @staticmethod
    def _make_checker(size: int = 32) -> np.ndarray:
        img = np.zeros((size, size), dtype=np.uint8)
        img[0::2, 0::2] = 255
        img[1::2, 1::2] = 255
        return img

    @staticmethod
    def _make_h_stripe(size: int = 32, period: int = 4) -> np.ndarray:
        img = np.zeros((size, size), dtype=np.uint8)
        for k in range(period // 2):
            img[k::period, :] = 255
        return img

    @staticmethod
    def _make_random(size: int = 32) -> np.ndarray:
        np.random.seed(42)
        return np.random.randint(0, 256, (size, size), dtype=np.uint8)

    # --- 全白画像 ---

    def test_white_0th_order_equals_valid_pixels(self):
        """全白画像の 0th order は valid 領域のピクセル数と一致."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_white())
        # 32x32 画像で mode="valid" + 3x3 カーネル → 30x30 = 900
        assert f["hlac_feature_00"] == 900

    def test_white_1st_order_all_equal(self):
        """全白画像の 1st order は全方向同じ値."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_white())
        values = [f[f"hlac_feature_{i:02d}"] for i in range(1, 9)]
        assert len(set(values)) == 1  # 全方向同値

    def test_white_equals_black(self):
        """全白と全黒は adaptive 二値化後に同じパターンになる."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        fw = ext.extract(self._make_white())
        fb = ext.extract(self._make_black())
        for i in range(37):
            assert fw[f"hlac_feature_{i:02d}"] == fb[f"hlac_feature_{i:02d}"]

    # --- チェッカーボード ---

    def test_checker_1st_order_axis_zero(self):
        """チェッカーボードの 1st order は軸方向 (feature_01,03,05,07) がゼロ."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_checker())
        # offsets: 01=(0,-1), 03=(-1,0), 05=(0,1), 07=(1,0) が軸方向
        for i in [1, 3, 5, 7]:
            assert f[f"hlac_feature_{i:02d}"] == 0

    def test_checker_1st_order_diagonal_nonzero(self):
        """チェッカーボードの 1st order は対角方向 (feature_02,04,06,08) が非ゼロ."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_checker())
        for i in [2, 4, 6, 8]:
            assert f[f"hlac_feature_{i:02d}"] > 0

    # --- 水平ストライプ ---

    def test_h_stripe_vertical_direction_highest(self):
        """水平ストライプは垂直方向 (feature_03, 07) の 1st order が最大."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_h_stripe())
        # feature_03=(-1,0)上, feature_07=(1,0)下 が垂直方向
        vertical = f["hlac_feature_03"]
        horizontal = f["hlac_feature_01"]  # (0,-1)左
        assert vertical > horizontal

    # --- ランダム ---

    def test_random_1st_order_nearly_uniform(self):
        """ランダム画像の 1st order は全方向ほぼ均等."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        f = ext.extract(self._make_random())
        values = [f[f"hlac_feature_{i:02d}"] for i in range(1, 9)]
        mean_val = sum(values) / len(values)
        for v in values:
            assert abs(v - mean_val) / mean_val < 0.1  # 10% 以内

    # --- ランダム vs チェッカーボード ---

    def test_checker_differs_from_random(self):
        """チェッカーボードとランダムの特徴量が区別可能."""
        ext = HLACTextureExtractor(config=self._CONFIG_NORM)
        fc = ext.extract(self._make_checker())
        fr = ext.extract(self._make_random())
        # チェッカーボードの軸方向 1st order = 0, ランダムは > 0
        assert fc["hlac_feature_01"] == 0
        assert fr["hlac_feature_01"] > 0

    # --- 正規化 ---

    def test_normalized_sum_is_one(self):
        """正規化時の全特徴量の合計は ~1.0."""
        ext = HLACTextureExtractor(config=self._CONFIG_NORM)
        f = ext.extract(self._make_random())
        total = sum(f.values())
        assert 0.99 <= total <= 1.01

    def test_normalized_0th_less_than_one(self):
        """正規化時の 0th order は 1.0 未満."""
        ext = HLACTextureExtractor(config=self._CONFIG_NORM)
        f = ext.extract(self._make_random())
        assert f["hlac_feature_00"] < 1.0

    # --- 水平 vs 垂直ストライプ ---

    def test_h_stripe_differs_from_v_stripe(self):
        """水平ストライプと垂直ストライプは方向性特徴量が異なる."""
        ext = HLACTextureExtractor(config=self._CONFIG_RAW)
        v_stripe = np.zeros((32, 32), dtype=np.uint8)
        v_stripe[:, 0::4] = 255
        v_stripe[:, 1::4] = 255

        fh = ext.extract(self._make_h_stripe())
        fv = ext.extract(v_stripe)
        # 水平ストライプは feature_03 (上) が高い, 垂直ストライプは feature_01 (左) が高い
        assert fh["hlac_feature_03"] > fh["hlac_feature_01"]
        assert fv["hlac_feature_01"] > fv["hlac_feature_03"]
