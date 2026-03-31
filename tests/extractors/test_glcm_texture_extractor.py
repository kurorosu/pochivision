"""GLCM特徴量抽出器のテストモジュール."""

import numpy as np
import pytest

from pochivision.exceptions.extractor import ExtractorValidationError
from pochivision.feature_extractors.glcm_texture import GLCMTextureExtractor
from tests.extractors.conftest import DummyImages


class TestGLCMTextureExtractor:
    """GLCMTextureExtractorのテストクラス."""

    def test_init_default(self):
        """デフォルト設定でのインスタンス化をテスト."""
        extractor = GLCMTextureExtractor()

        assert extractor.name == "glcm_texture"
        assert extractor.distances == [1, 2, 3]
        assert len(extractor.angles) == 4  # 0°, 45°, 90°, 135°
        assert extractor.levels == 256
        assert extractor.symmetric is True
        assert extractor.normed is True
        assert len(extractor.properties) == 6

    def test_init_custom_config(self):
        """カスタム設定でのインスタンス化をテスト."""
        custom_config = {
            "distances": [1, 2],
            "angles": [0, 90],  # 度数で指定
            "levels": 64,
            "symmetric": False,
            "normed": False,
            "properties": ["contrast", "energy"],
        }

        extractor = GLCMTextureExtractor(config=custom_config)

        assert extractor.distances == [1, 2]
        assert len(extractor.angles) == 2
        assert extractor.levels == 64
        assert extractor.symmetric is False
        assert extractor.normed is False
        assert extractor.properties == ["contrast", "energy"]

    def test_angle_degrees_list(self):
        """度数リストでの角度設定をテスト."""
        # 水平方向のみ
        extractor_h = GLCMTextureExtractor(config={"angles": [0]})
        assert len(extractor_h.angles) == 1
        assert np.isclose(extractor_h.angles[0], 0)

        # 垂直方向のみ
        extractor_v = GLCMTextureExtractor(config={"angles": [90]})
        assert len(extractor_v.angles) == 1
        assert np.isclose(extractor_v.angles[0], np.pi / 2)

        # 対角線方向
        extractor_d = GLCMTextureExtractor(config={"angles": [45, 135]})
        assert len(extractor_d.angles) == 2

        # 標準4方向
        extractor_s = GLCMTextureExtractor(config={"angles": [0, 45, 90, 135]})
        assert len(extractor_s.angles) == 4

        # 8方向
        extractor_8 = GLCMTextureExtractor(
            config={"angles": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]}
        )
        assert len(extractor_8.angles) == 8

    def test_angle_degrees_input(self):
        """度数での角度入力をテスト."""
        # 度数で指定
        extractor = GLCMTextureExtractor(config={"angles": [0, 45, 90, 135]})
        assert len(extractor.angles) == 4

        # 最初の角度が0度（0ラジアン）であることを確認
        assert np.isclose(extractor.angles[0], 0)
        # 2番目の角度が45度（π/4ラジアン）であることを確認
        assert np.isclose(extractor.angles[1], np.pi / 4)

    def test_invalid_angle_input(self):
        """無効な角度入力のエラーハンドリングをテスト."""
        # 文字列での指定（プリセットは廃止）
        with pytest.raises(
            (ValueError, ExtractorValidationError),
            match="Angles must be a list of numbers",
        ):
            GLCMTextureExtractor(config={"angles": "invalid_preset"})

        # 空のリスト
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Angles list cannot be empty"
        ):
            GLCMTextureExtractor(config={"angles": []})

        # 数値以外の要素を含むリスト
        with pytest.raises(
            (ValueError, ExtractorValidationError),
            match="All angles must be numeric values",
        ):
            GLCMTextureExtractor(config={"angles": [0, "45", 90]})

    def test_extract_basic_functionality(self):
        """基本的な特徴量抽出をテスト."""
        # シンプルなテスト画像を作成
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        extractor = GLCMTextureExtractor()
        features = extractor.extract(image)

        # 特徴量が辞書として返されることを確認
        assert isinstance(features, dict)
        assert len(features) > 0

        # すべての値が数値であることを確認
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_extract_grayscale_image(self):
        """グレースケール画像での特徴量抽出をテスト."""
        # グレースケール画像を作成
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        extractor = GLCMTextureExtractor()
        features = extractor.extract(image)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_uniform_image(self):
        """均一な画像での特徴量抽出をテスト."""
        # 均一な画像（すべて同じ値）
        image = np.full((50, 50, 3), 128, dtype=np.uint8)

        extractor = GLCMTextureExtractor()
        features = extractor.extract(image)

        assert isinstance(features, dict)
        assert len(features) > 0

        # 均一な画像では多くの特徴量が0になることを確認
        # （ただし、すべてが0になるとは限らない）
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_extract_checkerboard_pattern(self):
        """チェッカーボードパターンでの特徴量抽出をテスト."""
        # チェッカーボードパターンを作成
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            for j in range(50):
                if (i + j) % 2 == 0:
                    image[i, j] = [255, 255, 255]
                else:
                    image[i, j] = [0, 0, 0]

        extractor = GLCMTextureExtractor()
        features = extractor.extract(image)

        assert isinstance(features, dict)
        assert len(features) > 0

        # チェッカーボードパターンではコントラストが高くなることを期待
        contrast_features = [k for k in features.keys() if k.startswith("contrast")]
        assert len(contrast_features) > 0

    def test_extract_small_image(self):
        """小さな画像での特徴量抽出をテスト."""
        # 非常に小さな画像
        image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

        extractor = GLCMTextureExtractor()
        features = extractor.extract(image)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_custom_properties(self):
        """カスタムプロパティでの特徴量抽出をテスト."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # 特定のプロパティのみを指定
        config = {"properties": ["contrast", "energy"], "distances": [1], "angles": [0]}

        extractor = GLCMTextureExtractor(config=config)
        features = extractor.extract(image)

        # 指定したプロパティのみが含まれることを確認
        expected_features = ["contrast_1_0", "energy_1_0"]
        assert len(features) == len(expected_features)

        for feature_name in expected_features:
            assert feature_name in features

    def test_extract_multiple_distances_angles(self):
        """複数の距離と角度での特徴量抽出をテスト."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        config = {
            "distances": [1, 2, 3],
            "angles": [0, 90],  # 度数で指定
            "properties": ["contrast"],
        }

        extractor = GLCMTextureExtractor(config=config)
        features = extractor.extract(image)

        # 距離3つ × 角度2つ × プロパティ1つ = 6つの特徴量
        expected_count = 3 * 2 * 1
        assert len(features) == expected_count

        # 特徴量名の形式を確認
        for feature_name in features.keys():
            assert "_" in feature_name
            parts = feature_name.split("_")
            assert len(parts) == 3  # property_distance_angle

    def test_extract_different_levels(self):
        """異なるグレーレベル数での特徴量抽出をテスト."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        # 64レベルでテスト
        config = {"levels": 64}
        extractor = GLCMTextureExtractor(config=config)
        features = extractor.extract(image)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_empty_image_error(self):
        """空の画像でのエラーハンドリングをテスト."""
        extractor = GLCMTextureExtractor()

        # 空の画像
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image is empty or None"
        ):
            extractor.extract(np.array([]))

        # None画像
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image is empty or None"
        ):
            extractor.extract(None)

    def test_extract_invalid_shape_error(self):
        """無効な形状の画像でのエラーハンドリングをテスト."""
        extractor = GLCMTextureExtractor()

        # 1次元画像
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image must be 2D or 3D"
        ):
            extractor.extract(np.array([1, 2, 3]))

        # 4次元画像
        with pytest.raises(
            (ValueError, ExtractorValidationError), match="Input image must be 2D or 3D"
        ):
            extractor.extract(np.random.randint(0, 256, (10, 10, 3, 3)))

    def test_get_default_config(self):
        """デフォルト設定の取得をテスト."""
        config = GLCMTextureExtractor.get_default_config()

        assert isinstance(config, dict)
        assert "distances" in config
        assert "angles" in config
        assert "levels" in config
        assert "symmetric" in config
        assert "normed" in config
        assert "properties" in config

        # デフォルト値の確認
        # assert config["distances"] == [1, 2, 3]
        # assert config["angles"] == [0, 45, 90, 135]
        # assert config["levels"] == 256
        # assert config["symmetric"] is True
        # assert config["normed"] is True
        # assert len(config["properties"]) == 6

    def test_get_feature_names(self):
        """特徴量名リストの取得をテスト."""
        feature_names = GLCMTextureExtractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

        # デフォルト設定での期待される特徴量数を計算
        # standard = 4方向, distances = 3, properties = 6
        expected_count = 6 * 3 * 4
        assert len(feature_names) == expected_count

        # 特徴量名の形式を確認（単位付き）
        for name in feature_names:
            assert (
                "[" in name and "]" in name
            ), f"単位付き特徴量名に[]が含まれていません: {name}"
            # 単位部分を抽出
            if "[" in name and "]" in name:
                base_part = name.split("[")[0]
                unit_part = name.split("[")[1].split("]")[0]

                # 基本部分の形式確認
                parts = base_part.split("_")
                assert (
                    len(parts) == 3
                ), f"基本特徴量名の形式が正しくありません: {base_part}"
                prop, distance, angle = parts
                assert prop in [
                    "contrast",
                    "dissimilarity",
                    "homogeneity",
                    "energy",
                    "correlation",
                    "ASM",
                ]
                assert distance in ["1", "2", "3"]
                assert angle in ["0", "45", "90", "135"]

                # 単位の確認
                expected_units = [
                    "intensity_squared",
                    "intensity",
                    "ratio",
                    "correlation_coefficient",
                ]
                assert unit_part in expected_units, f"予期しない単位: {unit_part}"

        # 単位辞書の確認
        units = GLCMTextureExtractor.get_feature_units()
        print(f"特徴量単位辞書のサイズ: {len(units)}")
        print(f"特徴量単位辞書の例: {dict(list(units.items())[:5])}")

        # 基本特徴量名と単位辞書のキーが一致することを確認
        base_names = GLCMTextureExtractor.get_base_feature_names()
        assert set(base_names) == set(
            units.keys()
        ), "基本特徴量名と単位辞書のキーが一致しません"

        # 各プロパティの単位が正しいことを確認
        expected_property_units = {
            "contrast": "intensity_squared",
            "dissimilarity": "intensity",
            "homogeneity": "ratio",
            "energy": "ratio",
            "correlation": "correlation_coefficient",
            "ASM": "ratio",
        }

        for name in feature_names:
            prop = name.split("_")[0]
            expected_unit = expected_property_units[prop]
            assert name.endswith(f"[{expected_unit}]"), (
                f"プロパティ {prop} の単位が正しくありません: "
                f"期待値={expected_unit}, 実際={name[-len(expected_unit):]}"
            )

        # 抽出結果と特徴量名の整合性確認
        extractor = GLCMTextureExtractor()
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        features = extractor.extract(test_image)

        # 抽出された特徴量のキーが基本特徴量名と一致することを確認
        feature_keys = set(features.keys())
        base_names_set = set(base_names)
        print(f"抽出された特徴量のキー数: {len(feature_keys)}")
        print(f"基本特徴量名の数: {len(base_names_set)}")
        assert (
            feature_keys == base_names_set
        ), "抽出された特徴量のキーと基本特徴量名が一致しません"

        print("GLCM特徴量名・単位テスト: 成功")

    def test_feature_name_consistency(self):
        """特徴量名の一貫性をテスト."""
        extractor = GLCMTextureExtractor()

        # 実際の抽出で得られる特徴量名
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        features = extractor.extract(image)
        actual_names = set(features.keys())

        # get_base_feature_names()で得られる特徴量名（基本特徴量名）
        expected_names = set(GLCMTextureExtractor.get_base_feature_names())

        # 両者が一致することを確認
        assert actual_names == expected_names

    def test_reproducibility(self):
        """再現性をテスト."""
        # 同じ画像で複数回実行して同じ結果が得られることを確認
        np.random.seed(42)
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        extractor = GLCMTextureExtractor()

        features1 = extractor.extract(image)
        features2 = extractor.extract(image)

        # 同じ結果が得られることを確認
        assert features1.keys() == features2.keys()
        for key in features1.keys():
            assert np.isclose(features1[key], features2[key])

    def test_different_image_types(self):
        """異なる画像タイプでの動作をテスト."""
        extractor = GLCMTextureExtractor()

        # uint8画像
        image_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        features_uint8 = extractor.extract(image_uint8)
        assert len(features_uint8) > 0

        # float画像（0-1範囲）
        image_float = np.random.random((50, 50, 3)).astype(np.float32)
        features_float = extractor.extract(image_float)
        assert len(features_float) > 0

        # int画像
        image_int = np.random.randint(0, 256, (50, 50, 3), dtype=np.int32)
        features_int = extractor.extract(image_int)
        assert len(features_int) > 0

    def test_feature_names_and_units(self):
        """特徴量名と単位の包括的なテスト."""
        print("\n=== GLCM特徴量名・単位テスト ===")

        # 基本特徴量名の確認
        base_names = GLCMTextureExtractor.get_base_feature_names()
        print(f"基本特徴量名の数: {len(base_names)}")
        print(f"基本特徴量名の例: {base_names[:5]}")

        # 基本特徴量名の形式確認
        for name in base_names:
            parts = name.split("_")
            assert len(parts) == 3, f"基本特徴量名の形式が正しくありません: {name}"
            prop, distance, angle = parts
            assert prop in [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
                "ASM",
            ]
            assert distance in ["1", "2", "3"]
            assert angle in ["0", "45", "90", "135"]

        # 単位付き特徴量名の確認
        unit_names = GLCMTextureExtractor.get_feature_names()
        print(f"単位付き特徴量名の数: {len(unit_names)}")
        print(f"単位付き特徴量名の例: {unit_names[:5]}")

        # 単位付き特徴量名の形式確認
        for name in unit_names:
            assert (
                "[" in name and "]" in name
            ), f"単位付き特徴量名に[]が含まれていません: {name}"
            # 単位部分を抽出
            if "[" in name and "]" in name:
                base_part = name.split("[")[0]
                unit_part = name.split("[")[1].split("]")[0]

                # 基本部分の形式確認
                parts = base_part.split("_")
                assert (
                    len(parts) == 3
                ), f"基本特徴量名の形式が正しくありません: {base_part}"
                prop, distance, angle = parts
                assert prop in [
                    "contrast",
                    "dissimilarity",
                    "homogeneity",
                    "energy",
                    "correlation",
                    "ASM",
                ]
                assert distance in ["1", "2", "3"]
                assert angle in ["0", "45", "90", "135"]

                # 単位の確認
                expected_units = [
                    "intensity_squared",
                    "intensity",
                    "ratio",
                    "correlation_coefficient",
                ]
                assert unit_part in expected_units, f"予期しない単位: {unit_part}"

        # 単位辞書の確認
        units = GLCMTextureExtractor.get_feature_units()
        print(f"特徴量単位辞書のサイズ: {len(units)}")
        print(f"特徴量単位辞書の例: {dict(list(units.items())[:5])}")

        # 基本特徴量名と単位辞書のキーが一致することを確認
        base_names = GLCMTextureExtractor.get_base_feature_names()
        assert set(base_names) == set(
            units.keys()
        ), "基本特徴量名と単位辞書のキーが一致しません"

        # 各プロパティの単位が正しいことを確認
        expected_property_units = {
            "contrast": "intensity_squared",
            "dissimilarity": "intensity",
            "homogeneity": "ratio",
            "energy": "ratio",
            "correlation": "correlation_coefficient",
            "ASM": "ratio",
        }

        for base_name, unit in units.items():
            prop = base_name.split("_")[0]
            expected_unit = expected_property_units[prop]
            assert (
                unit == expected_unit
            ), f"プロパティ {prop} の単位が正しくありません: 期待値={expected_unit}, 実際={unit}"

        # 抽出結果と特徴量名の整合性確認
        extractor = GLCMTextureExtractor()
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        features = extractor.extract(test_image)

        # 抽出された特徴量のキーが基本特徴量名と一致することを確認
        feature_keys = set(features.keys())
        base_names_set = set(base_names)
        print(f"抽出された特徴量のキー数: {len(feature_keys)}")
        print(f"基本特徴量名の数: {len(base_names_set)}")
        assert (
            feature_keys == base_names_set
        ), "抽出された特徴量のキーと基本特徴量名が一致しません"

        print("GLCM特徴量名・単位テスト: 成功")


class TestGLCMBehavior:
    """GLCM 特徴量の振る舞いテスト."""

    _CONFIG = {
        "distances": [1],
        "angles": [0, 45, 90, 135],
        "levels": 256,
        "symmetric": True,
        "normed": True,
        "properties": [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ],
        "resize_shape": None,
    }

    def setup_method(self):
        """テストメソッドごとに extractor を初期化."""
        self.ext = GLCMTextureExtractor(config=self._CONFIG)

    # --- 均一画像 ---

    def test_uniform_contrast_is_zero(self):
        """均一画像の contrast は全方向で 0."""
        f = self.ext.extract(DummyImages.uniform())
        for angle in [0, 45, 90, 135]:
            assert f[f"contrast_1_{angle}"] == 0.0

    def test_uniform_homogeneity_is_one(self):
        """均一画像の homogeneity は全方向で 1.0."""
        f = self.ext.extract(DummyImages.uniform())
        for angle in [0, 45, 90, 135]:
            assert f[f"homogeneity_1_{angle}"] == 1.0

    def test_uniform_energy_is_one(self):
        """均一画像の energy は全方向で 1.0."""
        f = self.ext.extract(DummyImages.uniform())
        for angle in [0, 45, 90, 135]:
            assert f[f"energy_1_{angle}"] == 1.0

    def test_uniform_asm_is_one(self):
        """均一画像の ASM は全方向で 1.0."""
        f = self.ext.extract(DummyImages.uniform())
        for angle in [0, 45, 90, 135]:
            assert f[f"ASM_1_{angle}"] == 1.0

    def test_uniform_dissimilarity_is_zero(self):
        """均一画像の dissimilarity は全方向で 0."""
        f = self.ext.extract(DummyImages.uniform())
        for angle in [0, 45, 90, 135]:
            assert f[f"dissimilarity_1_{angle}"] == 0.0

    # --- チェッカーボード ---

    def test_checker_contrast_high_in_axis_directions(self):
        """チェッカーボードは 0 度と 90 度で contrast が非常に高い (65025)."""
        f = self.ext.extract(DummyImages.checker())
        assert f["contrast_1_0"] > 60000
        assert f["contrast_1_90"] > 60000

    def test_checker_contrast_zero_in_diagonal(self):
        """チェッカーボードは 45 度と 135 度で contrast が 0."""
        f = self.ext.extract(DummyImages.checker())
        assert f["contrast_1_45"] < 1.0
        assert f["contrast_1_135"] < 1.0

    def test_checker_correlation_negative_in_axis(self):
        """チェッカーボードは 0 度と 90 度で correlation が -1 (反相関)."""
        f = self.ext.extract(DummyImages.checker())
        assert f["correlation_1_0"] < -0.9
        assert f["correlation_1_90"] < -0.9

    # --- グラデーション ---

    def test_gradient_horizontal_contrast_zero(self):
        """垂直グラデーションは水平方向 (0 度) の contrast が 0."""
        f = self.ext.extract(DummyImages.gradient())
        assert f["contrast_1_0"] == 0.0

    def test_gradient_horizontal_homogeneity_one(self):
        """垂直グラデーションは水平方向 (0 度) の homogeneity が 1.0."""
        f = self.ext.extract(DummyImages.gradient())
        assert f["homogeneity_1_0"] == 1.0

    def test_gradient_horizontal_correlation_one(self):
        """垂直グラデーションは水平方向 (0 度) の correlation が ~1.0."""
        f = self.ext.extract(DummyImages.gradient())
        assert f["correlation_1_0"] > 0.99

    def test_gradient_vertical_contrast_positive(self):
        """垂直グラデーションは垂直方向 (90 度) の contrast が > 0."""
        f = self.ext.extract(DummyImages.gradient())
        assert f["contrast_1_90"] > 1.0

    # --- ランダム vs 均一 ---

    def test_random_contrast_higher_than_uniform(self):
        """ランダム画像の contrast は均一画像より大きい."""
        f_rand = self.ext.extract(DummyImages.random())
        f_uni = self.ext.extract(DummyImages.uniform())
        assert f_rand["contrast_1_0"] > f_uni["contrast_1_0"]

    def test_random_energy_lower_than_uniform(self):
        """ランダム画像の energy は均一画像より小さい."""
        f_rand = self.ext.extract(DummyImages.random())
        f_uni = self.ext.extract(DummyImages.uniform())
        assert f_rand["energy_1_0"] < f_uni["energy_1_0"]

    def test_random_homogeneity_lower_than_uniform(self):
        """ランダム画像の homogeneity は均一画像より小さい."""
        f_rand = self.ext.extract(DummyImages.random())
        f_uni = self.ext.extract(DummyImages.uniform())
        assert f_rand["homogeneity_1_0"] < f_uni["homogeneity_1_0"]

    # --- ランダム: 閾値テスト ---

    def test_random_energy_near_zero(self):
        """ランダム画像の energy は非常に小さい (< 0.05)."""
        f = self.ext.extract(DummyImages.random())
        assert f["energy_1_0"] < 0.05

    def test_random_contrast_above_1000(self):
        """ランダム画像の contrast は > 1000."""
        f = self.ext.extract(DummyImages.random())
        assert f["contrast_1_0"] > 1000


def test_schema_validation_rejects_invalid_config():
    """スキーマバリデーションが無効な設定を拒否することを確認."""
    from pochivision.feature_extractors import get_feature_extractor

    # levels が範囲外
    with pytest.raises((ValueError, ExtractorValidationError), match="Invalid config"):
        get_feature_extractor("glcm", {"levels": 1})

    # levels が範囲外 (上限)
    with pytest.raises((ValueError, ExtractorValidationError), match="Invalid config"):
        get_feature_extractor("glcm", {"levels": 999})
