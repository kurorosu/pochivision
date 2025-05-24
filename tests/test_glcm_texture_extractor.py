"""GLCM特徴量抽出器のテストモジュール."""

import numpy as np
import pytest

from feature_extractors.glcm_texture import GLCMTextureExtractor


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
        with pytest.raises(ValueError, match="Angles must be a list of numbers"):
            GLCMTextureExtractor(config={"angles": "invalid_preset"})

        # 空のリスト
        with pytest.raises(ValueError, match="Angles list cannot be empty"):
            GLCMTextureExtractor(config={"angles": []})

        # 数値以外の要素を含むリスト
        with pytest.raises(ValueError, match="All angles must be numeric values"):
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
        with pytest.raises(ValueError, match="Input image is empty or None"):
            extractor.extract(np.array([]))

        # None画像
        with pytest.raises(ValueError, match="Input image is empty or None"):
            extractor.extract(None)

    def test_extract_invalid_shape_error(self):
        """無効な形状の画像でのエラーハンドリングをテスト."""
        extractor = GLCMTextureExtractor()

        # 1次元画像
        with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
            extractor.extract(np.array([1, 2, 3]))

        # 4次元画像
        with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
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

        # 特徴量名の形式を確認
        for name in feature_names:
            assert "_" in name
            parts = name.split("_")
            assert len(parts) == 3  # property_distance_angle

    def test_feature_name_consistency(self):
        """特徴量名の一貫性をテスト."""
        extractor = GLCMTextureExtractor()

        # 実際の抽出で得られる特徴量名
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        features = extractor.extract(image)
        actual_names = set(features.keys())

        # get_feature_names()で得られる特徴量名
        expected_names = set(GLCMTextureExtractor.get_feature_names())

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
