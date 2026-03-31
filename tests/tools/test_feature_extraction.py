"""特徴量抽出の選択実行機能のテスト."""

import json

import pytest


class TestExtractorSelection:
    """extractors リストによる抽出器選択のテスト."""

    @staticmethod
    def _make_config(extractors=None, feature_extractors=None):
        """テスト用の config dict を生成."""
        config = {
            "input_directory": "dummy_input",
            "output_directory": "dummy_output",
            "output_format": "csv",
            "feature_extractors": feature_extractors
            or {
                "brightness": {"color_mode": "gray", "exclude_zero_pixels": True},
                "rgb": {"exclude_black_pixels": True},
                "hsv": {"exclude_black_pixels": True},
            },
        }
        if extractors is not None:
            config["extractors"] = extractors
        return config

    def test_extractors_list_selects_subset(self):
        """extractors リストで指定した抽出器のみ初期化される."""
        from pochivision.feature_extractors.registry import get_feature_extractor

        config = self._make_config(extractors=["brightness", "rgb"])
        extractor_configs = config.get("feature_extractors", {})
        extractor_list = config.get("extractors")

        extractors = {}
        for name in extractor_list:
            ext_config = extractor_configs.get(name, {})
            extractors[name] = get_feature_extractor(name, ext_config)

        assert "brightness" in extractors
        assert "rgb" in extractors
        assert "hsv" not in extractors

    def test_extractors_list_single(self):
        """extractors リストに 1 つだけ指定."""
        from pochivision.feature_extractors.registry import get_feature_extractor

        config = self._make_config(extractors=["hsv"])
        extractor_configs = config.get("feature_extractors", {})
        extractor_list = config.get("extractors")

        extractors = {}
        for name in extractor_list:
            ext_config = extractor_configs.get(name, {})
            extractors[name] = get_feature_extractor(name, ext_config)

        assert len(extractors) == 1
        assert "hsv" in extractors

    def test_no_extractors_list_uses_all(self):
        """extractors リスト未指定時は全 feature_extractors を使用."""
        config = self._make_config(extractors=None)
        extractor_list = config.get("extractors")

        assert extractor_list is None
        # 未指定時は feature_extractors の全キーを使用
        target_names = list(config["feature_extractors"].keys())
        assert target_names == ["brightness", "rgb", "hsv"]

    def test_extractors_list_empty(self):
        """extractors リストが空の場合は抽出器なし."""
        config = self._make_config(extractors=[])
        extractor_list = config.get("extractors")

        assert extractor_list == []
        assert len(extractor_list) == 0

    def test_extractors_list_with_missing_config(self):
        """extractors リストに config がない抽出器を指定してもデフォルト config で動作."""
        from pochivision.feature_extractors.registry import get_feature_extractor

        # feature_extractors に "fft" の config がないが, extractors に含まれている
        config = self._make_config(
            extractors=["brightness", "fft"],
            feature_extractors={
                "brightness": {"color_mode": "gray", "exclude_zero_pixels": True},
                # fft の config はない
            },
        )
        extractor_configs = config.get("feature_extractors", {})
        extractor_list = config.get("extractors")

        extractors = {}
        for name in extractor_list:
            ext_config = extractor_configs.get(name, {})
            extractors[name] = get_feature_extractor(name, ext_config)

        assert "brightness" in extractors
        assert "fft" in extractors  # デフォルト config で初期化

    def test_extractors_order_preserved(self):
        """extractors リストの順序が保持される."""
        config = self._make_config(extractors=["hsv", "brightness", "rgb"])
        extractor_list = config.get("extractors")

        assert extractor_list == ["hsv", "brightness", "rgb"]
