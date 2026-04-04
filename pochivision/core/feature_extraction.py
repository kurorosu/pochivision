"""特徴量抽出のビジネスロジッククラス."""

import inspect
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.capturelib.log_manager import LogManager
from pochivision.core.feature_csv_writer import FeatureCSVWriter
from pochivision.feature_extractors.registry import get_feature_extractor
from pochivision.utils.class_extraction import extract_class_from_filename
from pochivision.utils.image import get_image_files, load_image
from pochivision.workspace import OutputManager


class FeatureExtractionRunner:
    """特徴量抽出の実行クラス."""

    def __init__(
        self,
        config_path: str,
        output_manager: OutputManager | None = None,
    ) -> None:
        """FeatureExtractionRunnerのコンストラクタ.

        Args:
            config_path: 設定ファイルのパス.
            output_manager: 出力ディレクトリの統一管理クラス.
                None の場合はデフォルトの OutputManager を使用.

        Raises:
            ValueError: 使用可能な特徴量抽出器がない場合.
        """
        self.logger = LogManager().get_logger()
        self.config_path = config_path
        self.config = ConfigHandler.load_json(config_path)
        self.input_dir = Path(self.config["input_directory"])
        self.output_manager = output_manager or OutputManager()
        self.output_dir = self.output_manager.create_output_dir("features")
        self.extractors = self._initialize_extractors()
        self.csv_writer = FeatureCSVWriter(self.output_dir, self.config)

    def _copy_config_to_output(self) -> None:
        """使用した設定ファイルを出力ディレクトリにコピーする."""
        try:
            config_filename = Path(self.config_path).name
            output_config_path = self.output_dir / config_filename
            shutil.copy2(self.config_path, output_config_path)
            self.logger.info(f"設定ファイルをコピーしました: {output_config_path}")
        except Exception as e:
            self.logger.warning(f"設定ファイルのコピーに失敗しました: {e}")

    def _initialize_extractors(self) -> dict[str, Any]:
        """設定に基づいて特徴量抽出器を初期化する.

        Returns:
            初期化された特徴量抽出器の辞書.

        Raises:
            ValueError: 使用可能な特徴量抽出器がない場合.
        """
        extractors = {}
        extractor_configs = self.config.get("feature_extractors", {})
        extractor_list = self.config.get("extractors")

        if extractor_list is not None:
            target_names = extractor_list
        else:
            target_names = list(extractor_configs.keys())

        for name in target_names:
            config = extractor_configs.get(name, {})
            try:
                extractor = get_feature_extractor(name, config)
                extractors[name] = extractor
                self.logger.info(f"特徴量抽出器を初期化しました: {name}")
            except ValueError as e:
                self.logger.warning(f"特徴量抽出器の初期化に失敗しました ({name}): {e}")

        if not extractors:
            raise ValueError("使用可能な特徴量抽出器がありません")

        return extractors

    def _get_image_files(self) -> list[Path]:
        """入力ディレクトリから対象画像ファイルを取得する.

        Returns:
            画像ファイルのパスリスト.

        Raises:
            FileNotFoundError: 入力ディレクトリが存在しない場合.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir}")

        extensions = self.config.get("file_filters", {}).get(
            "extensions", [".jpg", ".png"]
        )
        case_sensitive = self.config.get("file_filters", {}).get(
            "case_sensitive", False
        )

        image_files = get_image_files(self.input_dir, extensions, case_sensitive)

        if not image_files:
            self.logger.warning(
                f"入力ディレクトリに対象画像ファイルが見つかりません: {self.input_dir}"
            )
            self.logger.warning(f"対象拡張子: {extensions}")

        return image_files

    def _extract_features_from_image(self, image_path: Path) -> dict[str, Any]:
        """1つの画像から特徴量を抽出する.

        Args:
            image_path: 画像ファイルのパス.

        Returns:
            抽出された特徴量の辞書.
        """
        try:
            image = load_image(image_path)
            if image is None:
                self.logger.warning(f"画像の読み込みに失敗しました: {image_path}")
                return {}

            features: dict[str, Any] = {"filename": image_path.name}

            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False):
                class_name = extract_class_from_filename(
                    image_path.stem,
                    delimiter=class_config.get("delimiter", "_"),
                    position=class_config.get("position", 0),
                )
                if class_name:
                    column_name = class_config.get("column_name", "class")
                    features[column_name] = class_name

            if self.config.get("output_settings", {}).get("include_timestamp", False):
                features["timestamp"] = datetime.now().isoformat()

            for extractor_name, extractor in self.extractors.items():
                try:
                    extracted = extractor.extract(image)

                    has_unit_support = hasattr(
                        extractor.__class__, "get_base_feature_names"
                    ) and hasattr(extractor.__class__, "get_feature_units")

                    if has_unit_support:
                        try:
                            extractor_config = getattr(extractor, "config", None)

                            sig = inspect.signature(
                                extractor.__class__.get_feature_units
                            )
                            if "config" in sig.parameters:
                                units = extractor.__class__.get_feature_units(
                                    extractor_config
                                )
                            else:
                                units = extractor.__class__.get_feature_units()

                            for base_name, value in extracted.items():
                                if base_name in units:
                                    unit_name = f"{base_name}[{units[base_name]}]"
                                    features[f"{extractor_name}_{unit_name}"] = value
                                else:
                                    features[f"{extractor_name}_{base_name}"] = value
                        except Exception as e:
                            self.logger.warning(
                                f"ユニット名の取得に失敗しました ({extractor_name}): {e}"
                            )
                            for feature_name, value in extracted.items():
                                features[f"{extractor_name}_{feature_name}"] = value
                    else:
                        for feature_name, value in extracted.items():
                            features[f"{extractor_name}_{feature_name}"] = value

                except Exception as e:
                    self.logger.warning(
                        f"特徴量抽出に失敗しました ({extractor_name}, {image_path}): {e}"
                    )

            return features

        except Exception as e:
            self.logger.error(f"画像処理中にエラーが発生しました ({image_path}): {e}")
            return {}

    def run(self) -> None:
        """特徴量抽出を実行する."""
        self.logger.info("=== 特徴量抽出を開始します ===")
        self.logger.info(f"入力ディレクトリ: {self.input_dir}")
        self.logger.info(f"出力ディレクトリ: {self.output_dir}")

        self._copy_config_to_output()

        image_files = self._get_image_files()
        if not image_files:
            return

        self.logger.info(f"対象画像ファイル数: {len(image_files)}")

        results = []
        for i, image_path in enumerate(image_files, 1):
            self.logger.info(f"処理中 ({i}/{len(image_files)}): {image_path.name}")
            features = self._extract_features_from_image(image_path)
            if features:
                results.append(features)

        if self.config.get("output_format", "csv") == "csv":
            self.csv_writer.save_wide_csv(results)

            if self.config.get("output_settings", {}).get("enable_long_format", False):
                self.csv_writer.save_long_csv(results)
        elif self.config.get("output_format", "csv") == "long_csv":
            self.csv_writer.save_long_csv(results)

        self.logger.info("=== 特徴量抽出が完了しました ===")
