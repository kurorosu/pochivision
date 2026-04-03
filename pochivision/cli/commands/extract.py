"""extract サブコマンド: 特徴量抽出."""

import csv
import inspect
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click

from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.exceptions.config import ConfigLoadError
from pochivision.feature_extractors.registry import get_feature_extractor
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
        """
        self.config_path = config_path
        self.config = self._load_config_or_exit(config_path)
        self.input_dir = Path(self.config["input_directory"])
        self.output_manager = output_manager or OutputManager()
        self.output_dir = self.output_manager.create_output_dir("features")
        self.extractors = self._initialize_extractors()

    @staticmethod
    def _load_config_or_exit(config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む. 失敗時はプロセスを終了する.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            設定辞書.
        """
        try:
            return ConfigHandler.load_json(config_path)
        except ConfigLoadError as e:
            raise click.ClickException(str(e))

    def _copy_config_to_output(self) -> None:
        """使用した設定ファイルを出力ディレクトリにコピーする."""
        try:
            config_filename = Path(self.config_path).name
            output_config_path = self.output_dir / config_filename
            shutil.copy2(self.config_path, output_config_path)
            print(f"設定ファイルをコピーしました: {output_config_path}")
        except Exception as e:
            print(f"警告: 設定ファイルのコピーに失敗しました: {e}")

    def _initialize_extractors(self) -> Dict[str, Any]:
        """設定に基づいて特徴量抽出器を初期化する.

        Returns:
            初期化された特徴量抽出器の辞書.
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
                print(f"特徴量抽出器を初期化しました: {name}")
            except ValueError as e:
                print(f"警告: 特徴量抽出器の初期化に失敗しました ({name}): {e}")

        if not extractors:
            raise click.ClickException("使用可能な特徴量抽出器がありません")

        return extractors

    def _get_image_files(self) -> List[Path]:
        """入力ディレクトリから対象画像ファイルを取得する.

        Returns:
            画像ファイルのパスリスト.
        """
        if not self.input_dir.exists():
            raise click.ClickException(
                f"入力ディレクトリが存在しません: {self.input_dir}"
            )

        extensions = self.config.get("file_filters", {}).get(
            "extensions", [".jpg", ".png"]
        )
        case_sensitive = self.config.get("file_filters", {}).get(
            "case_sensitive", False
        )

        image_files = get_image_files(self.input_dir, extensions, case_sensitive)

        if not image_files:
            print(
                f"警告: 入力ディレクトリに対象画像ファイルが見つかりません: {self.input_dir}"
            )
            print(f"対象拡張子: {extensions}")

        return image_files

    def _extract_class_from_filename(self, filename: str) -> str:
        """ファイル名からクラス名を抽出する.

        Args:
            filename: ファイル名 (拡張子なし).

        Returns:
            抽出されたクラス名. 抽出できない場合は空文字列.
        """
        class_config = self.config.get("output_settings", {}).get(
            "class_extraction", {}
        )

        if not class_config.get("enabled", False):
            return ""

        delimiter = class_config.get("delimiter", "_")
        position = class_config.get("position", 0)

        try:
            parts = filename.split(delimiter)

            if 0 <= position < len(parts):
                return str(parts[position])
            elif position < 0 and abs(position) <= len(parts):
                return str(parts[position])
            else:
                print(
                    f"警告: ファイル名 '{filename}' の位置 {position} にクラス名が見つかりません"
                )
                return ""
        except Exception as e:
            print(
                f"警告: ファイル名 '{filename}' からクラス名の抽出に失敗しました: {e}"
            )
            return ""

    def _extract_features_from_image(self, image_path: Path) -> Dict[str, Any]:
        """1つの画像から特徴量を抽出する.

        Args:
            image_path: 画像ファイルのパス.

        Returns:
            抽出された特徴量の辞書.
        """
        try:
            image = load_image(image_path)
            if image is None:
                print(f"警告: 画像の読み込みに失敗しました: {image_path}")
                return {}

            features: Dict[str, Any] = {"filename": image_path.name}

            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False):
                filename_without_ext = image_path.stem
                class_name = self._extract_class_from_filename(filename_without_ext)
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
                        except Exception:
                            for feature_name, value in extracted.items():
                                features[f"{extractor_name}_{feature_name}"] = value
                    else:
                        for feature_name, value in extracted.items():
                            features[f"{extractor_name}_{feature_name}"] = value

                except Exception as e:
                    print(
                        f"警告: 特徴量抽出に失敗しました ({extractor_name}, {image_path}): {e}"
                    )

            return features

        except Exception as e:
            print(f"エラー: 画像処理中にエラーが発生しました ({image_path}): {e}")
            return {}

    def _save_results_to_csv(self, results: List[Dict[str, Any]]) -> None:
        """抽出結果をCSVファイルに保存する.

        Args:
            results: 特徴量抽出結果のリスト.
        """
        if not results:
            print("警告: 保存する結果がありません")
            return

        output_filename = self.config.get("output_settings", {}).get(
            "output_filename", "features.csv"
        )
        output_path = self.output_dir / output_filename

        separator = self.config.get("output_settings", {}).get("csv_separator", ",")

        try:
            all_feature_names: set[str] = set()
            for result in results:
                all_feature_names.update(result.keys())

            headers: list[str] = []
            if "filename" in all_feature_names:
                headers.append("filename")
                all_feature_names.remove("filename")

            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False):
                column_name = class_config.get("column_name", "class")
                if column_name in all_feature_names:
                    headers.append(column_name)
                    all_feature_names.remove(column_name)

            if "timestamp" in all_feature_names:
                headers.append("timestamp")
                all_feature_names.remove("timestamp")
            headers.extend(sorted(all_feature_names))

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(results)

            print(f"横持ちCSV結果を保存しました: {output_path}")
            print(f"処理された画像数: {len(results)}")

        except Exception as e:
            print(f"エラー: CSV保存中にエラーが発生しました: {e}")

    def _save_results_to_long_csv(self, results: List[Dict[str, Any]]) -> None:
        """抽出結果を縦持ちCSVファイルに保存する.

        Args:
            results: 特徴量抽出結果のリスト.
        """
        if not results:
            print("警告: 保存する結果がありません")
            return

        output_filename = self.config.get("output_settings", {}).get(
            "long_format_filename", "features_long.csv"
        )
        output_path = self.output_dir / output_filename

        separator = self.config.get("output_settings", {}).get("csv_separator", ",")

        try:
            long_data: list[dict[str, Any]] = []

            for result in results:
                filename = result.get("filename", "")
                timestamp = result.get("timestamp", "")

                class_config = self.config.get("output_settings", {}).get(
                    "class_extraction", {}
                )
                class_name = ""
                if class_config.get("enabled", False):
                    column_name = class_config.get("column_name", "class")
                    class_name = result.get(column_name, "")

                excluded_columns = ["filename", "timestamp"]
                if class_config.get("enabled", False):
                    excluded_columns.append(class_config.get("column_name", "class"))

                for feature_name, feature_value in result.items():
                    if feature_name not in excluded_columns:
                        row: dict[str, Any] = {
                            "filename": filename,
                            "feature_name": feature_name,
                            "feature_value": feature_value,
                        }

                        if class_name:
                            row[class_config.get("column_name", "class")] = class_name

                        if timestamp:
                            row["timestamp"] = timestamp

                        long_data.append(row)

            headers: list[str] = ["filename"]

            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False) and any(
                row.get(class_config.get("column_name", "class")) for row in long_data
            ):
                headers.append(class_config.get("column_name", "class"))

            headers.extend(["feature_name", "feature_value"])

            if any(row.get("timestamp") for row in long_data):
                headers.insert(-2, "timestamp")

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(long_data)

            print(f"縦持ちCSV結果を保存しました: {output_path}")
            print(f"特徴量レコード数: {len(long_data)}")

        except Exception as e:
            print(f"エラー: 縦持ちCSV保存中にエラーが発生しました: {e}")

    def run(self) -> None:
        """特徴量抽出を実行する."""
        print("=== 特徴量抽出を開始します ===")
        print(f"入力ディレクトリ: {self.input_dir}")
        print(f"出力ディレクトリ: {self.output_dir}")

        self._copy_config_to_output()

        image_files = self._get_image_files()
        if not image_files:
            return

        print(f"対象画像ファイル数: {len(image_files)}")

        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"処理中 ({i}/{len(image_files)}): {image_path.name}")
            features = self._extract_features_from_image(image_path)
            if features:
                results.append(features)

        if self.config.get("output_format", "csv") == "csv":
            self._save_results_to_csv(results)

            if self.config.get("output_settings", {}).get("enable_long_format", False):
                self._save_results_to_long_csv(results)
        elif self.config.get("output_format", "csv") == "long_csv":
            self._save_results_to_long_csv(results)

        print("=== 特徴量抽出が完了しました ===")


@click.command()
@click.option(
    "--config", "-c", type=str, default="extractor_config.json", help="設定ファイルパス"
)
@click.pass_context
def extract(ctx: click.Context, config: str) -> None:
    """画像から特徴量を抽出する."""
    output_manager = ctx.obj.get("output_manager") if ctx.obj else None
    runner = FeatureExtractionRunner(config, output_manager)
    runner.run()
