"""特徴量抽出結果の CSV 出力を担当するモジュール."""

import csv
from pathlib import Path
from typing import Any

from pochivision.capturelib.log_manager import LogManager


class FeatureCSVWriter:
    """特徴量抽出結果を CSV ファイルに出力するクラス.

    Attributes:
        output_dir: 出力先ディレクトリ.
    """

    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        """FeatureCSVWriterのコンストラクタ.

        Args:
            output_dir: CSV ファイルの出力先ディレクトリ.
            config: 出力設定を含む設定辞書.
        """
        self.output_dir = output_dir
        self.config = config
        self.logger = LogManager().get_logger()

    def _get_output_settings(self) -> Any:
        """出力設定を取得する.

        Returns:
            出力設定の辞書.
        """
        return self.config.get("output_settings", {})

    def _get_class_config(self) -> Any:
        """クラス抽出設定を取得する.

        Returns:
            クラス抽出設定の辞書.
        """
        return self._get_output_settings().get("class_extraction", {})

    def save_wide_csv(self, results: list[dict[str, Any]]) -> None:
        """抽出結果を横持ち CSV ファイルに保存する.

        Args:
            results: 特徴量抽出結果のリスト.
        """
        if not results:
            self.logger.warning("保存する結果がありません")
            return

        output_settings = self._get_output_settings()
        output_filename = output_settings.get("output_filename", "features.csv")
        output_path = self.output_dir / output_filename
        separator = output_settings.get("csv_separator", ",")

        try:
            headers = self._build_wide_headers(results)

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(results)

            self.logger.info(f"横持ちCSV結果を保存しました: {output_path}")
            self.logger.info(f"処理された画像数: {len(results)}")

        except Exception as e:
            self.logger.error(f"CSV保存中にエラーが発生しました: {e}")

    def _build_wide_headers(self, results: list[dict[str, Any]]) -> list[str]:
        """横持ち CSV のヘッダーを構築する.

        Args:
            results: 特徴量抽出結果のリスト.

        Returns:
            ヘッダーのリスト.
        """
        all_feature_names: set[str] = set()
        for result in results:
            all_feature_names.update(result.keys())

        headers: list[str] = []
        if "filename" in all_feature_names:
            headers.append("filename")
            all_feature_names.remove("filename")

        class_config = self._get_class_config()
        if class_config.get("enabled", False):
            column_name = class_config.get("column_name", "class")
            if column_name in all_feature_names:
                headers.append(column_name)
                all_feature_names.remove(column_name)

        if "timestamp" in all_feature_names:
            headers.append("timestamp")
            all_feature_names.remove("timestamp")
        headers.extend(sorted(all_feature_names))

        return headers

    def save_long_csv(self, results: list[dict[str, Any]]) -> None:
        """抽出結果を縦持ち CSV ファイルに保存する.

        Args:
            results: 特徴量抽出結果のリスト.
        """
        if not results:
            self.logger.warning("保存する結果がありません")
            return

        output_settings = self._get_output_settings()
        output_filename = output_settings.get(
            "long_format_filename", "features_long.csv"
        )
        output_path = self.output_dir / output_filename
        separator = output_settings.get("csv_separator", ",")

        try:
            long_data = self._build_long_data(results)
            headers = self._build_long_headers(long_data)

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(long_data)

            self.logger.info(f"縦持ちCSV結果を保存しました: {output_path}")
            self.logger.info(f"特徴量レコード数: {len(long_data)}")

        except Exception as e:
            self.logger.error(f"縦持ちCSV保存中にエラーが発生しました: {e}")

    def _build_long_data(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """縦持ち形式のデータを構築する.

        Args:
            results: 特徴量抽出結果のリスト.

        Returns:
            縦持ち形式のデータリスト.
        """
        class_config = self._get_class_config()
        long_data: list[dict[str, Any]] = []

        for result in results:
            filename = result.get("filename", "")
            timestamp = result.get("timestamp", "")

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

        return long_data

    def _build_long_headers(self, long_data: list[dict[str, Any]]) -> list[str]:
        """縦持ち CSV のヘッダーを構築する.

        Args:
            long_data: 縦持ち形式のデータリスト.

        Returns:
            ヘッダーのリスト.
        """
        class_config = self._get_class_config()
        headers: list[str] = ["filename"]

        if class_config.get("enabled", False) and any(
            row.get(class_config.get("column_name", "class")) for row in long_data
        ):
            headers.append(class_config.get("column_name", "class"))

        headers.extend(["feature_name", "feature_value"])

        if any(row.get("timestamp") for row in long_data):
            headers.insert(-2, "timestamp")

        return headers
