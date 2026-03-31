"""
特徴量抽出のエントリーポイントファイル.

指定されたフォルダ内のすべての画像に対して特徴量を抽出し、
結果をCSVファイルに保存します。
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2

from pochivision.feature_extractors.registry import get_feature_extractor


class FeatureExtractionRunner:
    """特徴量抽出の実行クラス."""

    def __init__(self, config_path: str) -> None:
        """
        FeatureExtractionRunnerのコンストラクタ.

        Args:
            config_path (str): 設定ファイルのパス.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.input_dir = Path(self.config["input_directory"])
        self.base_output_dir = Path(self.config["output_directory"])
        self.output_dir = self._create_timestamped_output_dir()
        self.extractors = self._initialize_extractors()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む.

        Args:
            config_path (str): 設定ファイルのパス.

        Returns:
            Dict[str, Any]: 設定辞書.

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合.
            json.JSONDecodeError: JSON形式が不正な場合.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except FileNotFoundError:
            print(f"エラー: 設定ファイルが見つかりません: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"エラー: 設定ファイルのJSON形式が不正です: {e}")
            sys.exit(1)

    def _get_next_suffix(self, base_dir: Path, date_str: str) -> int:
        """
        指定された日付の次のサフィックス番号を取得します.

        同じ日付のディレクトリが存在する場合、最大のサフィックス番号に1を加えた値を返します.

        Args:
            base_dir (Path): 出力ベースディレクトリのパス.
            date_str (str): 日付文字列(YYYYMMDD形式).

        Returns:
            int: 次のサフィックス番号.
        """
        if not base_dir.exists():
            return 0

        max_suffix = -1
        for dir_path in base_dir.iterdir():
            if not dir_path.is_dir():
                continue

            # YYYYMMDD_N 形式のディレクトリを検索
            if dir_path.name.startswith(date_str + "_"):
                try:
                    suffix = int(dir_path.name.split("_")[-1])
                    max_suffix = max(max_suffix, suffix)
                except ValueError:
                    continue

        return max_suffix + 1

    def _create_timestamped_output_dir(self) -> Path:
        """
        日付とインクリメントを含む出力ディレクトリを作成します.

        同じ日付のディレクトリが存在する場合、サフィックスをインクリメントします.

        Returns:
            Path: 作成されたディレクトリのパス. 例：extraction_results/YYYYMMDD_{suffix}/
        """
        date_str = datetime.now().strftime("%Y%m%d")

        # ベース出力ディレクトリが存在しない場合は作成
        if not self.base_output_dir.exists():
            self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # 次のサフィックス番号を取得
        suffix = self._get_next_suffix(self.base_output_dir, date_str)

        # 日付ディレクトリ作成
        final_path = self.base_output_dir / f"{date_str}_{suffix}"
        final_path.mkdir(parents=True, exist_ok=True)
        return final_path

    def _copy_config_to_output(self) -> None:
        """使用した設定ファイルを出力ディレクトリにコピーします."""
        try:
            config_filename = Path(self.config_path).name
            output_config_path = self.output_dir / config_filename
            shutil.copy2(self.config_path, output_config_path)
            print(f"設定ファイルをコピーしました: {output_config_path}")
        except Exception as e:
            print(f"警告: 設定ファイルのコピーに失敗しました: {e}")

    def _initialize_extractors(self) -> Dict[str, Any]:
        """
        設定に基づいて特徴量抽出器を初期化.

        extractors リストが指定されている場合, そのリストの抽出器のみ初期化する.
        未指定の場合は feature_extractors の全エントリを使用 (後方互換).

        Returns:
            Dict[str, Any]: 初期化された特徴量抽出器の辞書.
        """
        extractors = {}
        extractor_configs = self.config.get("feature_extractors", {})
        extractor_list = self.config.get("extractors")

        # extractors リストが指定されていれば, そのリストの抽出器のみ初期化
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
            print("エラー: 使用可能な特徴量抽出器がありません")
            sys.exit(1)

        return extractors

    def _get_image_files(self) -> List[Path]:
        """
        入力ディレクトリから対象画像ファイルを取得.

        Returns:
            List[Path]: 画像ファイルのパスリスト.
        """
        if not self.input_dir.exists():
            print(f"エラー: 入力ディレクトリが存在しません: {self.input_dir}")
            sys.exit(1)

        extensions = self.config.get("file_filters", {}).get(
            "extensions", [".jpg", ".png"]
        )
        case_sensitive = self.config.get("file_filters", {}).get(
            "case_sensitive", False
        )

        image_files: List[Path] = []
        for ext in extensions:
            if case_sensitive:
                pattern = f"*{ext}"
            else:
                # 大文字小文字両方のパターンで検索
                pattern_lower = f"*{ext.lower()}"
                pattern_upper = f"*{ext.upper()}"
                image_files.extend(self.input_dir.glob(pattern_lower))
                image_files.extend(self.input_dir.glob(pattern_upper))
                continue

            image_files.extend(self.input_dir.glob(pattern))

        # 重複を除去
        image_files = list(set(image_files))
        image_files.sort()  # ソートして処理順序を一定にする

        if not image_files:
            print(
                f"警告: 入力ディレクトリに対象画像ファイルが見つかりません: {self.input_dir}"
            )
            print(f"対象拡張子: {extensions}")

        return image_files

    def _extract_class_from_filename(self, filename: str) -> str:
        """
        ファイル名からクラス名を抽出.

        Args:
            filename (str): ファイル名（拡張子なし）.

        Returns:
            str: 抽出されたクラス名. 抽出できない場合は空文字列.
        """
        class_config = self.config.get("output_settings", {}).get(
            "class_extraction", {}
        )

        if not class_config.get("enabled", False):
            return ""

        delimiter = class_config.get("delimiter", "_")
        position = class_config.get("position", 0)

        try:
            # ファイル名を区切り文字で分割
            parts = filename.split(delimiter)

            # 指定された位置の文字列を取得
            if 0 <= position < len(parts):
                return str(parts[position])
            elif position < 0 and abs(position) <= len(parts):
                # 負の値の場合は後ろから数える
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
        """
        1つの画像から特徴量を抽出.

        Args:
            image_path (Path): 画像ファイルのパス.

        Returns:
            Dict[str, Any]: 抽出された特徴量の辞書.
        """
        try:
            # 画像の読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 画像の読み込みに失敗しました: {image_path}")
                return {}

            # 各特徴量抽出器で処理
            features = {"filename": image_path.name}

            # クラス名の抽出（設定で有効な場合）
            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False):
                filename_without_ext = image_path.stem  # 拡張子なしのファイル名
                class_name = self._extract_class_from_filename(filename_without_ext)
                if class_name:
                    column_name = class_config.get("column_name", "class")
                    features[column_name] = class_name

            # タイムスタンプの追加（設定で有効な場合）
            if self.config.get("output_settings", {}).get("include_timestamp", False):
                features["timestamp"] = datetime.now().isoformat()

            for extractor_name, extractor in self.extractors.items():
                try:
                    extracted = extractor.extract(image)

                    # 抽出器が単位管理機能を持っているかチェック
                    has_unit_support = hasattr(
                        extractor.__class__, "get_base_feature_names"
                    ) and hasattr(extractor.__class__, "get_feature_units")

                    if has_unit_support:
                        # 単位付きの特徴量名を使用
                        try:
                            # 抽出器の設定を取得して単位管理機能に渡す
                            extractor_config = getattr(extractor, "config", None)

                            # get_feature_unitsメソッドが設定パラメータを受け取るかチェック
                            import inspect

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
                                    # 単位が見つからない場合は基本名を使用
                                    features[f"{extractor_name}_{base_name}"] = value
                        except Exception:
                            # 単位機能でエラーが発生した場合は基本名を使用
                            for feature_name, value in extracted.items():
                                features[f"{extractor_name}_{feature_name}"] = value
                    else:
                        # 単位管理機能がない場合は基本特徴量名を使用
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
        """
        抽出結果をCSVファイルに保存.

        Args:
            results (List[Dict[str, Any]]): 特徴量抽出結果のリスト.
        """
        if not results:
            print("警告: 保存する結果がありません")
            return

        # 出力ファイル名の決定
        output_filename = self.config.get("output_settings", {}).get(
            "output_filename", "features.csv"
        )
        output_path = self.output_dir / output_filename

        # CSV区切り文字の取得
        separator = self.config.get("output_settings", {}).get("csv_separator", ",")

        try:
            # すべての特徴量名を収集（ヘッダー用）
            all_feature_names: set[str] = set()
            for result in results:
                all_feature_names.update(result.keys())

            # ヘッダーの順序を決定（filename, class, timestamp, その他の順）
            headers = []
            if "filename" in all_feature_names:
                headers.append("filename")
                all_feature_names.remove("filename")

            # クラス名列を追加（設定で有効な場合）
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

            # CSVファイルに書き込み
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(results)

            print(f"横持ちCSV結果を保存しました: {output_path}")
            print(f"処理された画像数: {len(results)}")

        except Exception as e:
            print(f"エラー: CSV保存中にエラーが発生しました: {e}")

    def _save_results_to_long_csv(self, results: List[Dict[str, Any]]) -> None:
        """
        抽出結果を縦持ち（long format）CSVファイルに保存.

        Args:
            results (List[Dict[str, Any]]): 特徴量抽出結果のリスト.
        """
        if not results:
            print("警告: 保存する結果がありません")
            return

        # 出力ファイル名の決定
        output_filename = self.config.get("output_settings", {}).get(
            "long_format_filename", "features_long.csv"
        )
        output_path = self.output_dir / output_filename

        # CSV区切り文字の取得
        separator = self.config.get("output_settings", {}).get("csv_separator", ",")

        try:
            # 縦持ち形式のデータを作成
            long_data = []

            for result in results:
                filename = result.get("filename", "")
                timestamp = result.get("timestamp", "")

                # クラス名の取得（設定で有効な場合）
                class_config = self.config.get("output_settings", {}).get(
                    "class_extraction", {}
                )
                class_name = ""
                if class_config.get("enabled", False):
                    column_name = class_config.get("column_name", "class")
                    class_name = result.get(column_name, "")

                # filename, class, timestamp 以外の特徴量を縦持ちに変換
                excluded_columns = ["filename", "timestamp"]
                if class_config.get("enabled", False):
                    excluded_columns.append(class_config.get("column_name", "class"))

                for feature_name, feature_value in result.items():
                    if feature_name not in excluded_columns:
                        row = {
                            "filename": filename,
                            "feature_name": feature_name,
                            "feature_value": feature_value,
                        }

                        # クラス名が有効な場合は追加
                        if class_name:
                            row[class_config.get("column_name", "class")] = class_name

                        # タイムスタンプが有効な場合は追加
                        if timestamp:
                            row["timestamp"] = timestamp

                        long_data.append(row)

            # ヘッダーの決定
            headers = ["filename"]

            # クラス名列を追加（設定で有効な場合）
            class_config = self.config.get("output_settings", {}).get(
                "class_extraction", {}
            )
            if class_config.get("enabled", False) and any(
                row.get(class_config.get("column_name", "class")) for row in long_data
            ):
                headers.append(class_config.get("column_name", "class"))

            headers.extend(["feature_name", "feature_value"])

            if any(row.get("timestamp") for row in long_data):
                # feature_name, feature_valueの前にtimestampを挿入
                headers.insert(-2, "timestamp")

            # CSVファイルに書き込み
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

        # 設定ファイルのコピー
        self._copy_config_to_output()

        # 対象画像ファイルの取得
        image_files = self._get_image_files()
        if not image_files:
            return

        print(f"対象画像ファイル数: {len(image_files)}")

        # 各画像から特徴量を抽出
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"処理中 ({i}/{len(image_files)}): {image_path.name}")
            features = self._extract_features_from_image(image_path)
            if features:
                results.append(features)

        # 結果の保存
        if self.config.get("output_format", "csv") == "csv":
            self._save_results_to_csv(results)

            # 縦持ち形式も出力する場合
            if self.config.get("output_settings", {}).get("enable_long_format", False):
                self._save_results_to_long_csv(results)
        elif self.config.get("output_format", "csv") == "long_csv":
            self._save_results_to_long_csv(results)

        print("=== 特徴量抽出が完了しました ===")


def main(config_path: str | None = None) -> None:
    """
    メイン関数.

    Args:
        config_path: 設定ファイルのパス. None の場合は argparse から取得.
    """
    if config_path is None:
        parser = argparse.ArgumentParser(description="画像から特徴量を抽出します")
        parser.add_argument(
            "--config",
            default="extractor_config.json",
            help="設定ファイルのパス (デフォルト: extractor_config.json)",
        )
        args = parser.parse_args()
        config_path = args.config

    runner = FeatureExtractionRunner(config_path)
    runner.run()


if __name__ == "__main__":
    main()
