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

from feature_extractors.registry import get_feature_extractor


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
                return json.load(f)
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

        Returns:
            Dict[str, Any]: 初期化された特徴量抽出器の辞書.
        """
        extractors = {}
        extractor_configs = self.config.get("feature_extractors", {})

        for name, config in extractor_configs.items():
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

            # タイムスタンプの追加（設定で有効な場合）
            if self.config.get("output_settings", {}).get("include_timestamp", False):
                features["timestamp"] = datetime.now().isoformat()

            for extractor_name, extractor in self.extractors.items():
                try:
                    extracted = extractor.extract(image)
                    # 特徴量名に抽出器名をプレフィックスとして追加
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

            # ヘッダーの順序を決定（filename, timestamp, その他の順）
            headers = []
            if "filename" in all_feature_names:
                headers.append("filename")
                all_feature_names.remove("filename")
            if "timestamp" in all_feature_names:
                headers.append("timestamp")
                all_feature_names.remove("timestamp")
            headers.extend(sorted(all_feature_names))

            # CSVファイルに書き込み
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, delimiter=separator)
                writer.writeheader()
                writer.writerows(results)

            print(f"特徴量抽出結果を保存しました: {output_path}")
            print(f"処理された画像数: {len(results)}")

        except Exception as e:
            print(f"エラー: CSV保存中にエラーが発生しました: {e}")

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

        print("=== 特徴量抽出が完了しました ===")


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(description="画像から特徴量を抽出します")
    parser.add_argument(
        "--config",
        default="extractor_config.json",
        help="設定ファイルのパス (デフォルト: extractor_config.json)",
    )

    args = parser.parse_args()

    # 特徴量抽出の実行
    runner = FeatureExtractionRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
