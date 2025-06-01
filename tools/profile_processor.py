"""
プロファイルベース画像処理ツール.

既存のconfig.jsonのカメラプロファイル設定を使用して、
フォルダ内の画像に対して指定されたプロファイルの処理を適用します.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from processors.registry import get_processor


class ProfileProcessor:
    """プロファイルベースの画像処理クラス."""

    def __init__(self, config_path: str, profile_name: str) -> None:
        """
        ProfileProcessorのコンストラクタ.

        Args:
            config_path (str): config.jsonファイルのパス.
            profile_name (str): 使用するカメラプロファイル名.
        """
        self.config_path = config_path
        self.profile_name = profile_name
        self.config = self._load_config(config_path)
        self.profile_config = self._get_profile_config(profile_name)
        self.processors = self._initialize_processors()

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

    def _get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """
        指定されたプロファイルの設定を取得.

        Args:
            profile_name (str): プロファイル名.

        Returns:
            Dict[str, Any]: プロファイル設定.

        Raises:
            ValueError: プロファイルが見つからない場合.
        """
        cameras = self.config.get("cameras", {})
        if profile_name not in cameras:
            available_profiles = list(cameras.keys())
            print(f"エラー: プロファイル '{profile_name}' が見つかりません")
            print(f"利用可能なプロファイル: {available_profiles}")
            sys.exit(1)

        return cameras[profile_name]

    def _initialize_processors(self) -> List[Any]:
        """
        プロファイル設定に基づいてプロセッサを初期化.

        Returns:
            List[Any]: 初期化されたプロセッサのリスト.
        """
        processors = []
        processor_names = self.profile_config.get("processors", [])

        for processor_name in processor_names:
            try:
                # プロセッサ固有の設定を取得
                processor_config = self.profile_config.get(processor_name, {})
                processor = get_processor(processor_name, processor_config)
                processors.append(processor)
                print(f"プロセッサを初期化しました: {processor_name}")
            except Exception as e:
                print(f"警告: プロセッサの初期化に失敗しました ({processor_name}): {e}")

        if not processors:
            print("警告: 使用可能なプロセッサがありません")

        return processors

    def _get_image_files(self, input_dir: Path) -> List[Path]:
        """
        入力ディレクトリから画像ファイルを取得.

        Args:
            input_dir (Path): 入力ディレクトリのパス.

        Returns:
            List[Path]: 画像ファイルのパスリスト.
        """
        if not input_dir.exists():
            print(f"エラー: 入力ディレクトリが存在しません: {input_dir}")
            sys.exit(1)

        # 一般的な画像拡張子
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_files: List[Path] = []

        for ext in extensions:
            # 大文字小文字両方のパターンで検索
            image_files.extend(input_dir.glob(f"*{ext.lower()}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        # 重複を除去してソート
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"警告: 入力ディレクトリに画像ファイルが見つかりません: {input_dir}")
            print(f"対象拡張子: {extensions}")

        return image_files

    def _create_output_directory(self, base_output_dir: Path) -> Path:
        """
        タイムスタンプ付きの出力ディレクトリを作成.

        Args:
            base_output_dir (Path): ベース出力ディレクトリ.

        Returns:
            Path: 作成された出力ディレクトリのパス.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"profile_{self.profile_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _process_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        単一画像を処理.

        Args:
            image_path (Path): 画像ファイルのパス.

        Returns:
            Optional[np.ndarray]: 処理された画像、エラーの場合はNone.
        """
        try:
            # 画像を読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 画像の読み込みに失敗しました: {image_path}")
                return None

            # プロファイルのモードを確認
            mode = self.profile_config.get("mode", "pipeline")

            if mode == "pipeline":
                # パイプライン処理: 順次処理
                processed_image = image.copy()
                original_image = image.copy()  # 元画像を保持

                for processor in self.processors:
                    # mask_compositionプロセッサの場合、元画像を設定
                    if hasattr(processor, "set_target_image"):
                        processor.set_target_image(original_image)

                    processed_image = processor.process(processed_image)
                    if processed_image is None:
                        print(f"警告: プロセッサでの処理に失敗しました: {image_path}")
                        return None
                return processed_image

            elif mode == "parallel":
                # パラレル処理: 各プロセッサを独立して適用
                results = {}
                for processor in self.processors:
                    # mask_compositionプロセッサはパラレルモードでは使用できない
                    if hasattr(processor, "set_target_image"):
                        print(
                            f"警告: {processor.__class__.__name__}はパラレルモードでは使用できません"
                        )
                        continue

                    processor_name = processor.__class__.__name__.lower().replace(
                        "processor", ""
                    )
                    result = processor.process(image.copy())
                    if result is not None:
                        results[processor_name] = result

                # パラレル処理の場合は最後のプロセッサの結果を返す
                # または、すべての結果を保存する場合は別途実装
                if results:
                    return list(results.values())[-1]
                return None

            else:
                print(f"警告: 不明な処理モード: {mode}")
                return None

        except Exception as e:
            print(f"エラー: 画像処理中にエラーが発生しました ({image_path}): {e}")
            return None

    def process_directory(
        self, input_dir: str, output_dir: str, save_original: bool = True
    ) -> None:
        """
        ディレクトリ内のすべての画像を処理.

        Args:
            input_dir (str): 入力ディレクトリのパス.
            output_dir (str): 出力ディレクトリのパス.
            save_original (bool): 元画像も保存するかどうか.
        """
        input_path = Path(input_dir)
        base_output_path = Path(output_dir)

        # 出力ディレクトリを作成
        output_path = self._create_output_directory(base_output_path)

        # 画像ファイルを取得
        image_files = self._get_image_files(input_path)

        if not image_files:
            return

        print(f"処理開始: {len(image_files)}個の画像を処理します")
        print(f"使用プロファイル: {self.profile_name}")
        print(f"出力ディレクトリ: {output_path}")

        # 設定ファイルをコピー
        self._save_config_info(output_path)

        processed_count = 0
        failed_count = 0

        for image_path in image_files:
            print(f"処理中: {image_path.name}")

            # 元画像を保存
            if save_original:
                original_dir = output_path / "original"
                original_dir.mkdir(exist_ok=True)
                original_output_path = original_dir / image_path.name
                try:
                    import shutil

                    shutil.copy2(image_path, original_output_path)
                except Exception as e:
                    print(f"警告: 元画像のコピーに失敗しました: {e}")

            # 画像を処理
            processed_image = self._process_image(image_path)

            if processed_image is not None:
                # 処理済み画像を保存
                processed_dir = output_path / "processed"
                processed_dir.mkdir(exist_ok=True)
                processed_output_path = processed_dir / image_path.name

                success = cv2.imwrite(str(processed_output_path), processed_image)
                if success:
                    processed_count += 1
                    print(f"  → 保存完了: {processed_output_path}")
                else:
                    failed_count += 1
                    print(f"  → 保存失敗: {processed_output_path}")
            else:
                failed_count += 1
                print("  → 処理失敗")

        print("\n処理完了:")
        print(f"  成功: {processed_count}個")
        print(f"  失敗: {failed_count}個")
        print(f"  出力先: {output_path}")

    def _save_config_info(self, output_path: Path) -> None:
        """
        使用した設定情報を保存.

        Args:
            output_path (Path): 出力ディレクトリのパス.
        """
        try:
            # 使用したプロファイル設定を保存
            profile_info = {
                "profile_name": self.profile_name,
                "profile_config": self.profile_config,
                "timestamp": datetime.now().isoformat(),
                "config_file": self.config_path,
            }

            info_file = output_path / "profile_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(profile_info, f, indent=2, ensure_ascii=False)

            print(f"プロファイル情報を保存しました: {info_file}")

        except Exception as e:
            print(f"警告: プロファイル情報の保存に失敗しました: {e}")

    def list_available_profiles(self) -> None:
        """利用可能なプロファイルを一覧表示."""
        cameras = self.config.get("cameras", {})
        print("利用可能なプロファイル:")
        for profile_name, profile_config in cameras.items():
            label = profile_config.get("label", "No Label")
            processors = profile_config.get("processors", [])
            mode = profile_config.get("mode", "pipeline")
            print(f"  {profile_name}: {label}")
            print(f"    モード: {mode}")
            print(f"    プロセッサ: {', '.join(processors)}")
            print()


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="プロファイルベース画像処理ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # プロファイル一覧を表示
  python profile_processor.py --list-profiles

  # プロファイル "0" を使用して画像を処理
  python profile_processor.py -p 0 -i input_images -o output_results

  # プロファイル "all_para" を使用（元画像は保存しない）
  python profile_processor.py -p all_para -i input_images -o output_results \\
    --no-save-original
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="設定ファイルのパス (デフォルト: config.json)",
    )

    parser.add_argument(
        "-p", "--profile", help="使用するプロファイル名 (例: 0, all_para, high_fps)"
    )

    parser.add_argument("-i", "--input", help="入力ディレクトリのパス")

    parser.add_argument(
        "-o",
        "--output",
        default="profile_results",
        help="出力ディレクトリのパス (デフォルト: profile_results)",
    )

    parser.add_argument(
        "--no-save-original", action="store_true", help="元画像を保存しない"
    )

    parser.add_argument(
        "--list-profiles", action="store_true", help="利用可能なプロファイルを一覧表示"
    )

    args = parser.parse_args()

    # 設定ファイルの存在確認
    if not Path(args.config).exists():
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)

    # プロファイル一覧表示
    if args.list_profiles:
        try:
            # ダミープロファイルで初期化してリスト表示
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            cameras = config.get("cameras", {})
            if cameras:
                first_profile = list(cameras.keys())[0]
                processor = ProfileProcessor(args.config, first_profile)
                processor.list_available_profiles()
            else:
                print("利用可能なプロファイルがありません")
        except Exception as e:
            print(f"エラー: プロファイル一覧の表示に失敗しました: {e}")
        return

    # 必須引数の確認
    if not args.profile:
        print("エラー: プロファイル名を指定してください (-p/--profile)")
        parser.print_help()
        sys.exit(1)

    if not args.input:
        print("エラー: 入力ディレクトリを指定してください (-i/--input)")
        parser.print_help()
        sys.exit(1)

    # プロセッサを初期化して処理実行
    try:
        processor = ProfileProcessor(args.config, args.profile)
        processor.process_directory(
            args.input, args.output, save_original=not args.no_save_original
        )
    except Exception as e:
        print(f"エラー: 処理中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
