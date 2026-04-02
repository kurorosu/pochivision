"""process サブコマンド: プロファイル適用."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import cv2
import numpy as np

from pochivision.processors.registry import get_processor


class ProfileProcessor:
    """プロファイルベースの画像処理クラス."""

    def __init__(self, config_path: str, profile_name: str) -> None:
        """ProfileProcessorのコンストラクタ.

        Args:
            config_path: config.jsonファイルのパス.
            profile_name: 使用するカメラプロファイル名.
        """
        self.config_path = config_path
        self.profile_name = profile_name
        self.config = self._load_config(config_path)
        self.profile_config = self._get_profile_config(profile_name)
        self.processors = self._initialize_processors()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む.

        Args:
            config_path: 設定ファイルのパス.

        Returns:
            設定辞書.
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

    def _get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """指定されたプロファイルの設定を取得する.

        Args:
            profile_name: プロファイル名.

        Returns:
            プロファイル設定.
        """
        cameras = self.config.get("cameras", {})
        if profile_name not in cameras:
            available_profiles = list(cameras.keys())
            print(f"エラー: プロファイル '{profile_name}' が見つかりません")
            print(f"利用可能なプロファイル: {available_profiles}")
            sys.exit(1)

        result: Dict[str, Any] = cameras[profile_name]
        return result

    def _initialize_processors(self) -> List[Any]:
        """プロファイル設定に基づいてプロセッサを初期化する.

        Returns:
            初期化されたプロセッサのリスト.
        """
        processors = []
        processor_names = self.profile_config.get("processors", [])

        for processor_name in processor_names:
            try:
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
        """入力ディレクトリから画像ファイルを取得する.

        Args:
            input_dir: 入力ディレクトリのパス.

        Returns:
            画像ファイルのパスリスト.
        """
        if not input_dir.exists():
            print(f"エラー: 入力ディレクトリが存在しません: {input_dir}")
            sys.exit(1)

        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_files: List[Path] = []

        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext.lower()}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"警告: 入力ディレクトリに画像ファイルが見つかりません: {input_dir}")
            print(f"対象拡張子: {extensions}")

        return image_files

    def _create_output_directory(self, base_output_dir: Path) -> Path:
        """タイムスタンプ付きの出力ディレクトリを作成する.

        Args:
            base_output_dir: ベース出力ディレクトリ.

        Returns:
            作成された出力ディレクトリのパス.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"profile_{self.profile_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _process_image(self, image_path: Path) -> Optional[np.ndarray]:
        """単一画像を処理する.

        Args:
            image_path: 画像ファイルのパス.

        Returns:
            処理された画像. エラーの場合は None.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告: 画像の読み込みに失敗しました: {image_path}")
                return None

            mode = self.profile_config.get("mode", "pipeline")

            if mode == "pipeline":
                processed_image = image.copy()
                original_image = image.copy()

                for processor in self.processors:
                    if hasattr(processor, "set_target_image"):
                        processor.set_target_image(original_image)

                    processed_image = processor.process(processed_image)
                    if processed_image is None:
                        print(f"警告: プロセッサでの処理に失敗しました: {image_path}")
                        return None
                return processed_image

            elif mode == "parallel":
                results = {}
                for processor in self.processors:
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
        """ディレクトリ内のすべての画像を処理する.

        Args:
            input_dir: 入力ディレクトリのパス.
            output_dir: 出力ディレクトリのパス.
            save_original: 元画像も保存するかどうか.
        """
        input_path = Path(input_dir)
        base_output_path = Path(output_dir)

        output_path = self._create_output_directory(base_output_path)

        image_files = self._get_image_files(input_path)

        if not image_files:
            return

        print(f"処理開始: {len(image_files)}個の画像を処理します")
        print(f"使用プロファイル: {self.profile_name}")
        print(f"出力ディレクトリ: {output_path}")

        self._save_config_info(output_path)

        processed_count = 0
        failed_count = 0

        for image_path in image_files:
            print(f"処理中: {image_path.name}")

            if save_original:
                original_dir = output_path / "original"
                original_dir.mkdir(exist_ok=True)
                original_output_path = original_dir / image_path.name
                try:
                    shutil.copy2(image_path, original_output_path)
                except Exception as e:
                    print(f"警告: 元画像のコピーに失敗しました: {e}")

            processed_image = self._process_image(image_path)

            if processed_image is not None:
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
        """使用した設定情報を保存する.

        Args:
            output_path: 出力ディレクトリのパス.
        """
        try:
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
        """利用可能なプロファイルを一覧表示する."""
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


@click.command()
@click.option(
    "--config", "-c", type=str, default="config.json", help="設定ファイルパス"
)
@click.option(
    "--input", "-i", "input_dir", type=str, required=True, help="入力ディレクトリ"
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=str,
    default="profile_results",
    help="出力ディレクトリ",
)
@click.option("--profile", "-p", type=str, required=True, help="カメラプロファイル名")
@click.option("--no-save-original", is_flag=True, help="元画像を保存しない")
@click.option("--list-profiles", is_flag=True, help="利用可能なプロファイルを一覧表示")
def process(
    config: str,
    input_dir: str,
    output_dir: str,
    profile: str,
    no_save_original: bool,
    list_profiles: bool,
) -> None:
    """カメラプロファイルを画像に適用する."""
    if list_profiles:
        try:
            with open(config, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            cameras = config_data.get("cameras", {})
            if cameras:
                first_profile = list(cameras.keys())[0]
                proc = ProfileProcessor(config, first_profile)
                proc.list_available_profiles()
            else:
                click.echo("利用可能なプロファイルがありません")
        except Exception as e:
            click.echo(f"エラー: プロファイル一覧の表示に失敗しました: {e}")
        return

    processor = ProfileProcessor(config, profile)
    processor.process_directory(
        input_dir, output_dir, save_original=not no_save_original
    )
