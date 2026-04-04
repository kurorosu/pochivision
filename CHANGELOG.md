# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- ライブプレビューに `h` キーでトグルするヘルプオーバーレイ表示機能を追加. 黒文字 + 白縁で表示し, 保存画像には影響しない. (NA.)

### Changed
- 設定ファイル (`config.json`, `extractor_config.json`) を `config/` ディレクトリに移動し, CLI のデフォルトパスを更新. (NA.)

### Fixed
- `ImageSaver.save()` で `cv2.imwrite()` の戻り値を検証し, 保存失敗時に警告ログを出力するよう修正. ([#291](https://github.com/kurorosu/pochivision/pull/291))
- `RecordingManager.add_frame()` のロック外チェックを削除しスレッド安全性を改善. `start_recording()` にフレームサイズ検証を追加. ([#292](https://github.com/kurorosu/pochivision/pull/292))
- `ImageSaver.save()` のログ出力でグレースケール画像の幅/高さが正しく取得されるよう `image.shape[:2]` に修正. ([#293](https://github.com/kurorosu/pochivision/pull/293))

### Removed
- `feature_extractors/__init__.py` から未使用の Params クラス 9 件のエクスポートを削除. (NA.)

## [0.3.0] - 2026-04-04

### Added
- `PipelineExecutor`, `ImageSaver`, `RecordingManager`, `CameraSetup`, `ImageAggregator`, `ProcessorFolderFinder`, `get_image_files`, `load_image` のユニットテストを追加. ([#279](https://github.com/kurorosu/pochivision/pull/279))
- CLI サブコマンドの実行テストを追加 (extract, process, aggregate, output-root 伝播). ([#280](https://github.com/kurorosu/pochivision/pull/280))

### Changed
- `FeatureExtractionRunner` と `ProfileProcessor` の `_load_config()` を `ConfigHandler.load_json()` に統合. ([#267](https://github.com/kurorosu/pochivision/pull/267))
- `extract.py` と `process.py` の `_get_image_files()` を `utils/image.py` の `get_image_files()` に統合. ([#268](https://github.com/kurorosu/pochivision/pull/268))
- `cv2.imread` + None チェックパターンを `utils/image.py` の `load_image()` に統合. ([#269](https://github.com/kurorosu/pochivision/pull/269))
- CLI コマンドの `sys.exit(1)` を `click.ClickException` に置換. ([#270](https://github.com/kurorosu/pochivision/pull/270))
- `extract.py` と `process.py` の `print()` を `LogManager` に統一. ([#271](https://github.com/kurorosu/pochivision/pull/271))
- マジックナンバーを `pochivision/constants.py` に定数化. ([#273](https://github.com/kurorosu/pochivision/pull/273))
- CLI コマンドからビジネスロジッククラスを `core/` に分離. ([#274](https://github.com/kurorosu/pochivision/pull/274))
- `PipelineExecutor` の File I/O 責務を `ImageSaver` クラスに分離. ([#275](https://github.com/kurorosu/pochivision/pull/275))

### Fixed
- `ConfigHandler.save()` の `strftime` フォーマットを `%Y-%m%d-%H%M-%S` から `%Y%m%d_%H%M%S` に修正. ([#263](https://github.com/kurorosu/pochivision/pull/263))
- `RecordingManager.start_recording()` で `VideoWriter` 初期化失敗時に `video_writer` を `None` にクリアするよう修正. ([#264](https://github.com/kurorosu/pochivision/pull/264))
- mypy 型エラー 17 件を修正. `types-tqdm`, `scipy-stubs` を dev 依存に追加, `cv2.normalize` の `dst` 引数修正, `mask_composition.py` の `signedinteger` 変換修正など. ([#281](https://github.com/kurorosu/pochivision/pull/281))

### Removed
- 未使用の `ExtractorRuntimeError` 例外クラスを削除. ([#265](https://github.com/kurorosu/pochivision/pull/265))
- `CameraConfigHandler` の未使用メソッド (`get_camera_config`, `get_all_camera_indices`, `get_selected_camera_index`) を削除. ([#266](https://github.com/kurorosu/pochivision/pull/266))
- `tools/` ディレクトリを全削除. ([#278](https://github.com/kurorosu/pochivision/pull/278))
- `extractor_config.json` から未使用の `output_directory`, `include_filename` キーを削除. ([#282](https://github.com/kurorosu/pochivision/pull/282))

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
