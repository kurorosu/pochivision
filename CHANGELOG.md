# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- `ConfigHandler.save()` の `strftime` フォーマットを `%Y-%m%d-%H%M-%S` から `%Y%m%d_%H%M%S` に修正. ([#263](https://github.com/kurorosu/pochivision/pull/263))
- `RecordingManager.start_recording()` で `VideoWriter` 初期化失敗時に `video_writer` を `None` にクリアするよう修正. ([#264](https://github.com/kurorosu/pochivision/pull/264))

### Removed
- 未使用の `ExtractorRuntimeError` 例外クラスを削除. (NA.)

## [0.2.0] - 2026-04-02

### Added
- `exceptions/extractor.py` を新設し `ExtractorValidationError` / `ExtractorRuntimeError` を追加. 全例外に標準例外の多重継承を適用. ([#235](https://github.com/kurorosu/pochivision/pull/235))
- `BaseProcessor` / `BaseFeatureExtractor` に `abc.ABC` 継承を追加し抽象メソッドを強制. ([#236](https://github.com/kurorosu/pochivision/pull/236))
- プロセッサ / 特徴量抽出器の両レジストリに重複登録時の警告ログを追加. ([#237](https://github.com/kurorosu/pochivision/pull/237))
- バリデータの `validate_config` を削除しスキーマに一本化. パラメータ解析をプロセッサの `__init__` に移動. ([#238](https://github.com/kurorosu/pochivision/pull/238))
- プロセッサスキーマの検証漏れを補完. `MaskCompositionParams` を新設. `@field_validator` / `@model_validator` で複合条件を追加. ([#240](https://github.com/kurorosu/pochivision/pull/240))
- `extractor_config.json` に `extractors` リストを追加し, 選択した抽出器のみ実行する機能を追加. ([#241](https://github.com/kurorosu/pochivision/pull/241))
- `pochi` CLI を click サブコマンド構成に変更. `pochi run` / `pochi extract` / `pochi process` / `pochi aggregate` / `pochi fft` を追加. ([#242](https://github.com/kurorosu/pochivision/pull/242))
- CLI サブコマンドを `commands/` ディレクトリに分離. tools/ のスクリプト本体を移動し, `sys.argv` 操作を排除. `tqdm` を依存に追加. ([#245](https://github.com/kurorosu/pochivision/pull/245))

### Changed
- 全 9 抽出器のエラーハンドリングを `LogManager` + `raise` パターンに統一. brightness, rgb, hsv, circle_counter に try-except を追加. ([#226](https://github.com/kurorosu/pochivision/pull/226))
- README.md / README.en.md を CLI サブコマンド構成に合わせて更新. ディレクトリ構成・アーキテクチャセクションを削除. CLAUDE.md の CLI・tools 記述を更新. ([#246](https://github.com/kurorosu/pochivision/pull/246))
- `OutputManager` を新設し全サブコマンドの出力先を `outputs/{command_type}/YYYYMMDD_{suffix}/` に統一. `CaptureManager` を廃止し `PipelineExecutor` に `Path` を直接渡す構造に変更. `--output-root` オプションを追加. ([#247](https://github.com/kurorosu/pochivision/pull/247))

### Fixed
- brightness, rgb, hsv, circle_counter に float (0-1) 入力の uint8 スケール変換を追加. dtype 一致テスト 4 件も追加. ([#227](https://github.com/kurorosu/pochivision/pull/227))
- Brightness スキーマに `exclude_zero_pixels`, HLAC スキーマに `binarization_method` / `adaptive_block_size` / `adaptive_c` を追加. ([#228](https://github.com/kurorosu/pochivision/pull/228))
- circle_counter の `blur_kernel_size` に偶数バリデーションを追加. ([#230](https://github.com/kurorosu/pochivision/pull/230))
- RGB/HSV/Brightness の `exclude_black_pixels` / `exclude_zero_pixels` の動作を docstring とコメントに明記. ([#231](https://github.com/kurorosu/pochivision/pull/231))
- `get_feature_extractor` で Pydantic スキーマによる設定バリデーションを実行するよう変更. ([#232](https://github.com/kurorosu/pochivision/pull/232))
- スキーマの関心分離, プロセッサバリデーション追加, 例外階層の統一. ([#233](https://github.com/kurorosu/pochivision/pull/233))
  - プロセッサスキーマを `processors/schema.py` に分離し `config_schema.py` を `schema.py` にリネーム
  - `get_processor` で Pydantic スキーマによる設定バリデーションを実行するよう変更
  - `ConfigLoadError` / `CameraConfigError` を `exceptions/config.py` に移動し `VisionCaptureError` 階層に統一

### Removed
- `CaptureManager` を廃止. `PipelineExecutor` が `output_dir: Path` を直接受け取る構造に変更. ([#247](https://github.com/kurorosu/pochivision/pull/247))
- tools/ の 4 スクリプト (`feature_extraction.py`, `profile_processor.py`, `image_aggregator.py`, `fft_visualizer.py`) を `cli/commands/` に移動し削除. ([#245](https://github.com/kurorosu/pochivision/pull/245))

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
