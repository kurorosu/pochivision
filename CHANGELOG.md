# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- `SimpleFFTVisualizer` の `print()` を `LogManager` 経由のログ出力に統一. ([#318](https://github.com/kurorosu/pochivision/pull/318))
- `processors/registry.py` と `feature_extractors/registry.py` のロガーを `LogManager` に統一. ([#319](https://github.com/kurorosu/pochivision/pull/319))
- プロセッサ (`binarization.py`, `clahe.py`, `equalize.py`) の `logging.getLogger` を `LogManager` に統一. (NA.)

### Fixed
- 無し

### Removed
- `RecordingManager` の未使用属性 `frame_queue`, `recording_thread` を削除. ([#320](https://github.com/kurorosu/pochivision/pull/320))
- `tests/conftest.py` の未使用フィクスチャ `dummy_color_image`, `dummy_grayscale_image` を削除. ([#321](https://github.com/kurorosu/pochivision/pull/321))

## [0.4.0] - 2026-04-04

### Added
- ライブプレビューに `h` キーでトグルするヘルプオーバーレイ表示機能を追加. 黒文字 + 白縁で表示し, 保存画像には影響しない. ([#295](https://github.com/kurorosu/pochivision/pull/295))
- `FeatureExtractionRunner` の統合テスト 7 件を追加. ([#314](https://github.com/kurorosu/pochivision/pull/314))
- `ProfileProcessor` のテスト 7 件を追加. ([#315](https://github.com/kurorosu/pochivision/pull/315))
- `SimpleFFTVisualizer` のロジックテスト 10 件を追加. ([#316](https://github.com/kurorosu/pochivision/pull/316))

### Changed
- 設定ファイル (`config.json`, `extractor_config.json`) を `config/` ディレクトリに移動し, CLI のデフォルトパスを更新. ([#294](https://github.com/kurorosu/pochivision/pull/294))
- `FeatureExtractionRunner` の CSV 出力を `FeatureCSVWriter` に, クラス名抽出を `extract_class_from_filename()` に分離. ([#297](https://github.com/kurorosu/pochivision/pull/297))

### Fixed
- `ImageSaver.save()` で `cv2.imwrite()` の戻り値を検証し, 保存失敗時に警告ログを出力するよう修正. ([#291](https://github.com/kurorosu/pochivision/pull/291))
- `RecordingManager.add_frame()` のロック外チェックを削除しスレッド安全性を改善. `start_recording()` にフレームサイズ検証を追加. ([#292](https://github.com/kurorosu/pochivision/pull/292))
- `ImageSaver.save()` のログ出力でグレースケール画像の幅/高さが正しく取得されるよう `image.shape[:2]` に修正. ([#293](https://github.com/kurorosu/pochivision/pull/293))
- `run.py` の `SystemExit(1)` を `click.ClickException` に統一, `logger` パラメータ型と `_setup_camera()` 戻り値型を修正. ([#309](https://github.com/kurorosu/pochivision/pull/309))
- `FeatureExtractionRunner` の特徴量ユニット名取得失敗時に警告ログを出力するよう修正. ([#310](https://github.com/kurorosu/pochivision/pull/310))
- `_measure_actual_fps()` の除算ゼロリスクに防御コードを追加. ([#311](https://github.com/kurorosu/pochivision/pull/311))
- `RecordingManager.start_recording()` の `is_recording` チェックをロック内に移動し race condition を修正. ([#312](https://github.com/kurorosu/pochivision/pull/312))

### Removed
- `feature_extractors/__init__.py` から未使用の Params クラス 9 件のエクスポートを削除. ([#296](https://github.com/kurorosu/pochivision/pull/296))

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
