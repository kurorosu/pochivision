# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- `exceptions/extractor.py` を新設し `ExtractorValidationError` / `ExtractorRuntimeError` を追加. 全例外に標準例外の多重継承を適用. ([#235](https://github.com/kurorosu/pochivision/pull/235))
- `BaseProcessor` / `BaseFeatureExtractor` に `abc.ABC` 継承を追加し抽象メソッドを強制. ([#236](https://github.com/kurorosu/pochivision/pull/236))
- プロセッサ / 特徴量抽出器の両レジストリに重複登録時の警告ログを追加. ([#237](https://github.com/kurorosu/pochivision/pull/237))
- バリデータの `validate_config` を削除しスキーマに一本化. パラメータ解析をプロセッサの `__init__` に移動. (NA.)

### Changed
- 全 9 抽出器のエラーハンドリングを `LogManager` + `raise` パターンに統一. brightness, rgb, hsv, circle_counter に try-except を追加. ([#226](https://github.com/kurorosu/pochivision/pull/226))

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
- 無し

## [0.1.5] - 2026-03-29

### Added
- LBP の振る舞いテスト 21 件を追加. ([#223](https://github.com/kurorosu/pochivision/pull/223))
- `docs/lbp_features.md` を追加. LBP 特徴量抽出器の全特徴量・パラメータ・設計制約の解説. ([#224](https://github.com/kurorosu/pochivision/pull/224))

### Changed
- 振る舞いテスト用ダミー画像を `tests/extractors/conftest.py` の `DummyImages` クラスに共通化. ([#215](https://github.com/kurorosu/pochivision/pull/215))
- Pydantic V2 非推奨 API を移行 (`min_items` → `min_length`, `class Config` → `ConfigDict`, `each_item_gt` を削除). ([#216](https://github.com/kurorosu/pochivision/pull/216))
- LBP `lbp_uniformity` を `lbp_energy` にリネームし GLCM の energy (ASM) と名称を統一. ([#219](https://github.com/kurorosu/pochivision/pull/219))
- LBP の mean/std/skewness/kurtosis を LBP 画像の直接統計に変更. ([#221](https://github.com/kurorosu/pochivision/pull/221))

### Fixed
- LBP ヒストグラム計算を `density=True` から手動正規化に変更. var メソッドの値域と nri_uniform のビン数を修正. ([#217](https://github.com/kurorosu/pochivision/pull/217))
- LBP エントロピーを `log2(n_bins)` で正規化し [0, 1] 範囲に変更. ([#218](https://github.com/kurorosu/pochivision/pull/218))
- リサイズ対応 5 抽出器のスキーマに `preserve_aspect_ratio` / `aspect_ratio_mode` を追加. ([#220](https://github.com/kurorosu/pochivision/pull/220))
- LBP の `except Exception` を `LogManager` ログ出力 + `raise` に変更. ([#222](https://github.com/kurorosu/pochivision/pull/222))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
