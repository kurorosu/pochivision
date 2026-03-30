# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 全 9 抽出器のエラーハンドリングを `LogManager` + `raise` パターンに統一. brightness, rgb, hsv, circle_counter に try-except を追加. (NA.)

### Fixed
- brightness, rgb, hsv, circle_counter に float (0-1) 入力の uint8 スケール変換を追加. dtype 一致テスト 4 件も追加. (NA.)
- Brightness スキーマに `exclude_zero_pixels`, HLAC スキーマに `binarization_method` / `adaptive_block_size` / `adaptive_c` を追加. (NA.)

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
