# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- pochitrain 推論 API と連携するリアルタイム推論オーバーレイ機能を追加. `i` キーで現在のフレームを API に送信し, クラス名・信頼度・推論時間をプレビュー上に表示する. CLI に `--inference-url`, `--inference-format` オプションを追加. (NA.)
  - 推論 API クライアント (`pochivision/request/api/inference/`)
  - 推論結果オーバーレイ (`InferenceOverlay`)
  - 推論例外クラス (`InferenceError`, `InferenceConnectionError`)
  - テスト 25 件

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## [0.4.2] - 2026-04-04

### Added
- 例外クラスのテスト 21 件を追加. ([#339](https://github.com/kurorosu/pochivision/pull/339))
- `PipelineExecutor.from_config()` のエラーケーステスト 5 件を追加. ([#340](https://github.com/kurorosu/pochivision/pull/340))
- `FeatureExtractionRunner._initialize_extractors()` のエラーケーステスト 3 件を追加. ([#341](https://github.com/kurorosu/pochivision/pull/341))
- `ProfileProcessor._process_image()` の parallel モード・エッジケーステスト 4 件を追加. ([#342](https://github.com/kurorosu/pochivision/pull/342))

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
