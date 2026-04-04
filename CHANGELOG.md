# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 例外クラスのテスト 21 件を追加. (NA.)

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## [0.4.1] - 2026-04-04

### Added
- 無し

### Changed
- `SimpleFFTVisualizer` の `print()` を `LogManager` 経由のログ出力に統一. ([#318](https://github.com/kurorosu/pochivision/pull/318))
- `processors/registry.py` と `feature_extractors/registry.py` のロガーを `LogManager` に統一. ([#319](https://github.com/kurorosu/pochivision/pull/319))
- プロセッサ (`binarization.py`, `clahe.py`, `equalize.py`) の `logging.getLogger` を `LogManager` に統一. ([#332](https://github.com/kurorosu/pochivision/pull/332))
- 型アノテーションを Python 3.12+ 新スタイル (`dict`, `list`, `tuple`, `X | None`) に統一. ([#333](https://github.com/kurorosu/pochivision/pull/333))

### Fixed
- `RecordingManager.stop_recording()` の `is_recording` チェックをロック内に移動し race condition を修正. ([#334](https://github.com/kurorosu/pochivision/pull/334))
- `_resize_for_preview()` でフレームサイズが 0 の場合に元フレームを返す防御コードを追加. ([#335](https://github.com/kurorosu/pochivision/pull/335))
- `CircleCounterExtractor` の `circle_density` 計算で画像面積が 0 の場合に除算ゼロを防止. ([#336](https://github.com/kurorosu/pochivision/pull/336))
- `ResizeProcessor` の `aspect_ratio` 計算で元画像の高さが 0 の場合に除算ゼロを防止. ([#337](https://github.com/kurorosu/pochivision/pull/337))

### Removed
- `RecordingManager` の未使用属性 `frame_queue`, `recording_thread` を削除. ([#320](https://github.com/kurorosu/pochivision/pull/320))
- `tests/conftest.py` の未使用フィクスチャ `dummy_color_image`, `dummy_grayscale_image` を削除. ([#321](https://github.com/kurorosu/pochivision/pull/321))

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
