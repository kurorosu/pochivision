# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## [0.5.0] - 2026-04-06

### Added
- pochitrain 推論 API と連携するリアルタイム推論オーバーレイ機能を追加. `i` キーで現在のフレームを API に送信し, クラス名・信頼度・推論時間をプレビュー上に表示する. CLI に `--inference-url`, `--inference-format` オプションを追加. ([#345](https://github.com/kurorosu/pochivision/pull/345))
  - 推論 API クライアント (`pochivision/request/api/inference/`)
  - 推論結果オーバーレイ (`InferenceOverlay`)
  - 推論例外クラス (`InferenceError`, `InferenceConnectionError`)
  - テスト 25 件
- 推論実行時にリサイズ+パディング後のフレーム画像を保存する機能を追加. `infer_config.json` の `save_frame` で有効/無効を制御. ([#355](https://github.com/kurorosu/pochivision/pull/355))
- 推論結果を CSV ファイルに蓄積する機能を追加. `infer_config.json` の `save_csv` で有効/無効を制御. ([#356](https://github.com/kurorosu/pochivision/pull/356))

### Changed
- 推論オーバーレイにクライアント側ネットワーク往復時間 (RTT) の表示を追加. ([#350](https://github.com/kurorosu/pochivision/pull/350))
- 推論設定を `infer_config.json` に集約し, CLI オプション `--inference-url` / `--inference-format` を `--infer-config` に統合. 送信前のフレームリサイズ+パディング (アスペクト比維持) をサポート. ([#353](https://github.com/kurorosu/pochivision/pull/353))

### Fixed
- 推論実行時にプレビューがブロックされる問題を修正. `_run_inference` をバックグラウンドスレッドで実行するよう変更し, 推論中の二重送信防止と "Inferring..." 表示を追加. ([#348](https://github.com/kurorosu/pochivision/pull/348))
- `_inferring` フラグの check-then-act を `threading.Lock` で保護し, 推論の二重起動を確実に防止. ([#357](https://github.com/kurorosu/pochivision/pull/357))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
