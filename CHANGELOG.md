# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- pipeline モードでプロセッサが失敗した際に古い `result` が後続プロセッサへ渡され状態不整合が発生する問題を修正. 失敗時はパイプラインを中断し, `processed_images` にエラー情報を記録する. ((NA.))
- parallel モードで複数プロセッサが同一 numpy 配列を共有し `ImageSaver` の `file_naming_manager` も非スレッドセーフだった問題を修正. 各プロセッサへ `image.copy()` を渡し, `ImageSaver` に `threading.Lock` を追加. ((NA.))

### Removed
- 無し

## [0.6.0] - 2026-04-07

### Added
- OS 自動検出によるカメラバックエンド自動選択機能を追加. `config.json` で `backend` 未指定時に Windows → DSHOW, Linux → V4L2, macOS → AVFOUNDATION を自動選択. ([#365](https://github.com/kurorosu/pochivision/pull/365))
- プレビュー上でマウスドラッグにより ROI (関心領域) を指定する機能を追加. 推論・キャプチャ時に ROI 領域のみを使用. `d` キーで ROI リセット. ([#366](https://github.com/kurorosu/pochivision/pull/366))
- セッション終了時にカメラ設定 (解像度, FPS, ホワイトバランス, 露出等) を `camera_config.json` として結果フォルダに保存する機能を追加. ([#368](https://github.com/kurorosu/pochivision/pull/368))

### Changed
- 推論オーバーレイを複数行表示にリデザイン. 推論結果, 信頼度, 推論時間, RTT, 画像サイズ, サーバー URL を表示. エラー時にメッセージをオーバーレイ表示. ([#362](https://github.com/kurorosu/pochivision/pull/362))

### Fixed
- 推論失敗時にもフレーム画像が保存される問題を修正. `predict` 成功後に保存するよう順序を変更. ([#360](https://github.com/kurorosu/pochivision/pull/360))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
