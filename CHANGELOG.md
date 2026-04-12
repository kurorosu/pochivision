# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- `GaussianBlurProcessor` / `MedianBlurProcessor` のカーネルサイズに対する奇数チェックを追加. 偶数・0 以下・1 を設定した場合, 実行時の `cv2.error` ではなく起動時に `ProcessorValidationError` を投げるよう修正. ([#384](https://github.com/kurorosu/pochivision/pull/384))
- `MaskCompositionProcessor` のマスク合成ロジックを明確化. 白領域に `target_image` を出力し 黒領域は 0 で埋めるセマンティクスに統一. 不要なカラー往復変換を削除し, shape/dtype のミスマッチ検証を追加. ([#385](https://github.com/kurorosu/pochivision/pull/385))
- pipeline モードでプロセッサが失敗した際に古い `result` が後続プロセッサへ渡され状態不整合が発生する問題を修正. 失敗時はパイプラインを中断し, `processed_images` にエラー情報を記録する. ([#386](https://github.com/kurorosu/pochivision/pull/386))
- parallel モードで複数プロセッサが同一 numpy 配列を共有し `ImageSaver` の `file_naming_manager` も非スレッドセーフだった問題を修正. 各プロセッサへ `image.copy()` を渡し, `ImageSaver` に `threading.Lock` を追加. ([#386](https://github.com/kurorosu/pochivision/pull/386))
- `CannyEdgeProcessor` で NaN のみフィルタされ Inf が uint8 キャスト時に不正値となる問題を修正. `np.nan_to_num` に `posinf=0.0, neginf=0.0` を追加し, 正規化バッファを `float32` 化してメモリを削減. ([#387](https://github.com/kurorosu/pochivision/pull/387))
- adaptive 2値化の `block_size` が奇数かつ 3 以上であることを起動時に検証するよう修正. 違反時は `ProcessorValidationError` を送出. `c` を `int` にキャスト. ([#388](https://github.com/kurorosu/pochivision/pull/388))
- `EqualizeProcessor` / `CLAHEProcessor` で shape `(H, W, 1)` 画像の処理を修正. `ndim==2` / `shape[2]==1` / カラー の 3 分岐を明示化し, 不要な `cvtColor(GRAY2BGR)` を削除. ([#389](https://github.com/kurorosu/pochivision/pull/389))
- `ResizeProcessor` のアスペクト比保持モードで `int()` 切り捨てによる 1px ずれを修正. `int(round(...))` で四捨五入に変更. (NA.)

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
