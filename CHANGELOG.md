# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- pochidetection 検出 API クライアント (`DetectionClient`) を追加. base64 エンコードで画像を送信し, bbox + class + confidence を取得する. `DetectConfig`, 専用例外 (`DetectionError` / `DetectionConnectionError`), サンプル設定 `config/detect_config.json` を同梱. ([#403](https://github.com/kurorosu/pochivision/pull/403))
- `DetectionOverlay` を追加し, `DetectionResponse` を受けてフレームに bbox / ラベル / メタ情報 (検出数, e2e_time_ms, rtt_ms, backend) を描画する. クラス ID から決定的に割り当てる 8 色パレットを内蔵. ([#407](https://github.com/kurorosu/pochivision/pull/407))

### Changed
- `DetectionClient` のバリデーションとレスポンスパースを堅牢化. フレーム dtype / shape / timeout / 接続先 URL / malformed JSON / detection 要素の型不一致を検知して適切な例外にマッピング. dtype 送信を `frame.dtype.name` で正規化. `inference/__init__.py` の docstring 半角スペースも統一. ([#406](https://github.com/kurorosu/pochivision/pull/406))
- `DetectionOverlay` で bbox 異常値 (NaN / Inf / 反転 / フレーム外) をガードしてスキップし, ラベル矩形をフレーム範囲でクリップ. BGR 3 チャネル以外のフレームは描画せず返す. スレッド安全性の注意書きを docstring に追加 (lock は #402 で導入予定). ([#410](https://github.com/kurorosu/pochivision/pull/410))

### Fixed
- 無し

### Removed
- 無し

## [0.7.0] - 2026-04-13

### Added
- 無し

### Changed
- `GrayscaleProcessor` で既にグレースケール化された入力 (`ndim == 2` / `shape[2] == 1`) を早期リターンし, 冗長な `cv2.cvtColor` 呼び出しを回避. ([#391](https://github.com/kurorosu/pochivision/pull/391))
- `ContourProcessor` で `cv2.findContours` の階層情報 (`hierarchy`) を `last_contours` / `last_hierarchy` として保持. OpenCV 3.x/4.x 両対応の `find_contours_compat` ラッパーを追加. ([#392](https://github.com/kurorosu/pochivision/pull/392))
- `CLAHEProcessor` に `update_params(clip_limit, tile_grid_size)` を追加. 内部 CLAHE オブジェクトを再生成することで動的なパラメータ変更に対応. ([#393](https://github.com/kurorosu/pochivision/pull/393))
- `BaseValidator` の docstring から削除済み `validate_config` への言及を除去. スキーマ検証への一本化方針を Note に明記. ([#396](https://github.com/kurorosu/pochivision/pull/396))
- `Registry` (processors / feature_extractors) で重複登録時に `ProcessorRegistrationError` / `ExtractorRegistrationError` を送出. 意図的な上書きには `override=True` を指定する. ([#397](https://github.com/kurorosu/pochivision/pull/397))
- バリデータのエラーメッセージに `[processor_name]` プレフィックスと受け取った値・期待値を含めるよう統一. 原因特定と UI/ログでの識別性を改善. ([#398](https://github.com/kurorosu/pochivision/pull/398))

### Fixed
- `GaussianBlurProcessor` / `MedianBlurProcessor` のカーネルサイズに対する奇数チェックを追加. 偶数・0 以下・1 を設定した場合, 実行時の `cv2.error` ではなく起動時に `ProcessorValidationError` を投げるよう修正. ([#384](https://github.com/kurorosu/pochivision/pull/384))
- `MaskCompositionProcessor` のマスク合成ロジックを明確化. 白領域に `target_image` を出力し 黒領域は 0 で埋めるセマンティクスに統一. 不要なカラー往復変換を削除し, shape/dtype のミスマッチ検証を追加. ([#385](https://github.com/kurorosu/pochivision/pull/385))
- pipeline モードでプロセッサが失敗した際に古い `result` が後続プロセッサへ渡され状態不整合が発生する問題を修正. 失敗時はパイプラインを中断し, `processed_images` にエラー情報を記録する. ([#386](https://github.com/kurorosu/pochivision/pull/386))
- parallel モードで複数プロセッサが同一 numpy 配列を共有し `ImageSaver` の `file_naming_manager` も非スレッドセーフだった問題を修正. 各プロセッサへ `image.copy()` を渡し, `ImageSaver` に `threading.Lock` を追加. ([#386](https://github.com/kurorosu/pochivision/pull/386))
- `CannyEdgeProcessor` で NaN のみフィルタされ Inf が uint8 キャスト時に不正値となる問題を修正. `np.nan_to_num` に `posinf=0.0, neginf=0.0` を追加し, 正規化バッファを `float32` 化してメモリを削減. ([#387](https://github.com/kurorosu/pochivision/pull/387))
- adaptive 2値化の `block_size` が奇数かつ 3 以上であることを起動時に検証するよう修正. 違反時は `ProcessorValidationError` を送出. `c` を `int` にキャスト. ([#388](https://github.com/kurorosu/pochivision/pull/388))
- `EqualizeProcessor` / `CLAHEProcessor` で shape `(H, W, 1)` 画像の処理を修正. `ndim==2` / `shape[2]==1` / カラー の 3 分岐を明示化し, 不要な `cvtColor(GRAY2BGR)` を削除. ([#389](https://github.com/kurorosu/pochivision/pull/389))
- `ResizeProcessor` のアスペクト比保持モードで `int()` 切り捨てによる 1px ずれを修正. `int(round(...))` で四捨五入に変更. ([#390](https://github.com/kurorosu/pochivision/pull/390))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
