# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- pochidetection 検出 API クライアント (`DetectionClient`) を追加. `DetectConfig` / 専用例外 / サンプル `config/detect_config.json` 同梱. ([#403](https://github.com/kurorosu/pochivision/pull/403))
- `DetectionOverlay` を追加. `DetectionResponse` を受けて bbox / ラベル / メタ情報 (検出数 / e2e_time_ms / rtt_ms / backend) を描画. class ID からの決定的 8 色パレット内蔵. ([#407](https://github.com/kurorosu/pochivision/pull/407))
- 常時検出ランタイムを `CaptureRunner` に統合. `time.perf_counter()` ベースのスロットリング + 非同期スレッドで検出し `DetectionOverlay` に反映. `i` キーで ON/OFF トグル, detect モードは ROI 無効化. `DetectionOverlay` の state 更新 / draw を `threading.Lock` で保護. ([#414](https://github.com/kurorosu/pochivision/pull/414))
- 検出 API の処理時間メトリクス (`e2e_time_ms` / `phase_times_ms` / `rtt_ms` / GPU clock・VRAM・温度) を `metrics_interval_s` 間隔でサンプリングし `capture/<run>/detection_metrics.csv` に pandas 経由で保存する `MetricsRecorder` を追加. 設定は `detect_config.json` の `metrics_interval_s` で制御. ([#418](https://github.com/kurorosu/pochivision/pull/418))

### Changed
- **BREAKING**: 検出モードの有効化を `DetectConfig.mode` から CLI フラグ `--detect` に変更. 既存 JSON の `mode` キーは warning を出して無視 (後方互換). ([#416](https://github.com/kurorosu/pochivision/pull/416))
- `DetectionResponse` に `phase_times_ms` / `gpu_clock_mhz` / `gpu_vram_used_mb` / `gpu_temperature_c` フィールドを追加. サーバー未提供時は空 dict / None で補う. ([#418](https://github.com/kurorosu/pochivision/pull/418))
- `DetectionOverlay` の `Inference: X.Xms` 表示を実体に合わせ `E2E: X.Xms` に変更. `phase_times_ms.pipeline_inference_ms` が返る場合は純粋な推論時間を `Infer: X.Xms` として別行に追加. ([#420](https://github.com/kurorosu/pochivision/pull/420))
- `DetectionOverlay` の E2E 内訳表示を拡張. `phase_times_ms` の `api_preprocess_ms` / `pipeline_preprocess_ms` / `pipeline_inference_ms` / `pipeline_postprocess_ms` / `api_postprocess_ms` を `- ` プレフィックス付きサブ行で時系列順に表示 (例: `- APIpre: 1.4ms` / `- Pre: 1.1ms` / `- Infer: 8.2ms` / `- Post: 0.5ms` / `- APIpost: 0.9ms`). キー欠損時はその行を出さない. ([#422](https://github.com/kurorosu/pochivision/pull/422))
- `MetricsRecorder` に `api_preprocess_ms` / `api_postprocess_ms` カラムを追加し, `detection_metrics.csv` の phase 群を時系列順 (api_pre → pipeline_* → api_post) に整理. キー欠損時は空セル. ([#422](https://github.com/kurorosu/pochivision/pull/422))
- `DetectionClient.detect()` の全体所要時間 (画像エンコード + RTT + JSON parse) を `DetectionResponse.total_ms` として計測・返却. オーバーレイに `Total: X.Xms` 行を追加 (Total ⊃ RTT ⊃ E2E の階層が縦並びで表示). `detection_metrics.csv` にも `total_ms` カラムを追加. ([#430](https://github.com/kurorosu/pochivision/pull/430))
- **BREAKING**: API クライアント / config のキー名を `url` → `base_url`, `format` → `image_format` に統一. 既存 `config/*.json` は要更新. ([#412](https://github.com/kurorosu/pochivision/pull/412))
- `DetectionClient` のバリデーション / レスポンスパースを堅牢化. frame dtype / shape / timeout / URL / JSON / 型不一致を適切な例外にマッピング. ([#406](https://github.com/kurorosu/pochivision/pull/406))
- `DetectionOverlay` で bbox 異常値 (NaN / Inf / 反転 / フレーム外) をガード, ラベル矩形をフレーム範囲でクリップ. 非 BGR 3ch フレームは描画しない. ([#410](https://github.com/kurorosu/pochivision/pull/410))
- `DetectionOverlay` / `InferenceOverlay` の色定数定義を統一. 共通色 (`META_COLOR` / `ERROR_COLOR`) を `capture_runner/_overlay_colors.py` に抽出. ([#413](https://github.com/kurorosu/pochivision/pull/413))

### Fixed
- 経過時間計測に `time.time()` (wall-clock) を使用していた 9 箇所を `time.perf_counter()` に置換 (`capture_runner/viewer._measure_actual_fps`, `core/image_saver`, `core/pipeline_executor`). NTP 同期 / 時刻調整で計測値が歪む問題を解消. ([#431](https://github.com/kurorosu/pochivision/pull/431))
- `GLCMTextureExtractor` で単色 / 均一画像時に `correlation` が NaN (σ_x σ_y = 0 の不定形) となり後段の集計で全体が NaN に汚染される問題を修正. correlation の NaN を 1.0 (完全相関) にフォールバックし, warning ログは従来通り出力. 他 5 特徴量 (contrast / dissimilarity / homogeneity / energy / ASM) は skimage が単色画像で正常な値 (0 / 1) を返すため影響なし. ([#432](https://github.com/kurorosu/pochivision/pull/432))
- `DetectionOverlay` で送信フレーム (オリジナル) 座標系の bbox をリサイズ済みプレビューにスケール補正なしで描画していた問題を修正. `ROIRectSelector` と同じパターンで `set_preview_scale(frame_w, preview_w)` を追加し, `_draw_bbox` 内で bbox を縮小描画する. `viewer.py` の detect モード分岐から毎フレーム scale を設定. 静止物体の bbox がキャプチャ画像とずれる現象を解消. ((NA.))

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
