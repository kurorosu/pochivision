# 検出モード (`--detect`)

`pochi run --detect` は pochidetection WebAPI を使った**常時検出ランタイム**を有効化する. 入力 FPS に対して `detect_fps` でスロットリングしつつ非同期に検出リクエストを送信し, 結果をプレビューに bbox + メタ情報として描画する.

detect モードでは ROI 選択は無効化され, 常にフル解像度フレームが送信される (bbox 座標系を保つためクライアント側のリサイズは行わない).

## 1. 基本コマンド

```bash
# 検出モードで起動 (デフォルト設定 config/detect_config.json を使用)
uv run pochi run --detect

# カスタム設定ファイルを指定
uv run pochi run --detect --detect-config config/my_detect.json
```

## 2. キー操作

起動直後は検出 OFF の状態で待機する. `i` キーで検出の ON/OFF をトグルできる (classify モードの「`i` で推論」と対称).

接続失敗やタイムアウト時はキャプチャループを止めず, overlay にエラーメッセージが表示される.

## 3. 設定ファイル (`detect_config.json`)

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `base_url` | Yes | - | pochidetection 検出 API のベース URL |
| `image_format` | No | `"raw"` | 画像送信形式 (`"raw"` / `"jpeg"`) |
| `score_threshold` | No | `0.5` | 検出信頼度の下限しきい値 (0.0-1.0) |
| `timeout` | No | `5.0` | リクエストタイムアウト (秒) |
| `jpeg_quality` | No | `90` | JPEG 圧縮品質 (1-100, `image_format="jpeg"` のとき) |
| `detect_fps` | No | `5.0` | `--detect` 有効時の検出リクエスト頻度 (Hz) |
| `metrics_interval_s` | No | `0.0` | 処理時間メトリクスのサンプリング間隔 (秒). 0 以下で無効 |

## 4. 処理時間メトリクス

`metrics_interval_s` を正の値 (例: `1.0`) に設定すると, 指定間隔ごとに検出 API のレスポンスから処理時間メトリクス (`e2e_time_ms` / phase 別 / RTT / GPU clock / VRAM / 温度) を採取し, 終了時に `capture/<timestamp>/detection_metrics.csv` へ pandas 経由で書き出す. 毎フレームではなくダウンサンプリングするため, 高 FPS 運用でもファイル肥大化しない.
