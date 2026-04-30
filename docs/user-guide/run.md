# ライブプレビュー (`pochi run`)

`pochi run` はカメラからのリアルタイムキャプチャとプレビューを起動する. 設定済みのカメラプロファイルに従い, 指定したプロセッサパイプラインを各フレームに適用しながらライブ表示する.

## 1. 基本コマンド

```bash
# デフォルト設定で起動 (config/config.json の selected_camera_index を使用)
uv run pochi run

# 特定のカメラデバイスとプロファイルを指定
uv run pochi run --camera 2 --profile high_fps

# 利用可能な全カメラプロファイルを一覧表示
uv run pochi run --list-profiles

# 録画機能を無効にして起動
uv run pochi run --no-recording
```

## 2. 引数

| 引数 | 短縮形 | 説明 |
|------|--------|------|
| `--camera` | `-c` | カメラデバイスインデックス (`0`, `1`, `2`...). 単独使用時はプロファイル "0" を使用 |
| `--profile` | `-p` | `config.json` のカメラプロファイル名 |
| `--list-profiles` | `-l` | 利用可能な全カメラプロファイルを表示 |
| `--config` | | 設定ファイルのパス (デフォルト: `config/config.json`) |
| `--no-recording` | | 録画機能を無効にして起動 |
| `--infer-config` | | 推論設定ファイルのパス (デフォルト: `config/infer_config.json`) |
| `--detect-config` | | 検出設定ファイルのパス (デフォルト: `config/detect_config.json`) |
| `--detect` | | 常時検出ランタイムを有効化 (指定無しは classify モード) |

## 3. キーボード操作

| キー | 説明 |
|------|------|
| `q` | 終了 |
| `i` | 推論 / 検出の ON/OFF トグル (classify モード: 推論 1 回実行, detect モード: 常時検出 ON/OFF) |
| `s` | カメラ設定ダイアログ (Windows) |

`Ctrl+C` でも安全に停止できる.

## 4. 録画

`--no-recording` を指定しない限り, `recording.select_format` で指定されたコーデックでフレームを動画として保存する. 出力先は `capture/<timestamp>/` 配下.

| `select_format` | 説明 |
|-----------------|------|
| `mp4v` | MP4 標準コーデック |
| `xvid` | Xvid (AVI) |
| `mjpg` | Motion JPEG (デフォルト) |
| `ffv1` | ロスレス (FFV1) |

詳細な録画設定は [設定ファイル](configuration.md) を参照.

## 5. 連携機能

- 検出 API (pochidetection) との常時検出オーバーレイ: [検出モード](detection.md)
- 推論 API (pochitrain) とのリアルタイム分類: [推論モード](inference.md)
