# pochivision

[![Version](https://img.shields.io/badge/version-0.7.0-green.svg)](https://github.com/kurorosu/pochivision/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

[English](README.en.md)

プラグイン方式でプロセッサを拡張可能なAIビジョンアプリケーション向けリアルタイム画像キャプチャ・前処理エンジン.

## 動作環境

- Python 3.12+

## 機能

- 設定可能なプロファイルを持つ複数カメラサポート
- リアルタイム画像キャプチャと処理
- パイプライン / パラレル実行モードによる画像処理
- レジストリパターンによる拡張可能なプロセッサ・特徴量抽出システム
- プレビューウィンドウサイズの設定
- コーデック選択可能な録画機能
- 柔軟なコマンドラインインターフェース

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/kurorosu/pochivision.git
cd pochivision

# 依存関係のインストール
uv sync
```

## 開発者向けセットアップ

開発・テスト・Lint 等も含めてインストールする場合:

```bash
uv sync --group dev
```

## クイックスタート

デフォルト設定でアプリケーションを実行:

```bash
uv run pochi run
```

## コマンドラインインターフェース

pochivision は `pochi` コマンドによるサブコマンド構成の CLI を提供しています:

```bash
# サブコマンド一覧を表示
uv run pochi --help
```

### `pochi run` - ライブプレビュー起動

カメラからのリアルタイムキャプチャとプレビューを起動します.

```bash
# デフォルト設定で起動
uv run pochi run

# 特定のカメラデバイスとプロファイルを指定
uv run pochi run --camera 2 --profile "high_fps"

# 利用可能な全てのカメラプロファイルを一覧表示
uv run pochi run --list-profiles

# 録画機能を無効にして起動
uv run pochi run --no-recording

# pochitrain 推論 API と連携 (i キーで推論実行, デフォルト: config/infer_config.json)
uv run pochi run

# 推論設定ファイルを明示的に指定
uv run pochi run --infer-config config/infer_config.json
```

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--camera` | `-c` | カメラデバイスインデックスを指定 (0, 1, 2...). 単独使用時はプロファイル "0" を使用 |
| `--profile` | `-p` | config.json からカメラプロファイルを指定 |
| `--list-profiles` | `-l` | 利用可能な全てのカメラプロファイルを表示 |
| `--config` | | 設定ファイルのパスを指定 (デフォルト: config/config.json) |
| `--no-recording` | | 録画機能を無効にして起動 |
| `--infer-config` | | 推論設定ファイルのパス (デフォルト: config/infer_config.json) |
| `--detect-config` | | 検出設定ファイルのパス (デフォルト: config/detect_config.json) |

#### 推論設定 (`infer_config.json`)

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `base_url` | Yes | - | pochitrain 推論 API のベース URL |
| `image_format` | No | `"jpeg"` | 画像送信形式 (`"raw"` / `"jpeg"`) |
| `resize.width` | No | なし (リサイズなし) | 送信画像の幅 |
| `resize.height` | No | なし (リサイズなし) | 送信画像の高さ |
| `resize.padding_color` | No | `[0, 0, 0]` | パディング色 (BGR) |
| `save_frame` | No | `false` | 推論実行時にフレーム画像を保存するか |
| `save_csv` | No | `false` | 推論結果を CSV ファイルに出力するか |

> **Migration (0.7 → 0.8)**: キー名が `url` → `base_url`, `format` → `image_format` に変更されました. お使いの `config/infer_config.json` / `config/detect_config.json` の該当キーをリネームしてください (旧キーを含む設定は `ConfigValidationError: 'base_url' が必要です` エラーになります).

#### 検出設定 (`detect_config.json`)

`mode = "detect"` を設定することで, pochidetection WebAPI を使った**常時検出ランタイム**が有効化されます. 入力 FPS に対して `detect_fps` でスロットリングしつつ非同期に検出リクエストを送信し, 結果をプレビューに bbox + メタ情報として描画します. detect モードでは ROI 選択は無効化され, 常にフル解像度フレームが送信されます (bbox 座標系を保つためクライアント側のリサイズは行わない).

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `base_url` | Yes | - | pochidetection 検出 API のベース URL |
| `image_format` | No | `"raw"` | 画像送信形式 (`"raw"` / `"jpeg"`) |
| `score_threshold` | No | `0.5` | 検出信頼度の下限しきい値 (0.0-1.0) |
| `timeout` | No | `5.0` | リクエストタイムアウト (秒) |
| `jpeg_quality` | No | `90` | JPEG 圧縮品質 (1-100, `image_format="jpeg"` のとき) |
| `mode` | No | `"classify"` | ランタイムモード (`"classify"` / `"detect"`). `"detect"` で常時検出ランタイム有効化 |
| `detect_fps` | No | `5.0` | `mode="detect"` 時の検出リクエスト頻度 (Hz) |

実行中は `i` キーで検出の ON/OFF をトグルできます. 接続失敗やタイムアウト時はキャプチャループを止めず, overlay にエラーメッセージが表示されます.

### `pochi extract` - 特徴量抽出

画像から特徴量を抽出し, CSV ファイルに出力します.

```bash
# デフォルト設定ファイルで実行
uv run pochi extract

# 設定ファイルを指定して実行
uv run pochi extract --config my_extractor_config.json
```

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--config` | `-c` | 設定ファイルのパスを指定 (デフォルト: config/extractor_config.json) |

### `pochi process` - プロファイル適用

カメラプロファイルの処理パイプラインを画像に適用します.

```bash
# 入力ディレクトリの画像にプロファイルを適用
uv run pochi process --input ./images --profile "high_res"

# 出力先を指定し, 元画像の保存をスキップ
uv run pochi process --input ./images --output ./processed --profile "high_res" --no-save-original

# 利用可能なプロファイルを一覧表示
uv run pochi process --list-profiles
```

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--config` | `-c` | 設定ファイルのパスを指定 (デフォルト: config/config.json) |
| `--input` | `-i` | 入力画像ディレクトリ (必須) |
| `--output` | `-o` | 出力ディレクトリ |
| `--profile` | `-p` | 適用するカメラプロファイル名 (必須) |
| `--no-save-original` | | 元画像の保存をスキップ |
| `--list-profiles` | | 利用可能な全てのカメラプロファイルを表示 |

### `pochi aggregate` - 画像集約

複数ディレクトリから画像を1つのディレクトリに集約します.

```bash
# 画像をコピーして集約
uv run pochi aggregate --input ./data

# 画像を移動して集約
uv run pochi aggregate --input ./data --mode move
```

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--input` | `-i` | 入力ディレクトリ (必須) |
| `--mode` | `-m` | 集約モード: `copy` または `move` (デフォルト: copy) |

### `pochi fft` - FFT ビジュアライザー

画像の FFT (高速フーリエ変換) スペクトルを可視化します.

```bash
uv run pochi fft --input image.png
```

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--input` | `-i` | 入力画像パス (必須) |

## 設定

アプリケーションは JSON 設定ファイルを使用して, カメラプロファイル, 処理パイプライン, 録画, プレビュー設定を定義します.

### 設定ファイル構造

```json
{
  "cameras": {
    "0": {
      "width": 3200,
      "height": 2400,
      "fps": 30,
      "backend": "DSHOW",
      "label": "Tokyo_Lab",
      "processors": ["resize", "gaussian_blur", "std_bin", "contour", "mask_composition"],
      "mode": "pipeline",
      "id_interval": 4,
      "gaussian_blur": { "kernel_size": [19, 19], "sigma": 0 },
      "std_bin": { "threshold": 20 },
      "resize": { "width": 1600, "height": 1200, "preserve_aspect_ratio": true, "aspect_ratio_mode": "width" },
      "contour": { "retrieval_mode": "list", "approximation_method": "simple", "min_area": 100 },
      "mask_composition": { "target_image": "original", "use_white_pixels": true, "enable_cropping": true }
    }
  },
  "recording": {
    "select_format": "mjpg"
  },
  "preview": {
    "width": 1280,
    "height": 720
  },
  "selected_camera_index": 0,
  "id_interval": 1
}
```

### トップレベル設定

| キー | 必須 | デフォルト | 説明 |
|-----|------|---------|------|
| `cameras` | Yes | - | カメラプロファイル定義 |
| `selected_camera_index` | Yes | - | CLI 未指定時に使用するカメラインデックス |
| `id_interval` | No | 1 | グローバルキャプチャ ID 間隔 |
| `recording.select_format` | No | `"mjpg"` | 録画コーデック (`mp4v`, `xvid`, `mjpg`, `ffv1` 等) |
| `preview.width` | No | 1280 | プレビューウィンドウ幅 |
| `preview.height` | No | 720 | プレビューウィンドウ高さ |

### カメラプロファイル設定

| キー | 必須 | デフォルト | 説明 |
|-----|------|---------|------|
| `processors` | Yes | - | プロセッサ名の配列 (空不可) |
| `width` | No | 640 | カメラ解像度の幅 |
| `height` | No | 480 | カメラ解像度の高さ |
| `fps` | No | 30 | フレームレート |
| `backend` | No | none | カメラバックエンド (`DSHOW`, `MSMF`, `V4L2` 等) |
| `mode` | No | `"parallel"` | `"parallel"` または `"pipeline"` |
| `label` | No | none | 出力ファイル名用カスタムラベル |
| `id_interval` | No | 1 | プロファイル別キャプチャ ID 間隔 |

### カメラプロファイルの注意点

- プロファイル名が数字の場合 (例: "0"), コマンドラインの `--camera` 引数と一致する必要があります
- コマンドラインで `--camera` のみ指定した場合, プロファイル "0" を使用します
- カメラプロファイルごとに異なるプロセッサと設定を指定できます
- 登録されていないプロセッサを指定するとエラーになります

## 利用可能なプロセッサ

| # | 名前 | 説明 | 主要パラメータ |
|---|------|------|----------------|
| 1 | `grayscale` | グレースケール変換 | なし |
| 2 | `gaussian_blur` | ガウシアンぼかし | `kernel_size`, `sigma` |
| 3 | `average_blur` | 平均値ブラー | `kernel_size` |
| 4 | `median_blur` | メディアンブラー | `kernel_size` (奇数) |
| 5 | `bilateral_filter` | エッジ保持ぼかし | `d`, `sigmaColor`, `sigmaSpace` |
| 6 | `motion_blur` | モーションブラー | `kernel_size` (奇数), `angle` |
| 7 | `std_bin` | 標準2値化 | `threshold` |
| 8 | `otsu_bin` | 大津の2値化 | なし |
| 9 | `gauss_adapt_bin` | ガウス適応的2値化 | `block_size`, `c` |
| 10 | `mean_adapt_bin` | 平均適応的2値化 | `block_size`, `c` |
| 11 | `resize` | リサイズ | `width`, `height`, `preserve_aspect_ratio`, `aspect_ratio_mode` |
| 12 | `canny_edge` | Canny エッジ検出 | `threshold1`, `threshold2`, `aperture_size` |
| 13 | `contour` | 輪郭検出 | `retrieval_mode`, `min_area`, `select_mode`, `contour_rank` |
| 14 | `clahe` | CLAHE コントラスト強調 | `clip_limit`, `tile_grid_size`, `color_mode` |
| 15 | `equalize` | ヒストグラム平坦化 | `color_mode` |
| 16 | `mask_composition` | マスクとソース画像の合成 | `target_image`, `use_white_pixels`, `enable_cropping` |

## 利用可能な Feature Extractor

処理済み画像を解析し, 下流の AI タスク向けに数値特徴量を出力します.

| # | 名前 | 説明 | ガイド |
|---|------|------|--------|
| 1 | `rgb` | RGB チャンネル統計量 (平均, 標準偏差等) | |
| 2 | `hsv` | HSV チャンネル統計量 | |
| 3 | `brightness` | 輝度統計量 | |
| 4 | `glcm` | GLCM テクスチャ特徴量 (コントラスト, エネルギー等) | [docs/glcm_features.md](docs/glcm_features.md) |
| 5 | `hlac` | HLAC テクスチャ特徴量 | [docs/hlac_features.md](docs/hlac_features.md) |
| 6 | `lbp` | LBP テクスチャ特徴量 | [docs/lbp_features.md](docs/lbp_features.md) |
| 7 | `fft` | FFT 周波数特徴量 | [docs/fft_features.md](docs/fft_features.md) |
| 8 | `swt` | SWT 周波数特徴量 | [docs/swt_features.md](docs/swt_features.md) |
| 9 | `circle_counter` | 円検出・カウント | |

## ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています - 詳細は LICENSE ファイルを参照してください.
