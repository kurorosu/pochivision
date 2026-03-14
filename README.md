# Vision Capture Core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## ディレクトリ構成

```
vision-capture-core/
├── src/
│   ├── cli/                  # CLI エントリーポイント (vcc コマンド)
│   ├── capture_runner/       # カメラキャプチャとプレビュー実行
│   ├── capturelib/           # カメラセットアップ, 設定, ログ, 録画
│   ├── core/                 # パイプラインエグゼキュータ
│   ├── exceptions/           # カスタム例外クラス
│   ├── feature_extractors/   # 特徴量抽出プラグイン
│   ├── processors/           # 画像処理パイプライン
│   ├── tools/                # ユーティリティスクリプト
│   └── utils/                # 共通ユーティリティ
├── tests/                    # テストスイート
├── config.json               # アプリケーション設定
└── pyproject.toml            # プロジェクトメタデータ・依存関係
```

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/kurorosu/vision-capture-core.git
cd vision-capture-core

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
uv run vcc
```

## コマンドラインインターフェース

Vision Capture Core は `vcc` コマンドによる柔軟な CLI を提供しています:

```bash
# 特定のカメラデバイスを使用 (インデックスで指定)
uv run vcc --camera 1

# 設定から特定のカメラプロファイルを使用
uv run vcc --profile "high_res"

# 特定のカメラとプロファイルの両方を使用
uv run vcc --camera 2 --profile "high_fps"

# 利用可能な全てのカメラプロファイルを一覧表示
uv run vcc --list-profiles

# 代替設定ファイルを使用
uv run vcc --config "my_config.json"

# 録画機能を無効にして起動
uv run vcc --no-recording
```

### CLI 引数

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--camera` | `-c` | カメラデバイスインデックスを指定 (0, 1, 2...). 単独使用時はプロファイル "0" を使用 |
| `--profile` | `-p` | config.json からカメラプロファイルを指定 |
| `--list-profiles` | `-l` | 利用可能な全てのカメラプロファイルを表示 |
| `--config` | | 設定ファイルのパスを指定 (デフォルト: config.json) |
| `--no-recording` | | 録画機能を無効にして起動 |

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

## アーキテクチャ

Vision Capture Core は SOLID 原則に従ったモジュール式アーキテクチャを採用しています:

- **CLI**: コマンドラインエントリーポイント (`vcc` コマンド)
- **Core**: パイプライン実行 (pipeline / parallel モード)
- **CaptureLib**: カメラセットアップ, 設定管理, ログ, 録画
- **Processors**: 画像処理モジュール (レジストリパターン)
- **Feature Extractors**: 特徴量抽出プラグイン (レジストリパターン)
- **Capture Runner**: ライブプレビューとアプリケーション制御

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

| # | 名前 | 説明 |
|---|------|------|
| 1 | `rgb` | RGB チャンネル統計量 (平均, 標準偏差等) |
| 2 | `hsv` | HSV チャンネル統計量 |
| 3 | `brightness` | 輝度統計量 |
| 4 | `glcm` | GLCM テクスチャ特徴量 (コントラスト, エネルギー等) |
| 5 | `hlac` | HLAC テクスチャ特徴量 |
| 6 | `lbp` | LBP テクスチャ特徴量 |
| 7 | `fft` | FFT 周波数特徴量 |
| 8 | `swt` | SWT 周波数特徴量 |
| 9 | `circle_counter` | 円検出・カウント |

## ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています - 詳細は LICENSE ファイルを参照してください.
