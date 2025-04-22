# Vision Capture Core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AIビジョンアプリケーション向けのリアルタイム画像キャプチャ・前処理エンジン。

## 機能

- 設定可能なプロファイルを持つ複数カメラサポート
- リアルタイム画像キャプチャと処理
- 画像処理のためのパイプラインアーキテクチャ
- 拡張可能なプロセッサシステム
- 簡単な設定のためのコマンドラインインターフェース

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/vision-capture-core.git
cd vision-capture-core

# 依存関係のインストール
pip install -r requirements.txt
```

## クイックスタート

デフォルト設定でアプリケーションを実行:

```bash
python app.py
```

## コマンドラインインターフェース

Vision Capture Coreは柔軟なコマンドラインインターフェースを提供しています:

```bash
# 特定のカメラデバイスを使用（インデックスで指定）
# カメラ1をプロファイル"0"で使用
python app.py --camera 1

# 設定から特定のカメラプロファイルを使用
python app.py --profile "high_res"

# 特定のカメラとプロファイルの両方を使用
python app.py --camera 2 --profile "high_fps"

# 利用可能なすべてのカメラプロファイルを一覧表示
python app.py --list-profiles

# 代替設定ファイルを使用
python app.py --config "my_config.json"
```

### CLI引数

| 引数 | 短縮形 | 説明 |
|----------|-------|-------------|
| `--camera` | `-c` | カメラデバイスインデックスを指定（0, 1, 2...）。単独で使用する場合、常にプロファイル"0"を使用 |
| `--profile` | `-p` | config.jsonからカメラプロファイルを指定 |
| `--list-profiles` | `-l` | 利用可能なすべてのカメラプロファイルを表示 |
| `--config` | | 設定ファイルのパスを指定（デフォルト: config.json） |

## 設定

アプリケーションはJSON設定ファイルを使用して、カメラプロファイルと処理パイプラインを定義します。各カメラプロファイルは独立した設定を持ち、異なる画像処理とパラメータを指定できます。

### 設定ファイル構造

```json
{
  "cameras": {
    "0": {
      "width": 3200,
      "height": 2400,
      "fps": 30,
      "backend": "DSHOW",
      "processors": ["gaussian_blur", "grayscale", "standard_binarization"],
      "mode": "parallel",
      "gaussian_blur": {
        "kernel_size": [15, 15],
        "sigma": 0
      },
      "grayscale": {},
      "standard_binarization": {
        "threshold": 128
      }
    },
    "high_res": {
      "width": 3840,
      "height": 2160,
      "fps": 30,
      "backend": "DSHOW",
      "processors": ["gaussian_blur"],
      "mode": "parallel",
      "gaussian_blur": {
        "kernel_size": [31, 31],
        "sigma": 0
      }
    },
    "high_fps": {
      "width": 1280,
      "height": 720,
      "fps": 60,
      "backend": "DSHOW",
      "processors": ["grayscale"],
      "mode": "pipeline",
      "grayscale": {}
    }
  },
  "selected_camera_index": 0
}
```

### 必須項目とオプション項目

**グローバル設定（必須）**:
- `cameras`: カメラプロファイルを定義するオブジェクト（必須）
- `selected_camera_index`: コマンドラインで指定がない場合に使用されるカメラインデックス（必須）

**各カメラプロファイル（`cameras`内）**:

必須項目:
- `processors`: 使用するプロセッサ名の配列（必須、空の場合はエラー）

オプション項目と省略時のデフォルト値:
- `width`: カメラ解像度の幅（オプション、デフォルト: 640）
- `height`: カメラ解像度の高さ（オプション、デフォルト: 480）
- `fps`: フレームレート（オプション、デフォルト: 30）
- `backend`: カメラバックエンドの種類（オプション、デフォルト: なし）
- `mode`: 処理モード（オプション、デフォルト: "parallel"）
  - `"parallel"`: 各プロセッサが独立して元画像を処理
  - `"pipeline"`: 各プロセッサが前のプロセッサの出力を受け取る
- 各プロセッサの設定: プロセッサ名をキーとしたオブジェクト（オプション、デフォルト: 空のオブジェクト）

**設定例**:
- `gaussian_blur`: ガウシアンぼかし処理のパラメータ
  - `kernel_size`: カーネルサイズ（例: [15, 15]）
  - `sigma`: ガウスぼかしのシグマ値（例: 0）

### カメラプロファイルの注意点

- プロファイル名が数字の場合（例: "0"）、コマンドラインの `--camera` 引数と一致する必要があります
- コマンドラインで `--camera` のみ指定した場合、プロファイル "0" を使用します
- カメラプロファイルごとに異なるプロセッサと設定を指定できます
- 各プロファイルの `processors` リストは必須で、空であってはなりません
- 登録されていないプロセッサを指定するとエラーになります

## アーキテクチャ

Vision Capture Core はSOLID原則に従ったモジュール式アーキテクチャを採用しています：

- **Core**: 中央パイプライン実行
- **CaptureLib**: カメラとシステム管理
- **Processors**: 画像処理モジュール
- **Capture Runner**: UIとアプリケーション制御

## 利用可能なプロセッサ

現在以下のプロセッサが利用可能です：

1. **grayscale**: カラー画像をグレースケールに変換
   - パラメータ: なし
   ```json
   "grayscale": {}
   ```

2. **gaussian_blur**: ガウシアンぼかしを適用
   - パラメータ:
     - `kernel_size`: ぼかしのカーネルサイズ（例: [15, 15]）
     - `sigma`: ガウスぼかしのシグマ値（例: 0）
   ```json
   "gaussian_blur": {
     "kernel_size": [15, 15],
     "sigma": 0
   }
   ```

3. **standard_binarization**: スタンダードな2値化（しきい値による通常の2値化）を適用
   - パラメータ:
     - `threshold`: 2値化の閾値（例: 128）
   ```json
   "standard_binarization": {
     "threshold": 128
   }
   ```

4. **median_blur**: メディアンブラーを適用
   - パラメータ:
     - `kernel_size`: カーネルサイズ（奇数、例: 5）
   ```json
   "median_blur": {
     "kernel_size": 5
   }
   ```

5. **bilateral_filter**: バイラテラルフィルタ（エッジを保つ高品質なぼかし）を適用
   - パラメータ:
     - `d`: 近傍領域の直径（例: 9）
     - `sigmaColor`: 色空間のシグマ値（例: 75）
     - `sigmaSpace`: 座標空間のシグマ値（例: 75）
   ```json
   "bilateral_filter": {
     "d": 9,
     "sigmaColor": 75,
     "sigmaSpace": 75
   }
   ```

6. **motion_blur**: モーションブラー（直線的な動きのぼかし）を適用
   - パラメータ:
     - `kernel_size`: ブラーの長さ（奇数、例: 15）
     - `angle`: ブラーの角度（度、0-359、例: 0）
   ```json
   "motion_blur": {
     "kernel_size": 15,
     "angle": 0
   }
   ```

## 貢献

貢献は歓迎します！Pull Requestを自由に提出してください。

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細はLICENSEファイルを参照してください。 