# pochivision

[![Version](https://img.shields.io/badge/version-0.7.0-green.svg)](https://github.com/kurorosu/pochivision/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

[English](README.en.md)

プラグイン方式でプロセッサを拡張可能な AI ビジョンアプリケーション向けリアルタイム画像キャプチャ・前処理エンジン.

> 詳細ドキュメントは `uv run mkdocs serve` でローカル閲覧してください. 各機能の詳細は [docs/user-guide/](docs/user-guide/) を参照してください.

## 特徴

- **複数カメラ対応**: 設定可能なカメラプロファイルで複数デバイスをサポート
- **リアルタイム処理**: ライブプレビューに対する画像キャプチャと処理
- **2 つの実行モード**: `pipeline` / `parallel` を JSON 設定で切替
- **拡張可能なレジストリ**: `@register_processor` / `@register_feature_extractor` でプラグインを動的登録
- **特徴量抽出**: RGB / HSV / 輝度 / GLCM / HLAC / LBP / FFT / SWT / 円検出など
- **検出 API 連携**: pochidetection WebAPI と連携した常時検出オーバーレイ (`--detect`)
- **推論 API 連携**: pochitrain 推論 API と連携したリアルタイム分類 (`i` キートグル)
- **録画機能**: コーデック選択可能な録画 (mp4v / xvid / mjpg / ffv1 等)

## クイックスタート

### 1. インストール

```bash
git clone https://github.com/kurorosu/pochivision.git
cd pochivision
uv sync
```

開発・テスト・Lint・ドキュメント (mkdocs) も含めてインストールする場合は `uv sync --group dev`.

### 2. ライブプレビューの起動

```bash
# デフォルト設定で起動
uv run pochi run

# カメラインデックスとプロファイルを指定
uv run pochi run --camera 2 --profile high_fps

# 利用可能な全カメラプロファイルを一覧表示
uv run pochi run --list-profiles
```

### 3. 主なサブコマンド

```bash
uv run pochi run        # ライブプレビュー
uv run pochi extract    # 特徴量抽出 (CSV 出力)
uv run pochi process    # 既存画像にプロファイル適用
uv run pochi aggregate  # ディレクトリ集約
uv run pochi fft        # FFT ビジュアライザー
```

詳細は以下のユーザーガイドを参照してください.

## ドキュメント

- [はじめに](docs/index.md) — プロジェクト概要.
- [インストール](docs/getting-started/installation.md) / [クイックスタート](docs/getting-started/quickstart.md)
- ユーザーガイド:
    - [ライブプレビュー (`pochi run`)](docs/user-guide/run.md)
    - [特徴量抽出 (`pochi extract`)](docs/user-guide/extract.md)
    - [プロファイル適用 (`pochi process`)](docs/user-guide/process.md)
    - [画像集約 (`pochi aggregate`)](docs/user-guide/aggregate.md)
    - [FFT ビジュアライザー (`pochi fft`)](docs/user-guide/fft.md)
    - [検出モード (`--detect`)](docs/user-guide/detection.md)
    - [推論モード](docs/user-guide/inference.md)
    - [設定ファイル](docs/user-guide/configuration.md)
    - [利用可能なプロセッサ](docs/user-guide/processors.md)
    - [利用可能な Feature Extractor](docs/user-guide/feature-extractors.md)
- 特徴量リファレンス:
    - [FFT](docs/user-guide/features/fft.md) / [GLCM](docs/user-guide/features/glcm.md) / [HLAC](docs/user-guide/features/hlac.md) / [LBP](docs/user-guide/features/lbp.md) / [SWT](docs/user-guide/features/swt.md)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています.
