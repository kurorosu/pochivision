# Pochivision

プラグイン方式でプロセッサを拡張可能な AI ビジョンアプリケーション向けリアルタイム画像キャプチャ・前処理エンジン.

## 特徴

- **複数カメラ対応**: 設定可能なカメラプロファイルで複数デバイスをサポート.
- **リアルタイム処理**: ライブプレビューに対する画像キャプチャと処理.
- **2 つの実行モード**: `pipeline` / `parallel` を JSON 設定で切替.
- **拡張可能なレジストリ**: `@register_processor` / `@register_feature_extractor` でプラグインを動的登録.
- **特徴量抽出**: RGB / HSV / 輝度 / GLCM / HLAC / LBP / FFT / SWT / 円検出など.
- **検出 API 連携**: pochidetection WebAPI と連携した常時検出オーバーレイ (`--detect`).
- **推論 API 連携**: pochitrain 推論 API と連携したリアルタイム分類 (`i` キートグル).
- **録画機能**: コーデック選択可能な録画 (mp4v / xvid / mjpg / ffv1 等).

## ドキュメントの構成

- [はじめに](getting-started/installation.md): インストールとクイックスタート.

## クイックリンク

- [インストール手順](getting-started/installation.md)
- [最短実行フロー](getting-started/quickstart.md)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています.
