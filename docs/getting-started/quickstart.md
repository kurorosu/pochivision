# クイックスタート

最速でライブプレビューと特徴量抽出までを試すためのシンプルなガイドです.

## 1. ライブプレビューの起動

デフォルト設定 (`config/config.json` の `selected_camera_index` で指定されたカメラ) でアプリケーションを起動します.

```bash
uv run pochi run
```

特定のカメラデバイスとプロファイルを指定:

```bash
uv run pochi run --camera 2 --profile "high_fps"
```

利用可能なカメラプロファイル一覧:

```bash
uv run pochi run --list-profiles
```

録画機能を無効にして起動:

```bash
uv run pochi run --no-recording
```

## 2. 検出モードの利用 (任意)

pochidetection WebAPI と連携する常時検出ランタイムを有効化します. 検出結果は bbox + メタ情報としてプレビューに描画されます.

```bash
uv run pochi run --detect
```

設定ファイルは `config/detect_config.json` がデフォルトです. `i` キーで検出 ON/OFF を切替できます.

## 3. 推論モードの利用 (任意)

pochitrain 推論 API と連携した分類を行います. `i` キーで推論実行 (classify モード).

```bash
uv run pochi run
```

推論設定ファイルは `config/infer_config.json` がデフォルトです.

## 4. 特徴量抽出

画像から特徴量を抽出し CSV に出力します.

```bash
uv run pochi extract
```

設定ファイル (デフォルト: `config/extractor_config.json`) を指定する場合:

```bash
uv run pochi extract --config my_extractor_config.json
```

## 5. プロファイル適用

カメラプロファイルの処理パイプラインを既存画像群に適用します.

```bash
uv run pochi process --input ./images --profile "high_res"
```

## 6. 画像集約 / FFT 可視化

複数ディレクトリの画像を 1 つに集約:

```bash
uv run pochi aggregate --input ./data
```

画像の FFT スペクトル可視化:

```bash
uv run pochi fft --input image.png
```

## 7. 次のステップ

- 各サブコマンドの詳細・設定ファイル構造・プロセッサ / 特徴量一覧は [ユーザーガイド](../user-guide/run.md) を参照してください.
- 設定ファイル全体の構造は [設定ファイル](../user-guide/configuration.md) を参照してください.
- 特徴量の詳細は [利用可能な Feature Extractor](../user-guide/feature-extractors.md) と特徴量リファレンス ([FFT](../user-guide/features/fft.md) / [GLCM](../user-guide/features/glcm.md) / [HLAC](../user-guide/features/hlac.md) / [LBP](../user-guide/features/lbp.md) / [SWT](../user-guide/features/swt.md)) を参照してください.
