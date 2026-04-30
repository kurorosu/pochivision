# 推論モード (推論 API 連携)

`pochi run` は pochitrain 推論 API と連携し, 起動中に `i` キーを押すたびに 1 回推論を実行する (classify モード). 検出モード (`--detect`) と異なり常時送信ではなくユーザートリガー方式.

## 1. 基本コマンド

```bash
# 推論設定ファイル (config/infer_config.json) を読み込んで起動
uv run pochi run

# 推論設定ファイルを明示的に指定
uv run pochi run --infer-config config/infer_config.json
```

## 2. キー操作

| キー | 説明 |
|------|------|
| `i` | 推論を 1 回実行. レスポンスの分類結果を overlay に表示 |

## 3. 設定ファイル (`infer_config.json`)

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `base_url` | Yes | - | pochitrain 推論 API のベース URL |
| `image_format` | No | `"jpeg"` | 画像送信形式 (`"raw"` / `"jpeg"`) |
| `resize.width` | No | なし (リサイズなし) | 送信画像の幅 |
| `resize.height` | No | なし (リサイズなし) | 送信画像の高さ |
| `resize.padding_color` | No | `[0, 0, 0]` | パディング色 (BGR) |
| `save_frame` | No | `false` | 推論実行時にフレーム画像を保存するか |
| `save_csv` | No | `false` | 推論結果を CSV ファイルに出力するか |

## 4. 出力

- `save_frame=true`: `capture/<timestamp>/infer_frames/` に送信フレームを保存.
- `save_csv=true`: `capture/<timestamp>/infer_results.csv` にタイムスタンプ・分類ラベル・スコアを記録.

## 5. detect モードとの関係

`--detect` を指定すると pochidetection 検出 API との常時検出モードに切り替わり, `i` キーは「検出 ON/OFF トグル」として再割当てされる. 詳細は [検出モード](detection.md) を参照.
