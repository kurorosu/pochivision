# プロファイル適用 (`pochi process`)

`pochi process` はカメラプロファイルに紐づく処理パイプラインを既存の画像群に適用する. ライブ運用時の前処理結果をオフラインで再現したいときに使用する.

## 1. 基本コマンド

```bash
# 入力ディレクトリの画像にプロファイルを適用
uv run pochi process --input ./images --profile high_res

# 出力先を指定し, 元画像の保存をスキップ
uv run pochi process --input ./images --output ./processed --profile high_res --no-save-original

# 利用可能なプロファイルを一覧表示
uv run pochi process --list-profiles
```

## 2. 引数

| 引数 | 短縮形 | 説明 |
|------|--------|------|
| `--config` | `-c` | 設定ファイルのパス (デフォルト: `config/config.json`) |
| `--input` | `-i` | 入力画像ディレクトリ (必須) |
| `--output` | `-o` | 出力ディレクトリ |
| `--profile` | `-p` | 適用するカメラプロファイル名 (必須) |
| `--no-save-original` | | 元画像の保存をスキップ |
| `--list-profiles` | | 利用可能な全カメラプロファイルを表示 |

## 3. 動作

1. `--profile` で指定されたカメラプロファイルの `processors` 配列を読み込む.
2. プロファイルの `mode` (`pipeline` / `parallel`) に応じて `PipelineExecutor` を構築する.
3. 入力ディレクトリ内の画像を順次パイプラインに通し, 結果を出力ディレクトリに保存する.

## 4. 出力構成

`--output` 省略時は `processed/<timestamp>/` に保存する. `--no-save-original` を指定しない場合は元画像のコピーも同梱する.
