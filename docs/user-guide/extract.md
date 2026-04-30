# 特徴量抽出 (`pochi extract`)

`pochi extract` は画像ディレクトリ内の各画像に対して特徴量を抽出し, CSV ファイルに出力する.

## 1. 基本コマンド

```bash
# デフォルト設定ファイル (config/extractor_config.json) を使用
uv run pochi extract

# 設定ファイルを指定
uv run pochi extract --config my_extractor_config.json
```

## 2. 引数

| 引数 | 短縮形 | 説明 |
|------|--------|------|
| `--config` | `-c` | 設定ファイルのパス (デフォルト: `config/extractor_config.json`) |

## 3. 設定ファイル (`extractor_config.json`)

抽出対象ディレクトリ, 出力先, 適用する Feature Extractor 名と各パラメータを JSON で定義する.

```json
{
  "input_dir": "capture/20260101_120000",
  "output_csv": "features.csv",
  "extractors": ["rgb", "hsv", "glcm"],
  "glcm": { "distances": [1], "angles": [0, 0.785, 1.571], "levels": 16 }
}
```

## 4. 利用可能な Feature Extractor

利用可能な抽出器の一覧と特徴量解説は [Feature Extractor 一覧](feature-extractors.md) を参照.

## 5. 出力 CSV

| カラム | 説明 |
|--------|------|
| `filename` | 入力画像のファイル名 |
| `<extractor>_<feature>` | Extractor ごとに動的に生成される特徴量カラム |
