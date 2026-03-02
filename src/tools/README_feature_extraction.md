# 特徴量抽出システム

このシステムは、指定されたフォルダ内のすべての画像から特徴量を抽出し、CSV形式で保存するツールです。

## ファイル構成

```
vision-capture-core/
├── feature_extractors/         # 特徴量抽出器パッケージ
│   ├── base.py                 # 基底クラス
│   ├── brightness_statistics.py # 輝度統計特徴量抽出器
│   ├── registry.py             # レジストリシステム
│   └── schema.py               # スキーマ定義
├── tools/                      # ツール・ユーティリティスクリプト
│   ├── feature_extraction.py   # 特徴量抽出エントリーポイント
│   ├── image_aggregator.py     # 画像集約ツール
│   └── cameratest.py           # カメラテストツール
├── extractor_config.json       # 特徴量抽出設定ファイル
└── README_feature_extraction.md # このファイル
```

## 使用方法

### 1. 入力画像の準備

特徴量を抽出したい画像を指定のフォルダに配置してください。

```bash
mkdir input_images
# 画像ファイルをinput_imagesフォルダにコピー
```

### 2. 設定ファイルの調整（オプション）

`extractor_config.json`で設定をカスタマイズできます：

```json
{
  "input_directory": "input_images",          // 入力フォルダ
  "output_directory": "extraction_results",   // 出力フォルダ
  "output_format": "csv",                     // 出力形式
  "feature_extractors": {
    "brightness": {
      "color_mode": "gray"                    // グレースケール、lab_l、hsv_v
    }
  },
  "file_filters": {
    "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
    "case_sensitive": false
  },
  "output_settings": {
    "include_filename": true,
    "include_timestamp": true,
    "csv_separator": ",",
    "output_filename": "brightness_features.csv"
  }
}
```

### 3. 特徴量抽出の実行

```bash
# デフォルト設定で実行
python -m tools.feature_extraction

# カスタム設定ファイルを指定して実行
python -m tools.feature_extraction --config my_config.json
```

### 4. 結果の確認

`extraction_results/brightness_features.csv`に以下の形式で結果が保存されます：

```csv
filename,timestamp,brightness_mean,brightness_median,brightness_variance,brightness_std_dev,brightness_cv
image1.jpg,2024-01-01T12:00:00,127.5,128.0,1024.5,32.0,0.251
image2.jpg,2024-01-01T12:00:01,98.3,95.0,890.2,29.8,0.303
```

## 特徴量の説明

### 輝度統計特徴量 (brightness)

- **mean**: 輝度平均値（背景ピクセル除外）
- **median**: 輝度中央値（背景ピクセル除外）
- **variance**: 輝度分散（背景ピクセル除外）
- **std_dev**: 輝度標準偏差（背景ピクセル除外）
- **cv**: 変動係数（標準偏差/平均値）

### 色空間オプション

- **gray**: BGRからグレースケールに変換
- **lab_l**: LAB色空間のL成分（明度）を使用
- **hsv_v**: HSV色空間のV成分（明度）を使用

## カスタマイズ

### 新しい特徴量抽出器の追加

1. `feature_extractors/`フォルダに新しい抽出器クラスを作成
2. `BaseFeatureExtractor`を継承
3. `@register_feature_extractor`デコレータで登録
4. 設定ファイルに追加

例：
```python
@register_feature_extractor("texture_features")
class TextureExtractor(BaseFeatureExtractor):
    def extract(self, image):
        # テクスチャ特徴量の抽出処理
        return {"contrast": 0.5, "homogeneity": 0.8}

    @staticmethod
    def get_default_config():
        return {"window_size": 7}
```

## エラー対処

### よくあるエラー

1. **入力ディレクトリが存在しません**
   - `input_images`フォルダを作成してください

2. **対象画像ファイルが見つかりません**
   - ファイル拡張子を確認してください
   - `extractor_config.json`の`extensions`設定を確認してください

3. **特徴量抽出器の初期化に失敗しました**
   - `feature_extractors/`パッケージが正しくインポートできるか確認してください
   - 設定ファイルの形式を確認してください

## 実行例

```bash
# 基本的な実行
python -m tools.feature_extraction

# 出力例:
# === 特徴量抽出を開始します ===
# 入力ディレクトリ: input_images
# 出力ディレクトリ: extraction_results
# 特徴量抽出器を初期化しました: brightness
# 対象画像ファイル数: 10
# 処理中 (1/10): image1.jpg
# 処理中 (2/10): image2.jpg
# ...
# 特徴量抽出結果を保存しました: extraction_results/brightness_features.csv
# 処理された画像数: 10
# === 特徴量抽出が完了しました ===
```

## 画像集約ツールの実行例

```bash
# 画像集約ツールの実行
python -m tools.image_aggregator -i ./capture/camera1 -m copy
```

## 技術仕様

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Pydantic（スキーマ検証用）

## ライセンス

本ツールは「vision-capture-core」プロジェクトのライセンスに従います。
