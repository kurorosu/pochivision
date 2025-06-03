# Histogram Generator

CSVファイルから特徴量のヒストグラムを生成するツールです。class列による分類で色分けしたヒストグラムを作成し、CSVファイルと同じフォルダの`hist`サブフォルダに保存します。

## 機能

- CSVファイルから数値データを自動検出
- class列による分類でヒストグラムを色分け
- 適切なビン数の自動計算
- 統計情報の表示
- 高解像度PNG画像での保存
- サマリーレポートの生成

## 使用方法

### 基本的な使用方法

```bash
python -m tools.histogram_generator path/to/your/features.csv
```

### クラス列名を指定する場合

```bash
python -m tools.histogram_generator path/to/your/features.csv --class-column category
```

## 引数

- `csv_path` (必須): CSVファイルのパス
- `--class-column` (オプション): クラス分類に使用する列名（デフォルト: "class"）

## 入力CSVファイルの要件

### 必要な列

- 数値データを含む列（1列以上）
- クラス分類用の列（オプション、デフォルトは"class"）

### 除外される列

以下の列は自動的にヒストグラム生成から除外されます：
- `filename`: ファイル名列
- `timestamp`: タイムスタンプ列
- クラス分類用の列（デフォルトは`class`）

### CSVファイルの例

```csv
filename,class,feature1,feature2,feature3,timestamp
image1.jpg,cat,0.123,45.67,0.891,2024-01-01 10:00:00
image2.jpg,dog,0.456,23.45,0.234,2024-01-01 10:01:00
image3.jpg,cat,0.789,67.89,0.567,2024-01-01 10:02:00
```

## 出力

### 出力ディレクトリ

CSVファイルと同じディレクトリに`hist`フォルダが作成されます。

```
your_csv_directory/
├── features.csv
└── hist/
    ├── feature1_histogram.png
    ├── feature2_histogram.png
    ├── feature3_histogram.png
    └── histogram_summary.txt
```

### ヒストグラム画像

- **ファイル名**: `{列名}_histogram.png`
- **解像度**: 300 DPI
- **サイズ**: 12×8インチ
- **形式**: PNG

### ヒストグラムの特徴

- **色分け**: クラスごとに異なる色で表示
- **透明度**: 重なりが見やすいように半透明（alpha=0.7）
- **凡例**: 各クラス名とサンプル数を表示
- **統計情報**: 平均、標準偏差、最小値、最大値を表示
- **グリッド**: 読みやすさのための薄いグリッド線

### サマリーレポート

`histogram_summary.txt`ファイルには以下の情報が含まれます：

- CSVファイルの基本情報
- データ形状とクラス分布
- 各数値列の詳細統計情報

## 使用例

### 例1: 基本的な使用

```bash
python -m tools.histogram_generator extraction_results/20241201_0/features.csv
```

### 例2: カスタムクラス列名

```bash
python -m tools.histogram_generator data/results.csv --class-column category
```

### 例3: 特徴量抽出結果の可視化

```bash
# 特徴量抽出を実行
python -m tools.feature_extraction

# 抽出結果のヒストグラムを生成
python -m tools.histogram_generator extraction_results/20241201_0/features.csv
```

## エラー処理

### よくあるエラーと対処法

1. **CSVファイルが見つからない**
   ```
   FileNotFoundError: CSVファイルが見つかりません: path/to/file.csv
   ```
   - ファイルパスが正しいか確認してください

2. **数値データの列が見つからない**
   ```
   ValueError: 数値データの列が見つかりません
   ```
   - CSVファイルに数値データが含まれているか確認してください

3. **クラス列が見つからない**
   ```
   警告: クラス列 'class' が見つかりません
   ```
   - 自動的に全データを一つのクラスとして処理されます
   - `--class-column`オプションで正しい列名を指定してください

## 技術仕様

### 依存ライブラリ

- `pandas`: CSVファイルの読み込みとデータ処理
- `matplotlib`: ヒストグラムの描画
- `seaborn`: カラーパレットの生成
- `numpy`: 数値計算

### ビン数の計算

ヒストグラムのビン数は以下の方法で自動計算されます：

1. **Sturgesの公式**: `ceil(log2(n) + 1)`
2. **Freedman-Diaconisルール**: `(max - min) / (2 * IQR * n^(-1/3))`
3. **最終値**: 上記2つの平均値（最小10、最大50で制限）

### カラーパレット

- **10クラス以下**: seabornの`tab10`パレット
- **10クラス超**: seabornの`hsl`パレット

## 注意事項

- 大量のデータ（数万行以上）の場合、処理に時間がかかる場合があります
- メモリ使用量はCSVファイルのサイズに比例します
- 特殊文字を含む列名は、ファイル名で安全な文字に変換されます

## トラブルシューティング

### メモリ不足エラー

大きなCSVファイルでメモリ不足が発生する場合：

1. 不要な列を事前に削除
2. データをサンプリングして小さくする
3. チャンクごとに処理する（将来の機能拡張予定）

### 文字化け

日本語を含むCSVファイルで文字化けが発生する場合：

1. CSVファイルがUTF-8エンコーディングで保存されているか確認
2. Excelで保存する場合は「UTF-8 CSV」形式を選択

## 実行例

```bash
# 基本的な実行
python -m tools.histogram_generator extraction_results/20241201_0/features.csv

# 出力例:
# === CSVヒストグラム生成を開始します ===
# CSVファイルを読み込みました: extraction_results/20241201_0/features.csv
# データ形状: (100, 15)
# 数値列を特定しました: 12列
# クラス数: 3
# 出力ディレクトリを作成しました: extraction_results/20241201_0/hist
# === 12個のヒストグラムを生成します ===
# 進行状況 (1/12): brightness_mean
# ヒストグラムを保存しました: extraction_results/20241201_0/hist/brightness_mean_histogram.png
# ...
# サマリーレポートを保存しました: extraction_results/20241201_0/hist/histogram_summary.txt
# === CSVヒストグラム生成が完了しました ===
```

## 更新履歴

- v1.0.0: 初回リリース
  - 基本的なヒストグラム生成機能
  - クラス別色分け機能
  - サマリーレポート生成機能 