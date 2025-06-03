# Scatter Plot Generator

CSVファイルから特徴量の散布図を生成するツールです。数値列の全ての組み合わせでclass列による分類で色分けした散布図を作成し、CSVファイルと同じフォルダの`scatter`サブフォルダに保存します。

## 機能

- CSVファイルから数値データを自動検出
- 数値列の全ての組み合わせで散布図を生成（重複なし）
- class列による分類で散布図を色分け
- 回帰直線と相関係数の表示
- 相関行列ヒートマップの生成
- 統計情報の表示
- 高解像度PNG画像での保存
- サマリーレポートの生成

## 使用方法

### 基本的な使用方法

```bash
python -m tools.scatter_plot_generator path/to/your/features.csv
```

### クラス列名を指定する場合

```bash
python -m tools.scatter_plot_generator path/to/your/features.csv --class-column category
```

## 引数

- `csv_path` (必須): CSVファイルのパス
- `--class-column` (オプション): クラス分類に使用する列名（デフォルト: "class"）

## 入力CSVファイルの要件

### 必要な列

- 数値データを含む列（2列以上）
- クラス分類用の列（オプション、デフォルトは"class"）

### 除外される列

以下の列は自動的に散布図生成から除外されます：
- `filename`: ファイル名列
- `timestamp`: タイムスタンプ列
- クラス分類用の列（デフォルトは`class`）

### CSVファイルの例

```csv
filename,class,feature1,feature2,feature3,feature4,timestamp
image1.jpg,cat,0.123,45.67,0.891,12.34,2024-01-01 10:00:00
image2.jpg,dog,0.456,23.45,0.234,56.78,2024-01-01 10:01:00
image3.jpg,cat,0.789,67.89,0.567,90.12,2024-01-01 10:02:00
```

上記の例では、feature1, feature2, feature3, feature4の4つの数値列から以下の6つの散布図が生成されます：
- feature1 vs feature2
- feature1 vs feature3
- feature1 vs feature4
- feature2 vs feature3
- feature2 vs feature4
- feature3 vs feature4

## 出力

### 出力ディレクトリ

CSVファイルと同じディレクトリに`scatter`フォルダが作成されます。

```
your_csv_directory/
├── features.csv
└── scatter/
    ├── feature1_vs_feature2_scatter.png
    ├── feature1_vs_feature3_scatter.png
    ├── feature1_vs_feature4_scatter.png
    ├── feature2_vs_feature3_scatter.png
    ├── feature2_vs_feature4_scatter.png
    ├── feature3_vs_feature4_scatter.png
    ├── correlation_matrix.png
    ├── correlation_matrix.csv
    └── scatter_summary.txt
```

### 散布図画像

- **ファイル名**: `{X軸列名}_vs_{Y軸列名}_scatter.png`
- **解像度**: 300 DPI
- **サイズ**: 12×10インチ
- **形式**: PNG

### 散布図の特徴

- **色分け**: クラスごとに異なる色で表示
- **透明度**: 重なりが見やすいように半透明（alpha=0.7）
- **凡例**: 各クラス名とサンプル数を表示
- **回帰直線**: 全データに対する線形回帰直線（赤い破線）
- **統計情報**: 相関係数、p値、データ範囲を表示
- **グリッド**: 読みやすさのための薄いグリッド線

### 相関行列

- **ヒートマップ**: `correlation_matrix.png`
- **CSV形式**: `correlation_matrix.csv`
- **特徴**: 上三角のみ表示（重複を避けるため）

### サマリーレポート

`scatter_summary.txt`ファイルには以下の情報が含まれます：

- CSVファイルの基本情報
- データ形状とクラス分布
- 数値列一覧
- 散布図組み合わせ一覧（相関係数付き）
- 各数値列の詳細統計情報

## 使用例

### 例1: 基本的な使用

```bash
python -m tools.scatter_plot_generator extraction_results/20241201_0/features.csv
```

### 例2: カスタムクラス列名

```bash
python -m tools.scatter_plot_generator data/results.csv --class-column category
```

### 例3: 特徴量抽出結果の可視化

```bash
# 特徴量抽出を実行
python -m tools.feature_extraction

# 抽出結果の散布図を生成
python -m tools.scatter_plot_generator extraction_results/20241201_0/features.csv
```

### 例4: ヒストグラムと散布図の両方を生成

```bash
# ヒストグラムを生成
python -m tools.histogram_generator extraction_results/20241201_0/features.csv

# 散布図を生成
python -m tools.scatter_plot_generator extraction_results/20241201_0/features.csv
```

## エラー処理

### よくあるエラーと対処法

1. **CSVファイルが見つからない**
   ```
   FileNotFoundError: CSVファイルが見つかりません: path/to/file.csv
   ```
   - ファイルパスが正しいか確認してください

2. **数値列が不足**
   ```
   ValueError: 散布図生成には最低2つの数値列が必要です
   ```
   - CSVファイルに2つ以上の数値データ列が含まれているか確認してください

3. **クラス列が見つからない**
   ```
   警告: クラス列 'class' が見つかりません
   ```
   - 自動的に全データを一つのクラスとして処理されます
   - `--class-column`オプションで正しい列名を指定してください

## 技術仕様

### 依存ライブラリ

- `pandas`: CSVファイルの読み込みとデータ処理
- `matplotlib`: 散布図の描画
- `seaborn`: カラーパレットとヒートマップの生成
- `numpy`: 数値計算
- `scipy`: 統計計算（相関係数、回帰分析）

### 組み合わせ計算

n個の数値列がある場合、生成される散布図の数は：
**C(n,2) = n × (n-1) / 2**

例：
- 3列 → 3個の散布図
- 4列 → 6個の散布図
- 5列 → 10個の散布図
- 10列 → 45個の散布図

### カラーパレット

- **10クラス以下**: seabornの`tab10`パレット
- **10クラス超**: seabornの`hsl`パレット

### 統計計算

- **相関係数**: Pearsonの積率相関係数
- **回帰分析**: 最小二乗法による線形回帰
- **p値**: 相関係数の有意性検定

## 注意事項

- 数値列が多い場合、生成される散布図の数が急激に増加します
- 大量のデータ（数万行以上）の場合、処理に時間がかかる場合があります
- メモリ使用量はCSVファイルのサイズと散布図の数に比例します
- 特殊文字を含む列名は、ファイル名で安全な文字に変換されます

## トラブルシューティング

### メモリ不足エラー

大きなCSVファイルや多数の散布図でメモリ不足が発生する場合：

1. 不要な列を事前に削除
2. データをサンプリングして小さくする
3. 数値列を絞り込む

### 処理時間が長い

散布図の数が多すぎる場合：

1. 重要な特徴量のみを選択
2. 相関行列を確認して関連性の高い組み合わせを特定
3. バッチ処理で分割実行

### 文字化け

日本語を含むCSVファイルで文字化けが発生する場合：

1. CSVファイルがUTF-8エンコーディングで保存されているか確認
2. Excelで保存する場合は「UTF-8 CSV」形式を選択

## 実行例

```bash
# 基本的な実行
python -m tools.scatter_plot_generator extraction_results/20241201_0/features.csv

# 出力例:
# === CSV散布図生成を開始します ===
# CSVファイルを読み込みました: extraction_results/20241201_0/features.csv
# データ形状: (100, 15)
# 数値列を特定しました: 12列
# クラス数: 3
# 生成される散布図数: 66個
# 出力ディレクトリを作成しました: extraction_results/20241201_0/scatter
# === 66個の散布図を生成します ===
# 進行状況 (1/66): brightness_mean vs brightness_median
# 散布図を保存しました: extraction_results/20241201_0/scatter/brightness_mean_vs_brightness_median_scatter.png
# ...
# 相関行列を保存しました: extraction_results/20241201_0/scatter/correlation_matrix.png
# 相関行列CSVを保存しました: extraction_results/20241201_0/scatter/correlation_matrix.csv
# サマリーレポートを保存しました: extraction_results/20241201_0/scatter/scatter_summary.txt
# === CSV散布図生成が完了しました ===
```

## 更新履歴

- v1.0.0: 初回リリース
  - 基本的な散布図生成機能
  - クラス別色分け機能
  - 回帰直線と相関係数表示
  - 相関行列ヒートマップ生成
  - サマリーレポート生成機能 