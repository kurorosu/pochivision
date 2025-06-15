# CSV Analytics CLI

CSVファイルの基本的なデータ分析と機械学習モデリングを行うコマンドラインツールです。

## 機能

### データ分析機能
- CSVファイルの読み込みと基本統計情報の表示
- インタラクティブなヒストグラム表示
- Long形式（縦持ち）からWide形式（横持ち）への自動変換
- クラス別データ分析とヒストグラム表示
- データ転置機能

### 機械学習モデリング機能
- **XGBoost分類モデリング（2つの訓練方式対応）**
  - ホールドアウト法（80%訓練、20%テスト）
  - クロスバリデーション法（全データで5-fold CV）
- **Optunaハイパーパラメータチューニング**
- **PCA散布図の自動生成**
- **特徴量重要度分析（両モデル対応）**
- **外部設定ファイルによるパラメータ管理**
- **結果の自動保存とモデル永続化**

## 必要な依存関係

- Python 3.8+
- pandas
- numpy
- plotext
- questionary
- rich
- click
- xgboost
- scikit-learn
- matplotlib
- seaborn
- optuna

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python -m analytics.cli.main
```

または

```bash
python tools/analytics/cli/main.py
```

## 機械学習モデリングの詳細

### 2つの訓練方式

#### 1. ホールドアウト法
- **データ分割**: 80%を訓練用、20%をテスト用に分割
- **モデル訓練**: 訓練データ（80%）でデフォルトパラメータを使用して訓練
- **評価**: テストデータ（20%）で評価
- **目的**: 未知データに対する真の汎化性能を測定

#### 2. クロスバリデーション法（Optuna最適化使用時）
- **ハイパーパラメータ最適化**: 全データで5-fold CV
- **最終モデル訓練**: 全データで最適化されたパラメータを使用して訓練
- **評価**: 全データで5-fold CVを実行し平均精度を計算
- **目的**: 全データを活用した最高性能モデルの構築

### 訓練からテスト評価の流れ

```
1. データ読み込み・前処理
   ├── CSVファイル読み込み
   ├── ラベルエンコーディング（文字列→数値）
   └── 特徴量とターゲットの分離

2. モデル訓練方式の選択
   ├── ホールドアウト法
   │   ├── データ分割（80%/20%）
   │   ├── モデル訓練（訓練データ、デフォルトパラメータ）
   │   └── テスト評価（テストデータ）
   │
   └── クロスバリデーション法（Optuna最適化使用時）
       ├── ハイパーパラメータ最適化（全データで5-fold CV）
       ├── 最終モデル訓練（全データ、最適化パラメータ）
       └── 評価（全データで5-fold CV平均）

3. 結果出力・保存
   ├── モデル保存（models{index}/）
   │   ├── xgboost_holdout_model.pkl（常に保存）
   │   ├── xgboost_cv_model.pkl（Optuna使用時のみ）
   │   └── label_encoder.pkl
   ├── 特徴量重要度分析
   │   ├── CSV出力（ホールドアウトTop5、CVはOptuna使用時のみ）
   │   └── コンソール表示（ホールドアウトTop3、CVはOptuna使用時のみ）
   ├── PCA散布図生成
   └── 評価結果表示
       ├── ホールドアウトテスト精度（常に表示）
       └── CV法最適平均精度（Optuna使用時のみ）
```

### 出力ファイル構造

```
models{index}/
├── xgboost_holdout_model.pkl      # ホールドアウトモデル
├── xgboost_cv_model.pkl           # CVモデル（Optuna使用時のみ）
├── label_encoder.pkl              # ラベルエンコーダー
├── feature_importance.csv         # 特徴量重要度（両モデル）
├── pca_scatter_plot.png          # PCA散布図
└── classification_results.csv     # 分類結果
```

## 機能詳細

### データ読み込み
- `extraction_results`フォルダからの自動選択
- 任意のパスからの読み込み

### ヒストグラム表示
- 単純なヒストグラム
- クラス別色分けヒストグラム

### データ変換
- Long形式の自動検出
- ピボット変換による横持ち化
- データ転置

### 分類モデリング
- XGBoostを使用した自動分類
- デフォルトパラメータまたはOptunaによる最適化
- PCA散布図の自動生成
- 特徴量重要度の分析
- 結果の自動保存（CSV、PNG）

### パラメータ管理
- `model_param.json`による設定管理
- デフォルトパラメータの外部設定
- Optuna探索範囲の柔軟な設定
- パラメータ説明の日本語対応
- **すべての最適化で設定ファイルの探索範囲を使用**（試行回数のみ選択可能）

## 設定ファイル

### model_param.json
XGBoostのパラメータ設定を管理します：

```json
{
  "xgboost": {
    "default_params": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1,
      "subsample": 1.0,
      "colsample_bytree": 1.0,
      "reg_alpha": 0,
      "reg_lambda": 1,
      "min_child_weight": 1,
      "gamma": 0,
      "scale_pos_weight": 1
    },
    "optuna_search_space": {
      "n_estimators": {
        "type": "int",
        "low": 50,
        "high": 300,
        "description": "ブースティングラウンド数"
      },
      "max_depth": {
        "type": "int",
        "low": 3,
        "high": 10,
        "description": "木の最大深度"
      },
      "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.3,
        "description": "学習率"
      }
    },
    "optuna_config": {
      "default_trials": 100,
      "cv_folds": 5,
      "sampler": "TPESampler"
    }
  }
}
```

## 使用例

### 基本的なモデリング
1. ツールを起動
2. CSVファイルを選択
3. 「分類モデリング」を選択
4. デフォルトパラメータで実行

### Optuna最適化
1. ツールを起動
2. CSVファイルを選択
3. 「分類モデリング」を選択
4. 「Optunaを使用する」を選択
5. 試行回数を指定（デフォルト100回）

## 開発者向け情報

### プロジェクト構造

```
analytics/
├── cli/                          # コマンドライン関連
│   └── main.py                   # メインエントリーポイント
├── core/                         # ビジネスロジック
│   ├── analyzer.py               # データ分析
│   ├── classification_modeler.py # 機械学習モデリング
│   └── data_loader.py           # データ読み込み
├── ui/                          # ユーザーインターフェース
│   └── interactive.py           # インタラクティブUI
├── utils/                       # ユーティリティ
│   └── param_manager.py         # パラメータ管理
├── tests/                       # テストコード
└── model_param.json            # パラメータ設定ファイル
```

### 主要クラス

- `ClassificationModeler`: 機械学習モデリングの中核クラス
- `ParameterManager`: パラメータ設定の管理
- `DataAnalyzer`: データ分析機能
- `DataLoader`: データ読み込み機能

### コード品質管理

このプロジェクトでは以下のツールを使用：
- `black`: コードフォーマット
- `isort`: import文の整理
- `flake8`: 静的解析
- `mypy`: 型チェック
- `pydocstyle`: Docstringチェック

### テスト実行

```bash
python -m pytest analytics/tests/
```

## ライセンス

このプロジェクトのライセンスに従います。 