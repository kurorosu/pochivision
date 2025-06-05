# CSV Analytics CLI

CSVファイルの基本的なデータ分析を行うコマンドラインツールです。

## 機能

- CSVファイルの読み込みと基本統計情報の表示
- インタラクティブなヒストグラム表示
- Long形式（縦持ち）からWide形式（横持ち）への自動変換
- クラス別データ分析とヒストグラム表示
- データ転置機能

## 必要な依存関係

- Python 3.8+
- pandas
- numpy
- plotext
- questionary
- rich
- click

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

## 開発者向け情報

### プロジェクト構造

```
analytics/
├── cli/           # コマンドライン関連
├── core/          # ビジネスロジック
├── ui/            # ユーザーインターフェース
└── utils/         # ユーティリティ
```

### テスト実行

```bash
python -m pytest analytics/tests/
```

## ライセンス

このプロジェクトのライセンスに従います。 