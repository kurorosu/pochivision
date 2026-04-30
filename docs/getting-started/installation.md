# インストール

## 1. 要件

- Python 3.12+

## 2. uv を使用する場合 (推奨)

uv のインストール (未インストールの場合):

```bash
pip install uv
```

リポジトリをクローンして依存関係をインストールします.

```bash
git clone https://github.com/kurorosu/pochivision.git
cd pochivision
uv sync
```

仮想環境の有効化:

```bash
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / Mac
```

## 3. 開発者向けセットアップ

開発・テスト・Lint・ドキュメント (mkdocs) も含めてインストールする場合:

```bash
uv sync --group dev
```

`dev` グループには以下のドキュメント関連ツールが含まれます.

- `mkdocs`: ドキュメントサイトジェネレータ.
- `mkdocs-material`: Material テーマ.
- `mkdocstrings[python]`: Python docstring からの API リファレンス自動生成.

## 4. ドキュメントのローカル閲覧

依存関係インストール後, 以下のコマンドでローカルサーバーが起動します.

```bash
uv run mkdocs serve
```

ブラウザで `http://127.0.0.1:8000/` を開くと本ドキュメントが閲覧できます.

静的ファイルとしてビルドする場合は以下を実行します.

```bash
uv run mkdocs build
```

`site/` ディレクトリに HTML が生成されます (リポジトリ管理外).
