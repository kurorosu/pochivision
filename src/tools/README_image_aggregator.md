# 画像集約ユーティリティ（image_aggregator.py）

このツールは、カメラディレクトリ内のすべての日付フォルダから、処理タイプごとに画像を集約し、新しい出力フォルダ（`image_aggregated`）にコピーまたは移動します。

---

## 使い方

### 基本コマンド

```bash
python -m tools.image_aggregator -i <入力カメラディレクトリ> [-m <モード>]
```

- `-i`, `--input` : 入力カメラディレクトリ（例: `./capture/camera1`）【必須】
- `-m`, `--mode`  : 操作モード（`copy` または `move`、デフォルトは `copy`）

### 実行例

```bash
# camera1の画像をコピーで集約
python -m tools.image_aggregator -i ./capture/camera1 -m copy

# camera0の画像を移動で集約
python -m tools.image_aggregator -i ./capture/camera0 -m move
```

---

## 出力

- 画像は常に `image_aggregated/YYYYMMDD_INDEX` フォルダに集約されます。
- 各処理タイプごとにサブフォルダが作成されます。
- 例:
  ```
  image_aggregated/
    └─ 20240524_0/
         ├─ original/
         ├─ pipeline/
         └─ ...
  ```

---

## 典型的な出力メッセージ

- `Successfully copyd 10 images to image_aggregated`
- `No images were processed.`
- `Error during image aggregation: ...`

---

## オプション・詳細

- `copy` モード: 画像をコピーして集約
- `move` モード: 画像を移動して集約（元画像は削除されます）

---

## エラー例と対処

- **入力ディレクトリが存在しない場合**
  → ディレクトリパスを確認してください

- **画像が見つからない場合**
  → 日付フォルダや処理タイプサブフォルダの構成を確認してください

- **PermissionError等**
  → ファイルのロックや権限を確認してください

---

## 動作環境

- Python 3.8 以上
- 標準ライブラリのみ（追加パッケージ不要）

---

## ライセンス

本ツールは「vision-capture-core」プロジェクトのライセンスに従います。
