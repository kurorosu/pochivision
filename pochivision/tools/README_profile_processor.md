# プロファイルベース画像処理ツール

既存のconfig.jsonのカメラプロファイル設定を使用して、フォルダ内の画像に対して指定されたプロファイルの処理を適用するツールです。

## 概要

このツールは、既存の画像に対してconfig.jsonで定義されたカメラプロファイル（"0"、"all_para"、"high_fps"、"resize_sample"など）の処理設定を適用します。リアルタイム撮影ではなく、既存の画像ファイルに対してバッチ処理を行います。

## 機能

- 複数のプロファイルから選択可能
- パイプライン処理とパラレル処理の両方に対応
- 元画像と処理済み画像の両方を保存
- 処理に使用した設定情報の保存
- 詳細な処理ログの出力

## 使用方法

### 基本的な使用方法

```bash
# プロファイル一覧を表示
python -m tools.profile_processor --list-profiles

# プロファイル "0" を使用して画像を処理
python -m tools.profile_processor -p 0 -i input_images -o output_results

# プロファイル "all_para" を使用
python -m tools.profile_processor -p all_para -i input_images -o output_results
```

### コマンドライン引数

| 引数 | 短縮形 | 説明 | デフォルト値 |
|------|--------|------|-------------|
| `--config` | `-c` | 設定ファイルのパス | `config.json` |
| `--profile` | `-p` | 使用するプロファイル名 | 必須 |
| `--input` | `-i` | 入力ディレクトリのパス | 必須 |
| `--output` | `-o` | 出力ディレクトリのパス | `profile_results` |
| `--no-save-original` | - | 元画像を保存しない | False |
| `--list-profiles` | - | 利用可能なプロファイルを一覧表示 | False |

### 使用例

#### 1. プロファイル一覧の確認

```bash
python -m tools.profile_processor --list-profiles
```

出力例：
```
利用可能なプロファイル:
  0: Tokyo_Lab
    モード: pipeline
    プロセッサ: resize, gaussian_blur, std_bin, contour, mask_composition

  all_para: All_Parallel
    モード: parallel
    プロセッサ: motion_blur, bilateral_filter, median_blur, gaussian_blur, ...

  high_fps: High_FPS
    モード: pipeline
    プロセッサ: grayscale, resize

  resize_sample: Resize_Sample
    モード: pipeline
    プロセッサ: resize, canny_edge
```

#### 2. 基本的な画像処理

```bash
# プロファイル "0" を使用してinput_imagesフォルダの画像を処理
python -m tools.profile_processor -p 0 -i input_images -o results
```

#### 3. 元画像を保存せずに処理

```bash
# 処理済み画像のみを保存
python -m tools.profile_processor -p all_para -i input_images -o results --no-save-original
```

#### 4. カスタム設定ファイルを使用

```bash
# 別の設定ファイルを使用
python -m tools.profile_processor -c custom_config.json -p 0 -i input_images -o results
```

#### 5. 実際の使用例（集約済み画像の処理）

```bash
# 画像集約ツールで集約された画像を処理
python -m tools.profile_processor -p 0 -i image_aggregated/20250524_2/original -o profile_results
```

## 出力構造

処理完了後、以下のような構造で結果が保存されます：

```
output_directory/
└── profile_{プロファイル名}_{タイムスタンプ}/
    ├── original/           # 元画像（--no-save-originalが指定されていない場合）
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── processed/          # 処理済み画像
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── profile_info.json   # 使用した設定情報
```

### profile_info.json の内容

```json
{
  "profile_name": "0",
  "profile_config": {
    "width": 3200,
    "height": 2400,
    "processors": ["resize", "gaussian_blur", "std_bin", "contour", "mask_composition"],
    "mode": "pipeline",
    ...
  },
  "timestamp": "2024-01-15T10:30:45.123456",
  "config_file": "config.json"
}
```

## 対応画像形式

以下の画像形式に対応しています：

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

大文字・小文字の区別なく処理されます。

## プロファイルの処理モード

### Pipeline モード

プロセッサが順次適用されます。前のプロセッサの出力が次のプロセッサの入力となります。

```
元画像 → プロセッサ1 → プロセッサ2 → ... → 最終結果
```

### Parallel モード

各プロセッサが元画像に対して独立して適用されます。現在の実装では最後のプロセッサの結果が保存されます。

```
元画像 → プロセッサ1 → 結果1
      → プロセッサ2 → 結果2
      → ...
      → プロセッサN → 結果N（保存される）
```

## エラー処理

- 存在しないプロファイルが指定された場合、利用可能なプロファイル一覧が表示されます
- 画像の読み込みや処理に失敗した場合、警告メッセージが表示され、次の画像の処理が続行されます
- 処理完了時に成功・失敗の統計が表示されます

## 注意事項

1. **プロセッサの依存関係**: 使用するプロファイルで指定されたプロセッサが正しく初期化できることを確認してください
2. **メモリ使用量**: 大きな画像や多数の画像を処理する場合、メモリ使用量にご注意ください
3. **処理時間**: プロファイルによっては処理に時間がかかる場合があります
4. **出力ディレクトリ**: 既存のファイルは上書きされる可能性があります

## トラブルシューティング

### よくある問題

1. **プロセッサの初期化エラー**
   ```
   警告: プロセッサの初期化に失敗しました (processor_name): エラーメッセージ
   ```
   → config.jsonの該当プロセッサの設定を確認してください

2. **画像の読み込みエラー**
   ```
   警告: 画像の読み込みに失敗しました: image_path
   ```
   → 画像ファイルが破損していないか、対応形式かを確認してください

3. **設定ファイルが見つからない**
   ```
   エラー: 設定ファイルが見つかりません: config.json
   ```
   → 設定ファイルのパスが正しいか確認してください

4. **Target image 'original' is not set エラー**
   ```
   エラー: Target image 'original' is not set
   ```
   → mask_compositionプロセッサが含まれるプロファイルをパラレルモードで実行しようとした場合に発生します。パイプラインモードを使用してください。

### デバッグ方法

1. まず `--list-profiles` でプロファイルが正しく読み込まれるか確認
2. 小さなテスト画像で動作確認
3. ログメッセージを確認して問題箇所を特定

## 実行例

```bash
# 基本的な実行
python -m tools.profile_processor -p 0 -i image_aggregated/20250524_2/original

# 出力例:
# プロセッサを初期化しました: resize
# プロセッサを初期化しました: gaussian_blur
# プロセッサを初期化しました: std_bin
# プロセッサを初期化しました: contour
# プロセッサを初期化しました: mask_composition
# 処理開始: 2個の画像を処理します
# 使用プロファイル: 0
# 出力ディレクトリ: profile_results\profile_0_20250601_093810
# プロファイル情報を保存しました: profile_results\profile_0_20250601_093810\profile_info.json
# 処理中: image1.bmp
#   → 保存完了: profile_results\profile_0_20250601_093810\processed\image1.bmp
# 処理中: image2.bmp
#   → 保存完了: profile_results\profile_0_20250601_093810\processed\image2.bmp
#
# 処理完了:
#   成功: 2個
#   失敗: 0個
#   出力先: profile_results\profile_0_20250601_093810
```

## 関連ファイル

- `config.json`: プロファイル設定ファイル
- `processors/`: 各種プロセッサの実装
- `processors/registry.py`: プロセッサの登録・取得機能
