# 設定ファイル (`config.json`)

カメラプロファイル, 処理パイプライン, 録画, プレビュー設定はすべて JSON 設定ファイル (`config/config.json`) で定義する. 本ページでは構造とフィールドを解説する.

## 1. 設定ファイル構造

```json
{
  "cameras": {
    "0": {
      "width": 3200,
      "height": 2400,
      "fps": 30,
      "backend": "DSHOW",
      "label": "Tokyo_Lab",
      "processors": ["resize", "gaussian_blur", "std_bin", "contour", "mask_composition"],
      "mode": "pipeline",
      "id_interval": 4,
      "gaussian_blur": { "kernel_size": [19, 19], "sigma": 0 },
      "std_bin": { "threshold": 20 },
      "resize": { "width": 1600, "height": 1200, "preserve_aspect_ratio": true, "aspect_ratio_mode": "width" },
      "contour": { "retrieval_mode": "list", "approximation_method": "simple", "min_area": 100 },
      "mask_composition": { "target_image": "original", "use_white_pixels": true, "enable_cropping": true }
    }
  },
  "recording": {
    "select_format": "mjpg"
  },
  "preview": {
    "width": 1280,
    "height": 720
  },
  "selected_camera_index": 0,
  "id_interval": 1
}
```

## 2. トップレベル設定

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `cameras` | Yes | - | カメラプロファイル定義 |
| `selected_camera_index` | Yes | - | CLI 未指定時に使用するカメラインデックス |
| `id_interval` | No | `1` | グローバルキャプチャ ID 間隔 |
| `recording.select_format` | No | `"mjpg"` | 録画コーデック (`mp4v`, `xvid`, `mjpg`, `ffv1` 等) |
| `preview.width` | No | `1280` | プレビューウィンドウ幅 |
| `preview.height` | No | `720` | プレビューウィンドウ高さ |

## 3. カメラプロファイル設定

| キー | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `processors` | Yes | - | プロセッサ名の配列 (空不可) |
| `width` | No | `640` | カメラ解像度の幅 |
| `height` | No | `480` | カメラ解像度の高さ |
| `fps` | No | `30` | フレームレート |
| `backend` | No | none | カメラバックエンド (`DSHOW`, `MSMF`, `V4L2` 等) |
| `mode` | No | `"parallel"` | `"parallel"` または `"pipeline"` |
| `label` | No | none | 出力ファイル名用カスタムラベル |
| `id_interval` | No | `1` | プロファイル別キャプチャ ID 間隔 |

各プロセッサ固有のパラメータ (例: `gaussian_blur.kernel_size`) はプロファイル直下にプロセッサ名のキーとしてネストする.

## 4. 実行モード

| モード | 説明 |
|--------|------|
| `pipeline` | プロセッサを直列に適用. 前段の出力が次段の入力になる |
| `parallel` | 各プロセッサを元画像に並列適用. 結果は個別に保存される |

## 5. 注意点

- プロファイル名が数字の場合 (例: `"0"`), CLI の `--camera` 引数と一致する必要がある.
- CLI で `--camera` のみ指定した場合, プロファイル `"0"` を使用する.
- カメラプロファイルごとに異なるプロセッサと設定を指定できる.
- 登録されていないプロセッサを指定するとエラーになる.

## 6. 関連設定ファイル

- 推論 API 設定: [推論モード](inference.md) の `infer_config.json`
- 検出 API 設定: [検出モード](detection.md) の `detect_config.json`
- 特徴量抽出設定: [特徴量抽出](extract.md) の `extractor_config.json`
