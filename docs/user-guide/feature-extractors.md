# 利用可能な Feature Extractor

`pochi extract` で利用できる組み込み Feature Extractor の一覧. 処理済み画像を解析し, 下流の AI タスク向けに数値特徴量を出力する.

## 1. Feature Extractor 一覧

| # | 名前 | 説明 | 詳細 |
|---|------|------|------|
| 1 | `rgb` | RGB チャンネル統計量 (平均, 標準偏差等) | |
| 2 | `hsv` | HSV チャンネル統計量 | |
| 3 | `brightness` | 輝度統計量 | |
| 4 | `glcm` | GLCM テクスチャ特徴量 (コントラスト, エネルギー等) | [GLCM 特徴量](features/glcm.md) |
| 5 | `hlac` | HLAC テクスチャ特徴量 | [HLAC 特徴量](features/hlac.md) |
| 6 | `lbp` | LBP テクスチャ特徴量 | [LBP 特徴量](features/lbp.md) |
| 7 | `fft` | FFT 周波数特徴量 | [FFT 特徴量](features/fft.md) |
| 8 | `swt` | SWT 周波数特徴量 | [SWT 特徴量](features/swt.md) |
| 9 | `circle_counter` | 円検出・カウント | |

## 2. 設定例

`config/extractor_config.json` で抽出器名と固有パラメータを指定する.

```json
{
  "input_dir": "capture/20260101_120000",
  "output_csv": "features.csv",
  "extractors": ["rgb", "glcm", "fft"],
  "glcm": { "distances": [1], "angles": [0, 0.785, 1.571], "levels": 16 },
  "fft": { "n_radial_bins": 8 }
}
```

## 3. Feature Extractor の追加

カスタム抽出器は `BaseFeatureExtractor` を継承し, `@register_feature_extractor` デコレータで登録する.

```python
from pochivision.feature_extractors.base import BaseFeatureExtractor
from pochivision.feature_extractors.registry import register_feature_extractor

@register_feature_extractor("my_feature")
class MyFeature(BaseFeatureExtractor):
    def extract(self, image):
        ...
```

## 4. 特徴量リファレンス

各 Extractor が出力する特徴量, 設計制約, パラメータ詳細は以下を参照.

- [FFT 特徴量](features/fft.md)
- [GLCM 特徴量](features/glcm.md)
- [HLAC 特徴量](features/hlac.md)
- [LBP 特徴量](features/lbp.md)
- [SWT 特徴量](features/swt.md)
