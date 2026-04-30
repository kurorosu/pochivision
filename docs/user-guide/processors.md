# 利用可能なプロセッサ

カメラプロファイルの `processors` 配列に列挙できる組み込みプロセッサの一覧. 各プロセッサは `@register_processor` でレジストリに登録されており, 設定ファイルから名前で参照する.

## 1. プロセッサ一覧

| # | 名前 | 説明 | 主要パラメータ |
|---|------|------|----------------|
| 1 | `grayscale` | グレースケール変換 | なし |
| 2 | `gaussian_blur` | ガウシアンぼかし | `kernel_size`, `sigma` |
| 3 | `average_blur` | 平均値ブラー | `kernel_size` |
| 4 | `median_blur` | メディアンブラー | `kernel_size` (奇数) |
| 5 | `bilateral_filter` | エッジ保持ぼかし | `d`, `sigmaColor`, `sigmaSpace` |
| 6 | `motion_blur` | モーションブラー | `kernel_size` (奇数), `angle` |
| 7 | `std_bin` | 標準 2 値化 | `threshold` |
| 8 | `otsu_bin` | 大津の 2 値化 | なし |
| 9 | `gauss_adapt_bin` | ガウス適応的 2 値化 | `block_size`, `c` |
| 10 | `mean_adapt_bin` | 平均適応的 2 値化 | `block_size`, `c` |
| 11 | `resize` | リサイズ | `width`, `height`, `preserve_aspect_ratio`, `aspect_ratio_mode` |
| 12 | `canny_edge` | Canny エッジ検出 | `threshold1`, `threshold2`, `aperture_size` |
| 13 | `contour` | 輪郭検出 | `retrieval_mode`, `min_area`, `select_mode`, `contour_rank` |
| 14 | `clahe` | CLAHE コントラスト強調 | `clip_limit`, `tile_grid_size`, `color_mode` |
| 15 | `equalize` | ヒストグラム平坦化 | `color_mode` |
| 16 | `mask_composition` | マスクとソース画像の合成 | `target_image`, `use_white_pixels`, `enable_cropping` |

## 2. 設定例

```json
{
  "processors": ["resize", "gaussian_blur", "std_bin"],
  "mode": "pipeline",
  "resize": { "width": 1600, "height": 1200, "preserve_aspect_ratio": true, "aspect_ratio_mode": "width" },
  "gaussian_blur": { "kernel_size": [19, 19], "sigma": 0 },
  "std_bin": { "threshold": 20 }
}
```

## 3. プロセッサの追加

カスタムプロセッサは `BaseProcessor` を継承し, `@register_processor` デコレータでレジストリに登録する. 詳細は `pochivision/processors/base.py` および既存実装を参照.

```python
from pochivision.processors.base import BaseProcessor
from pochivision.processors.registry import register_processor

@register_processor("my_filter")
class MyFilter(BaseProcessor):
    def process(self, image):
        ...
```

登録済みのプロセッサ名は `config.json` の `processors` 配列にそのまま指定できる.
