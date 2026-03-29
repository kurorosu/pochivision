# GLCM 特徴量ガイド

GLCM (Gray-Level Co-occurrence Matrix) 特徴量抽出器の各特徴量とパラメータの解説.

## 概要

GLCM は隣接ピクセル間のグレーレベルの共起関係を行列で表現し, テクスチャの粗さ, 方向性, 規則性を定量化する.

### 処理フロー

1. グレースケール変換
2. リサイズ (`resize_shape` 設定時)
3. uint8 に変換 (コントラスト情報を保持, NORM_MINMAX は使用しない)
4. `levels < 256` の場合は整数除算で量子化
5. `skimage.feature.graycomatrix` で共起行列を計算
6. `skimage.feature.graycoprops` で各プロパティを計算

## 特徴量一覧

各特徴量は距離・角度の組み合わせごとに出力される.
特徴量名の形式: `{property}_{distance}_{angle_deg}` (例: `contrast_1_0`, `energy_2_45`)

### プロパティ

| プロパティ | 単位 | 値の範囲 | 説明 |
|-----------|------|---------|------|
| `contrast` | intensity_squared | 0〜 | コントラスト. 隣接ピクセル間の強度差の二乗の加重和. 大きいほどテクスチャが粗い |
| `dissimilarity` | intensity | 0〜 | 非類似度. 隣接ピクセル間の強度差の加重和. contrast の平方根的な指標 |
| `homogeneity` | ratio | 0.0〜1.0 | 均質性. 局所的にピクセル値が均一なほど高い. contrast と逆の傾向 |
| `energy` | ratio | 0.0〜1.0 | エネルギー. ASM の平方根. テクスチャが規則的なほど高い |
| `correlation` | correlation_coefficient | -1.0〜1.0 | 相関. 隣接ピクセル間の線形依存関係. 1.0 は完全相関, -1.0 は反相関 |
| `ASM` | ratio | 0.0〜1.0 | Angular Second Moment. 共起行列の要素の二乗和. 均一なほど高い |

### 画像例との対応 (distance=1, angle=0)

```
                  contrast  dissimilarity  homogeneity  energy  correlation  ASM
均一画像:          0.00       0.00          1.00         1.00    1.00         1.00
グラデーション:     0.00       0.00          1.00         0.13    1.00         0.02
ランダム:          11364      87.3          0.01         0.01   -0.03         0.00
チェッカーボード:   65025      255.0         0.00         0.71   -1.00         0.50
```

### 各プロパティの解釈

**contrast / dissimilarity** — テクスチャの「粗さ」

```
滑らかな画像:  contrast ≈ 0     (隣接ピクセルが似ている)
粗いテクスチャ: contrast >> 0    (隣接ピクセルが大きく異なる)
```

**homogeneity** — テクスチャの「均一性」

```
均一な画像:    homogeneity ≈ 1.0 (局所的に同じ値)
粗いテクスチャ: homogeneity ≈ 0   (局所的にばらつく)
```

**energy / ASM** — テクスチャの「規則性」

```
均一画像:      energy = 1.0  (全ピクセル同値 → 共起行列に1つのピークのみ)
ランダム:      energy ≈ 0.01 (全値が均等に分散)
グラデーション: energy = 0.13 (規則的だが複数のグレーレベルに分散)
```

**correlation** — 隣接ピクセルの「相関」

```
グラデーション:    correlation ≈ 1.0  (隣接ピクセルが同じ傾向)
ランダム:          correlation ≈ 0    (無相関)
チェッカーボード:   correlation = -1.0 (隣接ピクセルが常に反転)
```

### 方向性

GLCM は角度ごとに計算されるため, テクスチャの方向性を検出できる.

```
垂直グラデーション (angle=0, 水平方向):
  contrast_1_0  = 0.0   ← 水平方向に隣接ピクセルが同値
  contrast_1_90 = 16.4  ← 垂直方向に隣接ピクセルが変化

チェッカーボード:
  contrast_1_0   = 65025 ← 水平方向に最大差
  contrast_1_45  = 0     ← 対角方向に同値
  contrast_1_90  = 65025 ← 垂直方向に最大差
  contrast_1_135 = 0     ← 対角方向に同値
```

## 特徴量数

特徴量数 = `len(properties) x len(distances) x len(angles)`

| 設定例 | 特徴量数 |
|--------|---------|
| デフォルト (6 props x 3 dists x 4 angles) | 72 |
| 簡易 (6 props x 1 dist x 4 angles) | 24 |
| 最小 (3 props x 1 dist x 1 angle) | 3 |

## パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `distances` | `[1, 2, 3]` | ピクセル間距離のリスト. 大きいほど遠い隣接関係を見る |
| `angles` | `[0, 45, 90, 135]` | 角度 (度数) のリスト. 0=水平, 90=垂直 |
| `levels` | `256` | グレーレベル数. 小さいほど計算が速いが量子化が粗くなる |
| `symmetric` | `true` | 対称性. (i,j) と (j,i) を同一視するか |
| `normed` | `true` | 共起行列を正規化するか |
| `properties` | `["contrast", ...]` | 計算するプロパティのリスト |
| `resize_shape` | `null` | リサイズ形状. `[512, 512]` 等. null=リサイズなし |
| `preserve_aspect_ratio` | `true` | アスペクト比を保持するか |
| `aspect_ratio_mode` | `"width"` | アスペクト比保持時の基準軸 (`"width"` or `"height"`) |

### `distances` の調整

```
distance=1:  直接隣接するピクセルの関係 (微細なテクスチャ)
distance=3:  3ピクセル離れた関係 (粗いテクスチャ)
distance=10: 10ピクセル離れた関係 (大域的パターン)
```

### `levels` の調整

```
levels=256: フル解像度 (高精度, 計算コスト高)
levels=64:  1/4 解像度 (高速, 微細な差異を無視)
levels=16:  粗い量子化 (非常に高速, 大まかなテクスチャのみ)
```

## 設計上の制約

1. **コントラスト正規化なし**: `cv2.normalize(NORM_MINMAX)` は使用しない. 異なるコントラストの画像で異なる特徴量が得られる.
2. **NaN 処理**: 均一画像の `correlation` が NaN になる場合, NaN のまま保持し警告ログを出力する (0.0 に置換しない).
3. **リサイズ**: `preserve_aspect_ratio` でアスペクト比保持を制御可能 (デフォルト: 保持).

## 設定例

```json
{
    "glcm": {
        "distances": [1],
        "angles": [0, 45, 90, 135],
        "levels": 256,
        "symmetric": true,
        "normed": true,
        "properties": [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM"
        ],
        "resize_shape": [512, 512],
        "preserve_aspect_ratio": true,
        "aspect_ratio_mode": "width"
    }
}
```
