# HLAC (Higher-order Local Auto-Correlation) 特徴量抽出

## 概要

HLAC（高次局所自己相関）は、画像のテクスチャや形状を特徴づける手法です。局所的なピクセルパターンの自己相関を計算することで、画像の特徴を数値化します。

## 特徴

- **スケール不変性**: 複数のスケールで特徴を抽出し、画像サイズに依存しない特徴量を生成
- **回転不変性**: オプションで回転に対して不変な特徴量を抽出可能
- **正規化**: 特徴量を正規化することで、明度変化に対する頑健性を向上

## HLAC特徴量の種類

### 標準HLAC（45次元）

0次から2次までの局所自己相関パターンを使用：

#### 0次自己相関（1パターン）
- 中心画素のみ

#### 1次自己相関（8パターン）
- 中心画素 + 8方向の隣接画素

#### 2次自己相関（36パターン）
- 中心画素 + 2つの隣接画素の組み合わせ

### 回転不変HLAC（11次元）

回転に対して不変な特徴量：

```
0 : 中心のみ                               (C)
1 : 中心 + 右                            (C + R)
2 : 中心 + 右 + 右                     (C + R + R)
3 : 中心 + 右 + 下                     (C + R + D)
4 : 中心 + 右 + 上                     (C + R + U)
5 : 中心 + 下 + 下                     (C + D + D)
6 : 中心 + 右 + 右 + 下               (C + R + R + D)
7 : 中心 + 右 + 右 + 上               (C + R + R + U)
8 : 中心 + 右 + 下 + 下               (C + R + D + D)
9 : 中心 + 右 + 上 + 上               (C + R + U + U)
10: 中心 + 右 + 右 + 下 + 上          (C + R + R + D + U)
```

## パラメータ

- `order`: 自己相関の次数（1または2）
- `rotate_invariant`: 回転不変性の有効/無効
- `normalize`: 特徴量の正規化の有効/無効
- `scales`: マルチスケール処理のスケール係数リスト

## 使用例

```python
from vision_capture_core.extractors.hlac import HLACExtractor

# 標準HLAC（45次元）
extractor = HLACExtractor(
    order=2,
    rotate_invariant=False,
    normalize=True,
    scales=[1.0, 0.75, 0.5]
)

# 回転不変HLAC（11次元）
extractor_ri = HLACExtractor(
    order=2,
    rotate_invariant=True,
    normalize=True,
    scales=[1.0, 0.75, 0.5]
)

# 特徴量抽出
features = extractor.extract(image)
```

## 応用分野

- テクスチャ分析
- 物体認識
- 画像分類
- パターンマッチング

## 参考文献

- Otsu, N., & Kurita, T. (1988). "A new scheme for practical flexible and intelligent vision systems." IAPR Workshop on Computer Vision.
- 高次局所自己相関特徴による画像認識に関する研究
