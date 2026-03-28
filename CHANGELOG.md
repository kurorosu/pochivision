# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- FFT 周波数正規化を対角線距離基準から軸方向最大距離基準に修正し, 帯域カバレッジを改善. ([#135](https://github.com/kurorosu/pochivision/pull/135))
- FFT 計算前に Hanning 窓関数を適用し, 画像境界のスペクトルリークを抑制. ([#136](https://github.com/kurorosu/pochivision/pull/136))
- FFT 前の uint8 正規化を削除し, float64 のまま処理するよう変更. コントラスト情報が保持される. ([#137](https://github.com/kurorosu/pochivision/pull/137))
- FFT のエネルギー・エントロピー計算から DC 成分を除外し, AC 成分のみで特徴量を計算するよう修正. (NA.)

### Removed
- 無し

## [0.1.1] - 2026-03-28

### Added
- 無し

### Changed
- `pochivision/tools/` をプロジェクトルートの `tools/` に移動し, パッケージから分離. 未使用の dev 依存 `flake8`, `pylint` を削除. ([#117](https://github.com/kurorosu/pochivision/pull/117))
- `test_blur_processors.py` のモックテスト 2 件を削除し, 全テストを古典派テストに統一. ([#118](https://github.com/kurorosu/pochivision/pull/118))
- FFT を1回計算して全ヘルパーで共有するようリファクタリング. HLAC 2次パターンの自己ペアを除外し特徴次元を 45→37 に修正. ([#126](https://github.com/kurorosu/pochivision/pull/126))
- Issue テンプレートの Branch/Label セクションを統一. ([#128](https://github.com/kurorosu/pochivision/pull/128))

### Fixed
- HSV 特徴量抽出の Hue チャンネルに循環統計 (`scipy.stats.circmean` / `circstd`) を適用し, 0/180 境界付近の統計値を修正. 単位ラベルを `hue_0_179` に変更. ([#119](https://github.com/kurorosu/pochivision/pull/119))
- CircleCounter の真円度フィルタを合成円ではなく実画像のエッジ輪郭に基づく評価に修正. ([#120](https://github.com/kurorosu/pochivision/pull/120))
- `(H,W,1)` 形状画像で CLAHE/Equalize がクラッシュする問題を修正. `to_grayscale` で BGRA 画像に `COLOR_BGRA2GRAY` を使用するよう修正. ([#121](https://github.com/kurorosu/pochivision/pull/121))
- モーションブラーカーネル構築を `cv2.line` 方式に変更し, 斜め角度でのギャップを解消. ([#122](https://github.com/kurorosu/pochivision/pull/122))
- FFT 方向エネルギーの 0/180 度境界処理に対応. スペクトルエントロピーのゼロ要素バイアスを修正. ([#123](https://github.com/kurorosu/pochivision/pull/123))
- `PipelineExecutor` に `mode` 値の検証を追加し, 個別プロセッサの例外でパイプライン全体が中断しないよう修正. ([#124](https://github.com/kurorosu/pochivision/pull/124))
- resize の補間方法を拡大時に `INTER_LINEAR` に切替, edge_detection の float 判定を `np.floating` に修正, CLAHE バリデータで tuple を許容. ([#125](https://github.com/kurorosu/pochivision/pull/125))
- SWT のサイズ調整を `2^max_level` の倍数に対応. 正規化判定を値ベースから `dtype` ベースに変更. ([#132](https://github.com/kurorosu/pochivision/pull/132))
- LBP ヒストグラムのビン数を理論値 (uniform: P+2, default: 2^P) に固定し, 画像依存の特徴ベクトル長不定を解消. ([#133](https://github.com/kurorosu/pochivision/pull/133))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
