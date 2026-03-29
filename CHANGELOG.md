# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- GLCM の振る舞いテスト 17 件を追加. 均一画像, チェッカーボード, グラデーション, ランダム画像で特徴量値を検証. ([#164](https://github.com/kurorosu/pochivision/pull/164))
- GLCM docstring の `asm` を `ASM` に修正. ([#165](https://github.com/kurorosu/pochivision/pull/165))
- GLCM docstring に特徴量名形式・特徴量数の計算式を追記. ([#166](https://github.com/kurorosu/pochivision/pull/166))
- HLAC の `except Exception` を削除しエラーをログ出力後に再送出するよう変更. ([#174](https://github.com/kurorosu/pochivision/pull/174))
- HLAC のゼロパディングを削除し, `convolve2d(mode="valid")` で境界を除外した正確なパターンマッチに変更. (NA.)

### Changed
- GLCM に `resize_shape` オプションを追加. ([#163](https://github.com/kurorosu/pochivision/pull/163))

### Fixed
- GLCM の `cv2.normalize(NORM_MINMAX)` を削除し, uint8 変換と整数除算量子化に変更. コントラスト情報が保持される. ([#160](https://github.com/kurorosu/pochivision/pull/160))
- GLCM の `except Exception` を削除しエラーをログ出力後に再送出するよう変更. ([#161](https://github.com/kurorosu/pochivision/pull/161))
- GLCM の NaN/Inf を 0.0 ではなく NaN として保持し, 警告ログを出力するよう変更. ([#162](https://github.com/kurorosu/pochivision/pull/162))

### Removed
- 無し

## [0.1.2] - 2026-03-28

### Added
- `docs/fft_features.md` を追加. FFT 特徴量抽出器の全特徴量・パラメータ・設計制約の解説. ([#149](https://github.com/kurorosu/pochivision/pull/149))

### Changed
- 無し

### Fixed
- FFT 周波数正規化を対角線距離基準から軸方向最大距離基準に修正し, 帯域カバレッジを改善. ([#135](https://github.com/kurorosu/pochivision/pull/135))
- FFT 計算前に Hanning 窓関数を適用し, 画像境界のスペクトルリークを抑制. ([#136](https://github.com/kurorosu/pochivision/pull/136))
- FFT 前の uint8 正規化を削除し, float64 のまま処理するよう変更. コントラスト情報が保持される. ([#137](https://github.com/kurorosu/pochivision/pull/137))
- FFT のエネルギー・エントロピー計算から DC 成分を除外し, AC 成分のみで特徴量を計算するよう修正. ([#138](https://github.com/kurorosu/pochivision/pull/138))
- FFT 帯域エネルギーの最終帯域を上限なしに変更し, 非正方形画像でも合計が ~1.0 になるよう修正. ([#145](https://github.com/kurorosu/pochivision/pull/145))
- FFT 抽出で最小画像サイズ (4x4) のバリデーションを追加. 極小画像で全特徴量がサイレントにゼロになる問題を解消. ([#146](https://github.com/kurorosu/pochivision/pull/146))
- FFT `max_peak_amp` を検出ピーク内の最大振幅に修正 (グローバル最大値ではなく). ([#147](https://github.com/kurorosu/pochivision/pull/147))
- FFT エントロピーを正規化エントロピー (`entropy / log2(N)`, [0, 1] 範囲) に変更し, 帯域間で比較可能に. ([#148](https://github.com/kurorosu/pochivision/pull/148))
- FFT `except Exception` を削除しエラーを伝播するよう変更. `mm_per_pixel` のバリデーションを追加. ([#150](https://github.com/kurorosu/pochivision/pull/150))
- FFT 特徴量抽出器のクラス docstring に前処理フロー・設計制約を追記. エントロピー単位を `normalized` に修正. ([#151](https://github.com/kurorosu/pochivision/pull/151))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
