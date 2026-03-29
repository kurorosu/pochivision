# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- SWT エントロピー計算を `np.histogram` から `np.bincount` ベースに変更し, 狭い値域でのクラッシュを解消. ([#193](https://github.com/kurorosu/pochivision/pull/193))
- SWT マルチスケールのレベルラベリングを修正し, L1=高周波, LN=低周波に変更. ([#194](https://github.com/kurorosu/pochivision/pull/194))
- SWT `multiscale=False` 時のレベル選択と docstring を明確化 (level 1, 高周波詳細). ([#195](https://github.com/kurorosu/pochivision/pull/195))
- SWT の dtype 正規化を統一し, uint8 と float32 (0-255) で同じ特徴量が得られるよう修正. (NA.)

### Removed
- 無し

## [0.1.3] - 2026-03-29

### Added
- `docs/glcm_features.md` を追加. GLCM 特徴量抽出器の全プロパティ・パラメータ・設計制約の解説. ([#176](https://github.com/kurorosu/pochivision/pull/176))
- `docs/hlac_features.md` を追加. FFT/GLCM docs にリサイズパラメータを追記. ([#184](https://github.com/kurorosu/pochivision/pull/184))
- GLCM の振る舞いテスト 17 件を追加. ([#164](https://github.com/kurorosu/pochivision/pull/164))
- HLAC の振る舞いテスト 11 件を追加. ([#181](https://github.com/kurorosu/pochivision/pull/181))
- GLCM に `resize_shape` オプションを追加. ([#163](https://github.com/kurorosu/pochivision/pull/163))
- FFT と SWT に `resize_shape` を追加. 全抽出器に `preserve_aspect_ratio` / `aspect_ratio_mode` を追加. ([#180](https://github.com/kurorosu/pochivision/pull/180))
- HLAC に適応的二値化 (`adaptive`) を追加しデフォルトに設定. ([#175](https://github.com/kurorosu/pochivision/pull/175))

### Changed
- GLCM docstring の `asm` を `ASM` に修正. ([#165](https://github.com/kurorosu/pochivision/pull/165))
- GLCM docstring に特徴量名形式と特徴量数の計算式を追記. ([#166](https://github.com/kurorosu/pochivision/pull/166))
- HLAC の二値化を `binary / 255` から `binary > 0` に変更, `convolve2d` を `correlate2d` に変更. ([#182](https://github.com/kurorosu/pochivision/pull/182))
- HLAC のスケールリサイザーを初回呼び出し時にキャッシュ. ([#179](https://github.com/kurorosu/pochivision/pull/179))
- HLAC docstring に回転不変性の制約を追記. ([#183](https://github.com/kurorosu/pochivision/pull/183))

### Fixed
- GLCM の `cv2.normalize(NORM_MINMAX)` を削除しコントラスト情報を保持. ([#160](https://github.com/kurorosu/pochivision/pull/160))
- GLCM の `except Exception` を削除しエラーをログ出力後に再送出. ([#161](https://github.com/kurorosu/pochivision/pull/161))
- GLCM の NaN/Inf を NaN として保持し警告ログを出力. ([#162](https://github.com/kurorosu/pochivision/pull/162))
- HLAC の `except Exception` を削除しエラーをログ出力後に再送出. ([#174](https://github.com/kurorosu/pochivision/pull/174))
- HLAC のゼロパディングを削除し境界の特徴量減衰を解消. ([#175](https://github.com/kurorosu/pochivision/pull/175))
- HLAC `_get_default_results` のフォールバック特徴量数を 45 → 37 に修正. ([#178](https://github.com/kurorosu/pochivision/pull/178))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
