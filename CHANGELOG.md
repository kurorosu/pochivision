# Changelog

このファイルは最新の changelog を保持します.
最新でなくなった履歴は `changelogs/` 配下へ移動して管理します.

## [Unreleased]

### Added
- 無し

### Changed
- 無し

### Fixed
- 無し

### Removed
- 無し

## [0.1.4] - 2026-03-29

### Added
- `docs/swt_features.md` を追加. SWT 特徴量抽出器の全特徴量・パラメータ・設計制約の解説. (NA.)
- SWT の振る舞いテスト 18 件を追加. ([#208](https://github.com/kurorosu/pochivision/pull/208))

### Changed
- SWT マルチスケールのレベルラベリングを修正し, L1=高周波, LN=低周波に変更. ([#194](https://github.com/kurorosu/pochivision/pull/194))
- SWT `multiscale=False` 時のレベル選択と docstring を明確化. ([#195](https://github.com/kurorosu/pochivision/pull/195))
- SWT docstring の LH/HL 説明を標準ウェーブレット用語に修正. `get_feature_names` の config 依存性を docstring に明記. ([#198](https://github.com/kurorosu/pochivision/pull/198))

### Fixed
- SWT エントロピー計算を `np.bincount` ベースに変更し, 狭い値域でのクラッシュを解消. ([#193](https://github.com/kurorosu/pochivision/pull/193))
- SWT の dtype 正規化を統一し, uint8 と float32 (0-255) で同じ特徴量が得られるよう修正. ([#196](https://github.com/kurorosu/pochivision/pull/196))
- SWT の `except Exception` を `LogManager` ログ出力 + `raise` に変更. ([#197](https://github.com/kurorosu/pochivision/pull/197))
- SWT 極小画像 (4x4 未満) のガードを追加. ([#198](https://github.com/kurorosu/pochivision/pull/198))

### Removed
- 無し

## Archived Changelogs

過去のバージョン履歴は [`changelogs/`](changelogs/) ディレクトリに保管しています.
