"""オーバーレイ共通色定数モジュール.

`DetectionOverlay` と `InferenceOverlay` で共通して使う色を定義する.
各色は BGR (OpenCV のデフォルト) で表現する.
"""

META_COLOR: tuple[int, int, int] = (200, 200, 200)
"""メタ情報 (推論時間, RTT, サーバー URL 等) の表示色. ライトグレー."""

ERROR_COLOR: tuple[int, int, int] = (0, 0, 200)
"""エラーメッセージの表示色. 赤."""
