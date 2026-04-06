"""プレビュー上での ROI (Region of Interest) 選択を管理するモジュール."""

import cv2
import numpy as np


class RoiSelector:
    """マウスドラッグで矩形 ROI を選択・管理するクラス.

    プレビュー座標系で ROI を選択し, 元フレーム座標系に変換して
    クロップを行う.

    Attributes:
        roi: 元フレーム座標系の ROI (x, y, w, h), 未設定なら None.
    """

    ROI_COLOR = (0, 255, 0)
    ROI_THICKNESS = 2

    def __init__(self) -> None:
        """コンストラクタ."""
        self.roi: tuple[int, int, int, int] | None = None
        self._dragging = False
        self._start: tuple[int, int] = (0, 0)
        self._end: tuple[int, int] = (0, 0)
        self._preview_scale: float = 1.0

    def set_preview_scale(self, frame_w: int, preview_w: int) -> None:
        """プレビュー座標と元フレーム座標の変換スケールを設定する.

        Args:
            frame_w: 元フレームの幅.
            preview_w: プレビュー表示の幅.
        """
        if preview_w > 0:
            self._preview_scale = frame_w / preview_w

    def mouse_callback(self, event: int, x: int, y: int, *_args: object) -> None:
        """マウスイベントを処理する.

        Args:
            event: マウスイベント種別.
            x: マウス X 座標 (プレビュー座標系).
            y: マウス Y 座標 (プレビュー座標系).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._start = (x, y)
            self._end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self._end = (x, y)
            self._finalize_roi()

    def _finalize_roi(self) -> None:
        """ドラッグ終了後に ROI を確定する.

        選択範囲が小さすぎる (5px 未満) 場合は無視する.
        """
        x1, y1 = self._start
        x2, y2 = self._end

        # 左上・右下に正規化
        px1, py1 = min(x1, x2), min(y1, y2)
        px2, py2 = max(x1, x2), max(y1, y2)

        pw = px2 - px1
        ph = py2 - py1

        if pw < 5 or ph < 5:
            return

        # プレビュー座標 → 元フレーム座標に変換
        s = self._preview_scale
        fx = int(px1 * s)
        fy = int(py1 * s)
        fw = int(pw * s)
        fh = int(ph * s)

        self.roi = (fx, fy, fw, fh)

    def clear(self) -> None:
        """ROI をクリアする."""
        self.roi = None

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """ROI で元フレームをクロップする.

        ROI 未設定の場合は元フレームをそのまま返す.

        Args:
            frame: 元フレーム.

        Returns:
            クロップされたフレーム, または元フレーム.
        """
        if self.roi is None:
            return frame

        x, y, w, h = self.roi
        fh, fw = frame.shape[:2]

        # ROI がフレーム外の場合は元フレームを返す
        if x >= fw or y >= fh:
            return frame

        # フレームサイズ内にクランプ
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)

        if w <= 0 or h <= 0:
            return frame

        return frame[y : y + h, x : x + w]

    def draw(self, preview: np.ndarray) -> None:
        """プレビューフレームに ROI 矩形を描画する.

        ドラッグ中は選択中の矩形を, 確定済みなら確定矩形を描画する.

        Args:
            preview: プレビュー用フレーム (直接変更される).
        """
        if self._dragging:
            x1, y1 = self._start
            x2, y2 = self._end
            cv2.rectangle(
                preview, (x1, y1), (x2, y2), self.ROI_COLOR, self.ROI_THICKNESS
            )
            return

        if self.roi is None:
            return

        # 元フレーム座標 → プレビュー座標に逆変換
        x, y, w, h = self.roi
        s = self._preview_scale
        if s <= 0:
            return
        px1 = int(x / s)
        py1 = int(y / s)
        px2 = int((x + w) / s)
        py2 = int((y + h) / s)
        cv2.rectangle(
            preview, (px1, py1), (px2, py2), self.ROI_COLOR, self.ROI_THICKNESS
        )
