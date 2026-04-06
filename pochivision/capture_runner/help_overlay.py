"""ライブプレビュー用ヘルプオーバーレイモジュール."""

import cv2
import numpy as np


class HelpOverlay:
    """プレビュー画面にキーバインドのヘルプを描画するクラス.

    黒文字 + 白縁のテキストで画面下部に一行表示する.
    プレビュー表示のみに適用し, 画像処理・保存パイプラインには影響しない.
    """

    HELP_TEXT = (
        "c:Capture  r:Rec  t:Stop  s:Settings" "  i:Infer  d:ClearROI  h:Help  q:Quit"
    )

    def __init__(self) -> None:
        """HelpOverlayのコンストラクタ."""
        self.visible = True

    def toggle(self) -> None:
        """ヘルプ表示のオン/オフを切り替える."""
        self.visible = not self.visible

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """フレーム下部にヘルプテキストを一行描画する.

        Args:
            frame: 描画先のフレーム. このフレームを直接変更する.

        Returns:
            ヘルプテキストが描画されたフレーム.
        """
        if not self.visible:
            return frame

        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        outline_thickness = thickness + 2
        x = 10
        y = h - 12

        # 白縁 (outline)
        cv2.putText(
            frame,
            self.HELP_TEXT,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            outline_thickness,
            cv2.LINE_AA,
        )
        # 黒文字 (foreground)
        cv2.putText(
            frame,
            self.HELP_TEXT,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        return frame
