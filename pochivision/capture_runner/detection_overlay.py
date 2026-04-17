"""検出結果オーバーレイモジュール."""

from dataclasses import dataclass

import cv2
import numpy as np

from pochivision.request.api.detection.models import Detection, DetectionResponse


@dataclass
class DetectionContext:
    """検出オーバーレイに表示する静的コンテキスト情報.

    Attributes:
        server_url: 検出 API サーバーの URL.
        image_size: 送信画像サイズ ("WxH" 形式, またはリサイズなしの場合 None).
    """

    server_url: str
    image_size: str | None = None


class DetectionOverlay:
    """プレビュー画面に検出結果 (複数 bbox) を描画するクラス.

    各検出を bbox + `"class_name conf"` のラベルで描画し,
    画面左上に以下を表示する:
    - 検出数
    - サーバー側推論時間 (e2e_time_ms)
    - RTT (rtt_ms)
    - バックエンド
    - 送信画像サイズ (context 経由, 任意)
    - サーバー URL (context 経由, 任意)

    検出失敗時はエラーメッセージを表示する.

    クラスごとの色は class_id から決定的に割り当てる.
    """

    INFERRING_TEXT = "Detecting..."
    META_COLOR: tuple[int, int, int] = (200, 200, 200)
    ERROR_COLOR: tuple[int, int, int] = (0, 0, 200)

    # 決定的な色パレット (BGR). class_id % len(PALETTE) で割当.
    PALETTE: tuple[tuple[int, int, int], ...] = (
        (0, 200, 0),  # 緑
        (0, 200, 200),  # 黄
        (200, 100, 0),  # 青緑
        (0, 100, 200),  # 橙
        (200, 0, 200),  # マゼンタ
        (200, 0, 0),  # 青
        (100, 200, 0),  # 若緑
        (0, 0, 200),  # 赤
    )

    def __init__(self, context: DetectionContext | None = None) -> None:
        """コンストラクタ.

        Args:
            context: サーバー URL や画像サイズなどの静的情報.
        """
        self.result: DetectionResponse | None = None
        self.error_message: str | None = None
        self.context = context
        self._inferring = False

    def update(self, result: DetectionResponse) -> None:
        """検出結果を更新する.

        エラーメッセージがある場合はクリアする.

        Args:
            result: 検出 API からのレスポンス.
        """
        self.result = result
        self.error_message = None

    def set_error(self, message: str) -> None:
        """エラーメッセージを設定する.

        Args:
            message: 表示するエラーメッセージ.
        """
        self.error_message = message
        self.result = None

    def clear(self) -> None:
        """検出結果とエラーメッセージをクリアする."""
        self.result = None
        self.error_message = None

    def set_inferring(self, inferring: bool) -> None:
        """推論中フラグを設定する.

        Args:
            inferring: 推論中かどうか.
        """
        self._inferring = inferring

    def _get_color(self, class_id: int) -> tuple[int, int, int]:
        """class_id に対応する決定的な BGR 色を返す.

        Args:
            class_id: クラス ID.

        Returns:
            BGR カラータプル.
        """
        return self.PALETTE[class_id % len(self.PALETTE)]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """フレームに bbox とメタ情報を描画する.

        BGR 3 チャネル以外のフレームは描画せずそのまま返す.

        Note:
            本メソッドは `self.result` / `self.error_message` / `self._inferring`
            を読み取るため, 複数スレッドから `update()` / `set_error()` /
            `set_inferring()` と並行呼び出しすると TOCTOU 競合の可能性がある.
            呼び出し側でシリアライズすること (ランタイム統合で lock を導入予定:
            #402).

        Args:
            frame: 描画先のフレーム. このフレームを直接変更する.

        Returns:
            検出結果が描画されたフレーム.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        inferring = self._inferring
        result = self.result
        error = self.error_message

        if inferring and result is None and error is None:
            self._draw_text(frame, self.INFERRING_TEXT, self.META_COLOR, y=30)
            return frame

        if error is not None:
            self._draw_error(frame, error)
            return frame

        if result is None:
            return frame

        self._draw_result(frame, result)
        return frame

    def _draw_result(self, frame: np.ndarray, result: DetectionResponse) -> None:
        """検出結果 (bbox + メタ情報) を描画する.

        Args:
            frame: 描画先のフレーム.
            result: 検出結果.
        """
        for det in result.detections:
            self._draw_bbox(frame, det)

        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"Detections: {len(result.detections)}", self.META_COLOR),
            (f"Inference: {result.e2e_time_ms:.1f}ms", self.META_COLOR),
            (f"RTT: {result.rtt_ms:.1f}ms", self.META_COLOR),
            (f"Backend: {result.backend}", self.META_COLOR),
        ]
        if self.context and self.context.image_size:
            lines.append((f"ImageSize: {self.context.image_size}", self.META_COLOR))
        if self.context:
            lines.append((f"Server: {self.context.server_url}", self.META_COLOR))

        y = 30
        for text, c in lines:
            self._draw_text(frame, text, c, y=y)
            y += 25

    def _draw_bbox(self, frame: np.ndarray, det: Detection) -> None:
        """1 つの検出を bbox + ラベルで描画する.

        NaN / Inf を含む bbox, x2 <= x1 または y2 <= y1 の反転 bbox,
        完全にフレーム外の bbox はスキップする.

        Args:
            frame: 描画先のフレーム.
            det: 検出結果 1 件.
        """
        bbox = det.bbox
        if not all(np.isfinite(v) for v in bbox):
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            return
        if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
            return

        color = self._get_color(det.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # ラベル矩形がフレーム外に飛び出さないようクリップ
        if y1 - 6 - th >= 0:
            label_y = y1 - 6
        else:
            label_y = y1 + th + 6
        rect_top = max(0, label_y - th - 4)
        rect_bottom = min(h - 1, label_y + baseline)
        rect_left = max(0, x1)
        rect_right = min(w - 1, x1 + tw + 4)
        if rect_top >= rect_bottom or rect_left >= rect_right:
            return

        cv2.rectangle(
            frame,
            (rect_left, rect_top),
            (rect_right, rect_bottom),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (rect_left + 2, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_error(self, frame: np.ndarray, error: str) -> None:
        """エラーメッセージを描画する.

        Args:
            frame: 描画先のフレーム.
            error: エラーメッセージ.
        """
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"Error: {error}", self.ERROR_COLOR),
        ]
        if self.context:
            lines.append((f"Server: {self.context.server_url}", self.META_COLOR))

        y = 30
        for text, c in lines:
            self._draw_text(frame, text, c, y=y)
            y += 25

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
        y: int,
    ) -> None:
        """フレーム左上にアウトライン付きテキストを描画する.

        Args:
            frame: 描画先のフレーム.
            text: 表示テキスト.
            color: BGR カラータプル.
            y: 描画 Y 座標.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        outline_thickness = thickness + 2
        x = 10

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
