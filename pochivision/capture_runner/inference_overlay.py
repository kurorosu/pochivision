"""推論結果オーバーレイモジュール."""

from dataclasses import dataclass

import cv2
import numpy as np

from pochivision.capture_runner import _overlay_colors
from pochivision.request.api.inference.models import PredictResponse


@dataclass
class InferenceContext:
    """推論オーバーレイに表示する静的コンテキスト情報.

    Attributes:
        server_url: 推論 API サーバーの URL.
        image_size: 推論画像サイズ ("WxH" 形式, またはリサイズなしの場合 None).
    """

    server_url: str
    image_size: str | None = None


class InferenceOverlay:
    """プレビュー画面に推論結果を複数行で描画するクラス.

    左上に以下を表示する:
    - 推論結果 (クラス名)
    - 信頼度 (%)
    - 推論時間 (e2e_time_ms)
    - RTT 込時間 (rtt_ms)
    - 推論画像サイズ
    - サーバー URL

    推論失敗時はエラーメッセージを表示する.
    """

    CONFIDENCE_HIGH = 0.7
    CONFIDENCE_LOW = 0.4

    INFERRING_TEXT = "Inferring..."
    META_COLOR: tuple[int, int, int] = _overlay_colors.META_COLOR
    ERROR_COLOR: tuple[int, int, int] = _overlay_colors.ERROR_COLOR
    HIGH_COLOR: tuple[int, int, int] = (0, 200, 0)
    MEDIUM_COLOR: tuple[int, int, int] = (0, 200, 200)
    # 信頼度低を示す赤. 現状 ERROR_COLOR と同値だが意味 (有効な低信頼推論 vs 通信エラー)
    # が異なるため独立定義する. 片方だけ変えたい将来の変更に備える.
    LOW_COLOR: tuple[int, int, int] = (0, 0, 200)

    def __init__(self, context: InferenceContext | None = None) -> None:
        """コンストラクタ.

        Args:
            context: サーバー URL や画像サイズなどの静的情報.
        """
        self.result: PredictResponse | None = None
        self.error_message: str | None = None
        self.context = context
        self._inferring = False

    def update(self, result: PredictResponse) -> None:
        """推論結果を更新する.

        エラーメッセージがある場合はクリアする.

        Args:
            result: 推論 API からのレスポンス.
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
        """推論結果とエラーメッセージをクリアする."""
        self.result = None
        self.error_message = None

    def set_inferring(self, inferring: bool) -> None:
        """推論中フラグを設定する.

        Args:
            inferring: 推論中かどうか.
        """
        self._inferring = inferring

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """フレーム左上に推論情報を描画する.

        Args:
            frame: 描画先のフレーム. このフレームを直接変更する.

        Returns:
            推論情報が描画されたフレーム.
        """
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

    def _draw_result(self, frame: np.ndarray, result: PredictResponse) -> None:
        """推論結果を複数行で描画する.

        Args:
            frame: 描画先のフレーム.
            result: 推論結果.
        """
        color = self._get_color(result.confidence)
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"Result: {result.class_name}", color),
            (f"Confidence: {result.confidence * 100:.1f}%", color),
            (f"Inference: {result.e2e_time_ms:.1f}ms", self.META_COLOR),
            (f"RTT: {result.rtt_ms:.1f}ms", self.META_COLOR),
        ]

        if self.context and self.context.image_size:
            lines.append((f"ImageSize: {self.context.image_size}", self.META_COLOR))
        if self.context:
            lines.append((f"Server: {self.context.server_url}", self.META_COLOR))

        y = 30
        for text, c in lines:
            self._draw_text(frame, text, c, y=y)
            y += 25

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
        """フレーム左上にテキストを描画する.

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

    def _get_color(self, confidence: float) -> tuple[int, int, int]:
        """信頼度に応じた BGR カラーを返す.

        Args:
            confidence: 信頼度 (0.0-1.0).

        Returns:
            BGR カラータプル.
        """
        if confidence >= self.CONFIDENCE_HIGH:
            return self.HIGH_COLOR
        if confidence >= self.CONFIDENCE_LOW:
            return self.MEDIUM_COLOR
        return self.LOW_COLOR
