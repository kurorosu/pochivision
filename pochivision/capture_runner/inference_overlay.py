"""推論結果オーバーレイモジュール."""

import cv2
import numpy as np

from pochivision.request.api.inference.models import PredictResponse


class InferenceOverlay:
    """プレビュー画面に推論結果を描画するクラス.

    フレーム左上にクラス名, 信頼度(%), 推論時間(ms) を表示する.
    信頼度に応じて文字色が変化する (緑: 高, 黄: 中, 赤: 低).
    """

    CONFIDENCE_HIGH = 0.7
    CONFIDENCE_LOW = 0.4

    INFERRING_TEXT = "Inferring..."

    def __init__(self) -> None:
        """コンストラクタ."""
        self.result: PredictResponse | None = None
        self._inferring = False

    def update(self, result: PredictResponse) -> None:
        """推論結果を更新する.

        Args:
            result: 推論 API からのレスポンス.
        """
        self.result = result

    def clear(self) -> None:
        """推論結果をクリアする."""
        self.result = None

    def set_inferring(self, inferring: bool) -> None:
        """推論中フラグを設定する.

        Args:
            inferring: 推論中かどうか.
        """
        self._inferring = inferring

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """フレーム左上に推論結果を描画する.

        Args:
            frame: 描画先のフレーム. このフレームを直接変更する.

        Returns:
            推論結果が描画されたフレーム.
        """
        if self._inferring and self.result is None:
            self._draw_text(frame, self.INFERRING_TEXT, (200, 200, 200))
            return frame

        if self.result is None:
            return frame

        text = (
            f"{self.result.class_name}  "
            f"{self.result.confidence * 100:.1f}%  "
            f"{self.result.e2e_time_ms:.1f}ms"
        )
        color = self._get_color(self.result.confidence)
        self._draw_text(frame, text, color)

        return frame

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        """フレーム左上にテキストを描画する.

        Args:
            frame: 描画先のフレーム.
            text: 表示テキスト.
            color: BGR カラータプル.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        outline_thickness = thickness + 2
        x = 10
        y = 30

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
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
            return (0, 200, 0)  # 緑
        if confidence >= self.CONFIDENCE_LOW:
            return (0, 200, 200)  # 黄
        return (0, 0, 200)  # 赤
