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

    def __init__(self) -> None:
        """コンストラクタ."""
        self.result: PredictResponse | None = None

    def update(self, result: PredictResponse) -> None:
        """推論結果を更新する.

        Args:
            result: 推論 API からのレスポンス.
        """
        self.result = result

    def clear(self) -> None:
        """推論結果をクリアする."""
        self.result = None

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """フレーム左上に推論結果を描画する.

        Args:
            frame: 描画先のフレーム. このフレームを直接変更する.

        Returns:
            推論結果が描画されたフレーム.
        """
        if self.result is None:
            return frame

        text = (
            f"{self.result.class_name}  "
            f"{self.result.confidence * 100:.1f}%  "
            f"{self.result.processing_time_ms:.1f}ms"
        )

        color = self._get_color(self.result.confidence)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        outline_thickness = thickness + 2
        x = 10
        y = 30

        # 白縁 (outline)
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
        # 色付き文字 (foreground)
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

        return frame

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
