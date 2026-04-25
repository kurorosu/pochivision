"""DetectionOverlay のテスト."""

import numpy as np

from pochivision.capture_runner.detection_overlay import (
    DetectionContext,
    DetectionOverlay,
)
from pochivision.request.api.detection.models import Detection, DetectionResponse


def _make_detection(
    class_id: int = 0,
    class_name: str = "pochi",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 100.0, 200.0),
) -> Detection:
    """テスト用の Detection を生成する."""
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )


def _make_response(
    detections: tuple[Detection, ...] | None = None,
    e2e_time_ms: float = 12.3,
    rtt_ms: float = 65.1,
    total_ms: float = 70.0,
    backend: str = "onnx",
    phase_times_ms: dict[str, float] | None = None,
) -> DetectionResponse:
    """テスト用の DetectionResponse を生成する.

    - `detections=None` (デフォルト): 代表的な Detection を 1 件含む応答を返す
    - `detections=()`: 空の検出応答を返す (メタ情報描画のみ検証したい場合に使う)
    - `detections=(det1, det2, ...)`: 明示的に指定した検出だけを含む

    暗黙のデフォルト検出に依存するテストは呼び出し側で意図を明記すること.
    """
    if detections is None:
        detections = (_make_detection(),)
    return DetectionResponse(
        detections=detections,
        e2e_time_ms=e2e_time_ms,
        backend=backend,
        rtt_ms=rtt_ms,
        total_ms=total_ms,
        phase_times_ms=phase_times_ms or {},
    )


def _make_context() -> DetectionContext:
    """テスト用の DetectionContext を生成する."""
    return DetectionContext(server_url="http://localhost:8000", image_size="640x640")


class TestState:
    """状態遷移のテスト."""

    def test_initial_state(self):
        overlay = DetectionOverlay()
        assert overlay.result is None
        assert overlay.error_message is None
        assert overlay._inferring is False

    def test_update(self):
        overlay = DetectionOverlay()
        result = _make_response()
        overlay.update(result)
        assert overlay.result is result
        assert overlay.error_message is None

    def test_update_clears_error(self):
        overlay = DetectionOverlay()
        overlay.set_error("connection error")
        overlay.update(_make_response())
        assert overlay.result is not None
        assert overlay.error_message is None

    def test_set_error_clears_result(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response())
        overlay.set_error("boom")
        assert overlay.error_message == "boom"
        assert overlay.result is None

    def test_clear(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response())
        overlay.clear()
        assert overlay.result is None
        assert overlay.error_message is None

    def test_set_inferring(self):
        overlay = DetectionOverlay()
        overlay.set_inferring(True)
        assert overlay._inferring is True

    def test_error_then_result_clears_error(self):
        overlay = DetectionOverlay()
        overlay.set_error("connection refused")
        overlay.update(_make_response())
        assert overlay.error_message is None
        assert overlay.result is not None


class TestGetColor:
    """_get_color (決定的色割当) のテスト."""

    def test_same_class_id_same_color(self):
        overlay = DetectionOverlay()
        assert overlay._get_color(0) == overlay._get_color(0)

    def test_different_class_id_different_color(self):
        overlay = DetectionOverlay()
        assert overlay._get_color(0) != overlay._get_color(1)

    def test_class_id_wraps_around_palette(self):
        overlay = DetectionOverlay()
        palette_size = len(DetectionOverlay.PALETTE)
        assert overlay._get_color(0) == overlay._get_color(palette_size)


class TestDraw:
    """draw のテスト."""

    def test_no_result_returns_frame_unchanged(self):
        overlay = DetectionOverlay()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        original = frame.copy()
        result = overlay.draw(frame)
        np.testing.assert_array_equal(result, original)

    def test_inferring_draws_text_when_no_result(self):
        overlay = DetectionOverlay()
        overlay.set_inferring(True)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_error_draws_on_frame(self):
        overlay = DetectionOverlay(context=_make_context())
        overlay.set_error("connection refused")
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_error_draws_without_context(self):
        """context なしでも error メッセージが描画される."""
        overlay = DetectionOverlay()
        overlay.set_error("boom")
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0

    def test_result_takes_precedence_over_inferring(self):
        """result がある状態で inferring=True でも 'Detecting...' ではなく result を描画する."""
        overlay = DetectionOverlay()
        overlay.update(_make_response(detections=()))
        overlay.set_inferring(True)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        # 結果のメタが出ており, それは最終状態での result 優先の証左
        assert result.sum() > 0
        assert overlay.result is not None

    def test_empty_detections_draws_meta_only(self):
        overlay = DetectionOverlay()
        # detections=() を明示して bbox 描画を発生させずメタ情報のみ描画することを検証
        overlay.update(_make_response(detections=()))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        assert result.sum() > 0
        assert overlay.result is not None
        assert overlay.result.detections == ()

    def test_bbox_drawn_within_frame(self):
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(50.0, 60.0, 150.0, 180.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        # bbox 領域周辺にピクセルが書かれているか
        assert result[60:180, 50:150].sum() > 0

    def test_bbox_color_matches_class_id(self):
        overlay = DetectionOverlay()
        det = _make_detection(class_id=2, bbox=(20.0, 20.0, 100.0, 100.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        expected_color = overlay._get_color(2)
        # bbox 辺上のどこかに期待色が存在する
        b, g, r = expected_color
        match = (result[..., 0] == b) & (result[..., 1] == g) & (result[..., 2] == r)
        assert match.any()

    def test_multiple_detections_different_colors(self):
        overlay = DetectionOverlay()
        dets = (
            _make_detection(class_id=0, bbox=(10.0, 10.0, 80.0, 80.0)),
            _make_detection(class_id=1, bbox=(100.0, 100.0, 180.0, 180.0)),
        )
        overlay.update(_make_response(detections=dets))
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        c0 = overlay._get_color(0)
        c1 = overlay._get_color(1)
        has_c0 = (
            (result[..., 0] == c0[0])
            & (result[..., 1] == c0[1])
            & (result[..., 2] == c0[2])
        ).any()
        has_c1 = (
            (result[..., 0] == c1[0])
            & (result[..., 1] == c1[1])
            & (result[..., 2] == c1[2])
        ).any()
        assert has_c0 and has_c1

    def test_label_y_at_top_of_frame_does_not_crash(self):
        """bbox が画面上端に張り付いてもラベル描画で例外が出ない."""
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(0.0, 0.0, 50.0, 30.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_inverted_bbox_is_skipped(self):
        overlay = DetectionOverlay()
        # メタ情報描画領域 (左上) と干渉しないよう右下に inverted bbox を置く
        det = _make_detection(bbox=(350.0, 350.0, 250.0, 250.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        original = frame.copy()
        result = overlay.draw(frame)
        # 反転 bbox が描画スキップされ, 該当領域に描画がないことを確認
        np.testing.assert_array_equal(
            result[245:355, 245:355], original[245:355, 245:355]
        )

    def test_nan_bbox_does_not_crash(self):
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(float("nan"), 10.0, 100.0, 100.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_inf_bbox_does_not_crash(self):
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(0.0, 0.0, float("inf"), 100.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_bbox_completely_outside_frame_is_skipped(self):
        overlay = DetectionOverlay()
        det = _make_detection(bbox=(500.0, 500.0, 600.0, 600.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_non_bgr_frame_returned_unchanged(self):
        overlay = DetectionOverlay()
        overlay.update(_make_response())
        frame_gray = np.zeros((200, 200), dtype=np.uint8)
        original = frame_gray.copy()
        result = overlay.draw(frame_gray)
        np.testing.assert_array_equal(result, original)

    def test_confidence_zero_boundary(self):
        overlay = DetectionOverlay()
        det = _make_detection(confidence=0.0, bbox=(10.0, 10.0, 80.0, 80.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_confidence_one_boundary(self):
        overlay = DetectionOverlay()
        det = _make_detection(confidence=1.0, bbox=(10.0, 10.0, 80.0, 80.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay.draw(frame)

    def test_text_outline_dark_pixels_drawn(self):
        """テキストのアウトライン (黒ストローク) が描画されていることを検証.

        実装は `cv2.putText` を 2 回呼び出す (outline: color=(0,0,0),
        thickness=4 -> text: color=META_COLOR, thickness=2). LINE_AA による
        アンチエイリアスで白背景と混合するため完全な (0, 0, 0) は中心に限られ,
        ストローク中心付近の十分に暗いピクセルで存在確認する.

        閾値 50 は「白 255 からの強い暗化があれば outline とみなせる」目安値で,
        アンチエイリアスによる縁 (中間色) が通常 100-200 程度になる性質に基づく.
        検出 bbox が描く塗りつぶしの影響を排除するため detections=() を使う.
        """
        overlay = DetectionOverlay()
        overlay.update(_make_response(detections=()))
        frame = np.full((200, 400, 3), 255, dtype=np.uint8)
        result = overlay.draw(frame)
        dark_mask = (
            (result[..., 0] < 50) & (result[..., 1] < 50) & (result[..., 2] < 50)
        )
        assert dark_mask.any()

    def test_text_outline_visible_on_black_background(self):
        """黒背景でもアウトライン描画が text の色変化として現れることを検証."""
        overlay = DetectionOverlay()
        overlay.update(_make_response(detections=()))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        # 黒背景ではテキスト本体 (META_COLOR) が唯一の非黒ピクセル源
        b, g, r = DetectionOverlay.META_COLOR
        bright_mask = (
            (result[..., 0] > b // 2)
            & (result[..., 1] > g // 2)
            & (result[..., 2] > r // 2)
        )
        assert bright_mask.any()


class TestBuildMetaLines:
    """`_build_meta_lines` のテスト (表示テキスト組み立て)."""

    def _texts(self, overlay: DetectionOverlay, result: DetectionResponse) -> list[str]:
        return [text for text, _ in overlay._build_meta_lines(result)]

    def test_e2e_label_replaces_inference_label(self):
        """旧 `Inference:` ラベルは使われず `E2E:` で e2e_time_ms を表示する."""
        overlay = DetectionOverlay()
        texts = self._texts(overlay, _make_response(detections=(), e2e_time_ms=12.3))

        assert "E2E: 12.3ms" in texts
        assert not any(t.startswith("Inference:") for t in texts)

    def test_breakdown_absent_when_phase_times_empty(self):
        """phase_times_ms が空なら `- ` プレフィックスの内訳行は出ない."""
        overlay = DetectionOverlay()
        texts = self._texts(overlay, _make_response(detections=()))
        assert not any(t.startswith("- ") for t in texts)

    def test_infer_line_shown_when_phase_times_present(self):
        """`pipeline_inference_ms` があれば内訳サブ行 `- Infer:` が追加される."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={"pipeline_inference_ms": 8.2},
            ),
        )
        assert "- Infer: 8.2ms" in texts

    def test_infer_line_ignores_gpu_phase_field(self):
        """`pipeline_inference_gpu_ms` が併せて返っても画面には併記しない."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={
                    "pipeline_inference_ms": 8.2,
                    "pipeline_inference_gpu_ms": 7.9,
                },
            ),
        )
        assert "- Infer: 8.2ms" in texts
        assert not any("GPU" in t for t in texts)

    def test_pre_post_lines_shown_when_present(self):
        """`pipeline_preprocess_ms` / `pipeline_postprocess_ms` があれば `- Pre:` / `- Post:` を表示."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={
                    "pipeline_preprocess_ms": 1.1,
                    "pipeline_postprocess_ms": 0.5,
                },
            ),
        )
        assert "- Pre: 1.1ms" in texts
        assert "- Post: 0.5ms" in texts

    def test_api_phase_lines_absent_when_keys_missing(self):
        """`api_preprocess_ms` / `api_postprocess_ms` が無ければ APIpre/APIpost 行は出ない."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={"pipeline_inference_ms": 8.2},
            ),
        )
        assert not any(t.startswith("- APIpre:") for t in texts)
        assert not any(t.startswith("- APIpost:") for t in texts)

    def test_api_phase_lines_shown_when_present(self):
        """`api_preprocess_ms` / `api_postprocess_ms` があれば `- APIpre:` / `- APIpost:` 行が追加される."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={
                    "api_preprocess_ms": 1.4,
                    "pipeline_inference_ms": 8.2,
                    "api_postprocess_ms": 0.9,
                },
            ),
        )
        assert "- APIpre: 1.4ms" in texts
        assert "- APIpost: 0.9ms" in texts

    def test_order_is_chronological_breakdown(self):
        """内訳サブ行は時系列順: Detections → Total → E2E → -APIpre → -Pre → -Infer → -Post → -APIpost → RTT → Backend."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={
                    "api_preprocess_ms": 1.4,
                    "pipeline_preprocess_ms": 1.1,
                    "pipeline_inference_ms": 8.2,
                    "pipeline_postprocess_ms": 0.5,
                    "api_postprocess_ms": 0.9,
                },
            ),
        )
        # 先頭 10 行の順序を検証 (Total が E2E の直前に固定で挿入される)
        assert texts[0].startswith("Detections:")
        assert texts[1].startswith("Total:")
        assert texts[2].startswith("E2E:")
        assert texts[3].startswith("- APIpre:")
        assert texts[4].startswith("- Pre:")
        assert texts[5].startswith("- Infer:")
        assert texts[6].startswith("- Post:")
        assert texts[7].startswith("- APIpost:")
        assert texts[8].startswith("RTT:")
        assert texts[9].startswith("Backend:")

    def test_order_with_only_infer_breakdown(self):
        """内訳が Infer のみ存在: Detections → Total → E2E → -Infer → RTT → Backend."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay,
            _make_response(
                detections=(),
                phase_times_ms={"pipeline_inference_ms": 8.2},
            ),
        )
        # 先頭 6 行の順序を検証
        assert texts[0].startswith("Detections:")
        assert texts[1].startswith("Total:")
        assert texts[2].startswith("E2E:")
        assert texts[3].startswith("- Infer:")
        assert texts[4].startswith("RTT:")
        assert texts[5].startswith("Backend:")

    def test_total_line_displayed_before_e2e(self):
        """`Total: X.Xms` 行が E2E の直前に固定で表示される."""
        overlay = DetectionOverlay()
        texts = self._texts(
            overlay, _make_response(detections=(), total_ms=15.0, e2e_time_ms=10.0)
        )

        assert "Total: 15.0ms" in texts
        total_idx = texts.index("Total: 15.0ms")
        e2e_idx = next(i for i, t in enumerate(texts) if t.startswith("E2E:"))
        assert total_idx + 1 == e2e_idx

    def test_total_line_shown_even_without_phase_times(self):
        """phase_times_ms が空でも Total 行は固定で表示される."""
        overlay = DetectionOverlay()
        texts = self._texts(overlay, _make_response(detections=(), total_ms=20.5))

        assert any(t.startswith("Total: 20.5") for t in texts)


class TestPreviewScale:
    """`set_preview_scale` による bbox 描画位置の補正テスト."""

    def test_default_scale_is_one(self):
        """初期値はスケール 1.0 (補正なし) で従来動作を維持."""
        overlay = DetectionOverlay()
        # bbox 50-150, 60-180 をそのまま描画する.
        det = _make_detection(bbox=(50.0, 60.0, 150.0, 180.0))
        overlay.update(_make_response(detections=(det,)))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = overlay.draw(frame)
        # オリジナル位置に bbox が描画される.
        assert result[60:180, 50:150].sum() > 0

    def test_scale_shrinks_bbox_to_preview(self):
        """frame_w=400 / preview_w=200 のとき bbox が 1/2 に縮小描画される."""
        overlay = DetectionOverlay()
        # 送信フレーム座標 bbox: (100, 80, 300, 160).
        det = _make_detection(bbox=(100.0, 80.0, 300.0, 160.0))
        overlay.update(_make_response(detections=(det,)))
        # frame_w=400, preview_w=200 → scale=0.5.
        overlay.set_preview_scale(frame_w=400, preview_w=200)
        # preview サイズ (200x100) に描画.
        preview = np.zeros((100, 200, 3), dtype=np.uint8)
        result = overlay.draw(preview)
        # 縮小後の bbox: (50, 40, 150, 80) 領域に描画.
        assert result[40:80, 50:150].sum() > 0
        # 縮小前の元位置 (300 など) は preview 範囲外なので描画されない.
        # 念のため 0-50 / 150-200 の bbox 周辺ではない領域に矩形辺がないことは
        # 厳密検証しない (色パレット衝突を避ける).

    def test_scale_zero_frame_w_keeps_previous_scale(self):
        """frame_w=0 を渡された場合は scale を更新せず従来値を維持."""
        overlay = DetectionOverlay()
        overlay.set_preview_scale(frame_w=400, preview_w=200)  # scale=0.5
        overlay.set_preview_scale(frame_w=0, preview_w=100)  # 無視される
        # scale=0.5 のままなので bbox が 1/2 に縮小描画されることを確認.
        det = _make_detection(bbox=(100.0, 80.0, 300.0, 160.0))
        overlay.update(_make_response(detections=(det,)))
        preview = np.zeros((100, 200, 3), dtype=np.uint8)
        result = overlay.draw(preview)
        assert result[40:80, 50:150].sum() > 0
