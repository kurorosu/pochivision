"""カメラプレビュー・キャプチャ用のコントローラークラスを提供するモジュール."""

import platform
import time

import cv2
import numpy as np

from pochivision.capture_runner.help_overlay import HelpOverlay
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.constants import DEFAULT_PREVIEW_HEIGHT, DEFAULT_PREVIEW_WIDTH
from pochivision.core import PipelineExecutor
from pochivision.exceptions import VisionCaptureError


class LivePreviewRunner:
    """
    カメラプレビューとキャプチャ機能を統合したコントローラークラス.

    録画機能も統合し、リアルタイムでの録画とキャプチャの同時実行をサポートします。

    Attributes:
        cap (cv2.VideoCapture): カメラオブジェクト.
        pipeline: キャプチャ後に処理を行うパイプラインインスタンス.
        recording_manager (RecordingManager | None): 録画機能を管理するマネージャー.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        pipeline: PipelineExecutor,
        recording_manager: RecordingManager | None = None,
        preview_size: tuple[int, int] = (DEFAULT_PREVIEW_WIDTH, DEFAULT_PREVIEW_HEIGHT),
    ) -> None:
        """
        LivePreviewRunnerを初期化する.

        Args:
            cap: 初期化済みの cv2.VideoCapture オブジェクト.
            pipeline: .run(image) を持つ画像処理パイプラインインスタンス.
            recording_manager: 録画機能を管理するマネージャー（オプション）.
            preview_size: プレビューウィンドウの表示サイズ (width, height).
        """
        self.cap = cap
        self.pipeline = pipeline
        self.recording_manager = recording_manager
        self.preview_size = preview_size
        self.os_name = platform.system()
        self.logger = LogManager().get_logger()
        self.help_overlay = HelpOverlay()

    def _resize_for_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        アスペクト比を維持しつつ, preview_size に収まるようリサイズする.

        Args:
            frame: 入力フレーム.

        Returns:
            np.ndarray: リサイズされたフレーム.
        """
        h, w = frame.shape[:2]
        max_w, max_h = self.preview_size
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    def _measure_actual_fps(self, duration: float = 2.0) -> float:
        """
        実際のフレームレートを測定します.

        Args:
            duration (float): 測定時間（秒）

        Returns:
            float: 実測されたFPS
        """
        self.logger.info(f"Measuring actual FPS for {duration} seconds...")

        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            ret, frame = self.cap.read()
            if ret:
                frame_count += 1
                # プレビュー表示も含めて測定（実際の録画環境に近づける）
                cv2.imshow("Live View", self._resize_for_preview(frame))
                cv2.waitKey(1)

        actual_duration = time.time() - start_time
        measured_fps = frame_count / actual_duration if actual_duration > 0 else 0.0

        self.logger.info(
            f"Measured FPS: {measured_fps:.2f} "
            f"({frame_count} frames in {actual_duration:.2f}s)"
        )
        return measured_fps

    def run(self) -> None:
        """
        ライブビューを開始し、各種キー操作を処理します.

        キー操作:
        - 'c': キャプチャ
        - 'r': 録画開始
        - 't': 録画停止
        - 's': カメラ設定
        - 'h': ヘルプオーバーレイ表示/非表示
        - 'q': 終了

        Raises:
            VisionCaptureError: カメラ設定ダイアログが非対応のOSで呼び出された場合.
        """
        self.logger.info("Starting live preview with key controls:")
        self.logger.info(
            "Press 'c' to capture, 'r' to start recording, 't' to stop recording,"
        )
        self.logger.info("'s' for camera settings, 'h' for help, 'q' to quit.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # 録画中の場合、フレームを録画ファイルに追加
                if (
                    self.recording_manager
                    and self.recording_manager.get_recording_status()
                ):
                    self.recording_manager.add_frame(frame)

                # プレビュー表示 (リサイズ後にオーバーレイを描画)
                preview = self._resize_for_preview(frame)
                self.help_overlay.draw(preview)
                cv2.imshow("Live View", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    # キャプチャ処理
                    snapshot = frame.copy()
                    self.pipeline.run(snapshot)
                elif key == ord("r"):
                    # 録画開始
                    self._start_recording(frame)
                elif key == ord("t"):
                    # 録画停止
                    self._stop_recording()
                elif key == ord("s"):
                    # カメラ設定ダイアログを開く
                    if self.os_name == "Windows":
                        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
                    else:
                        self.logger.warning(
                            f"Camera settings dialog is only supported on Windows. "
                            f"Current OS: {self.os_name}"
                        )
                elif key == ord("h"):
                    # ヘルプオーバーレイのトグル
                    self.help_overlay.toggle()
                elif key == ord("q"):
                    self.logger.info("Quit key pressed, exiting application")
                    break

        except VisionCaptureError as e:
            self.logger.critical(
                f"Critical error occurred in pipeline: "
                f"{e}\nCamera resource will be released."
            )
        finally:
            # クリーンアップ処理
            self._cleanup()

    def _start_recording(self, frame) -> None:
        """
        録画を開始します.

        Args:
            frame: 現在のフレーム（フレームサイズの取得に使用）
        """
        if not self.recording_manager:
            self.logger.warning("Recording is not available (RecordingManager not set)")
            return

        if self.recording_manager.get_recording_status():
            self.logger.info("Recording is already in progress")
            return

        # フレームサイズを取得
        height, width = frame.shape[:2]
        frame_size = (width, height)

        # 実際のフレームレートを測定
        measured_fps = self._measure_actual_fps(duration=2.0)

        # 測定されたFPSを録画FPSとして使用
        recording_fps = measured_fps
        self.logger.info(f"Using measured FPS for recording: {recording_fps:.2f}")

        # 出力ディレクトリを取得 (pipeline から)
        output_dir = self.pipeline.output_dir

        # 録画開始
        success = self.recording_manager.start_recording(
            output_dir=output_dir,
            fps=recording_fps,
            frame_size=frame_size,
        )

        if success:
            self.logger.info(
                f"Recording started successfully at {recording_fps:.2f} FPS"
            )
        else:
            self.logger.error("Failed to start recording")

    def _stop_recording(self) -> None:
        """録画を停止します."""
        if not self.recording_manager:
            self.logger.warning("Recording is not available (RecordingManager not set)")
            return

        if not self.recording_manager.get_recording_status():
            self.logger.info("Recording is not in progress")
            return

        success = self.recording_manager.stop_recording()
        if success:
            self.logger.info("Recording stopped successfully")
        else:
            self.logger.error("Failed to stop recording")

    def _cleanup(self) -> None:
        """リソースのクリーンアップを行います."""
        # 録画が進行中の場合は停止
        if self.recording_manager and self.recording_manager.get_recording_status():
            self.recording_manager.stop_recording()

        # 録画マネージャーのクリーンアップ
        if self.recording_manager:
            self.recording_manager.cleanup()

        # カメラリソースの解放
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Camera resource and windows have been released.")
