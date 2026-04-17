"""カメラプレビュー・キャプチャ用のコントローラークラスを提供するモジュール."""

import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pochivision.capture_runner.detection_overlay import (
    DetectionContext,
    DetectionOverlay,
)
from pochivision.capture_runner.help_overlay import HelpOverlay
from pochivision.capture_runner.inference_overlay import (
    InferenceContext,
    InferenceOverlay,
)
from pochivision.capture_runner.roi_selector import RoiSelector
from pochivision.capturelib.camera_config_saver import save_camera_config
from pochivision.capturelib.camera_setup import CameraSetup
from pochivision.capturelib.log_manager import LogManager
from pochivision.capturelib.recording_manager import RecordingManager
from pochivision.constants import (
    DEFAULT_DETECTION_FPS,
    DEFAULT_PREVIEW_HEIGHT,
    DEFAULT_PREVIEW_WIDTH,
)
from pochivision.core import PipelineExecutor
from pochivision.exceptions import (
    DetectionConnectionError,
    DetectionError,
    InferenceConnectionError,
    InferenceError,
    VisionCaptureError,
)
from pochivision.request.api.detection.client import DetectionClient
from pochivision.request.api.inference.client import InferenceClient
from pochivision.request.api.inference.csv_writer import InferenceCsvWriter
from pochivision.request.api.inference.models import PredictResponse


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
        inference_client: InferenceClient | None = None,
        camera_setup: CameraSetup | None = None,
        detection_client: DetectionClient | None = None,
        detect_fps: float = DEFAULT_DETECTION_FPS,
    ) -> None:
        """
        LivePreviewRunnerを初期化する.

        `inference_client` と `detection_client` は排他. `detection_client` が
        渡された場合は常時検出ランタイムが有効化され, `inference_client` は
        無視される (ROI 選択も無効化).

        Args:
            cap: 初期化済みの cv2.VideoCapture オブジェクト.
            pipeline: .run(image) を持つ画像処理パイプラインインスタンス.
            recording_manager: 録画機能を管理するマネージャー（オプション）.
            preview_size: プレビューウィンドウの表示サイズ (width, height).
            inference_client: pochitrain 推論 API クライアント（オプション）.
            camera_setup: カメラセットアップ情報（オプション）.
            detection_client: pochidetection 検出 API クライアント（オプション）.
                指定時は detect モードで動作.
            detect_fps: 検出モード時のスロットリング頻度 (Hz).
        """
        self.cap = cap
        self.pipeline = pipeline
        self.recording_manager = recording_manager
        self.preview_size = preview_size
        self.inference_client = inference_client
        self.camera_setup = camera_setup
        self.detection_client = detection_client
        self.os_name = platform.system()
        self.logger = LogManager().get_logger()
        self.help_overlay = HelpOverlay()
        self.inference_overlay = InferenceOverlay(self._build_inference_context())
        self.detection_overlay = DetectionOverlay(self._build_detection_context())
        self._inferring = False
        self._inferring_lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self.roi_selector = RoiSelector()

        # Detection runtime state.
        # `_detect_period_s` は時間経過ベースのスロットリング周期.
        # `_detection_enabled` は i キートグルで変化し, True のとき送信.
        # `_detecting` は in-flight ガード (前リクエスト未完了なら次を送らない).
        self._detect_period_s = 1.0 / detect_fps if detect_fps > 0 else 0.0
        self._detection_enabled = self.detection_client is not None
        self._last_detect_ts: float = -float("inf")
        self._detecting = False
        self._detecting_lock = threading.Lock()
        self._detection_thread: threading.Thread | None = None

    def _build_inference_context(self) -> InferenceContext | None:
        """推論クライアントからオーバーレイ用コンテキストを構築する.

        Returns:
            コンテキスト情報, またはクライアントがない場合は None.
        """
        client = self.inference_client
        if client is None:
            return None

        image_size = None
        if client.resize is not None:
            image_size = f"{client.resize.width}x{client.resize.height}"

        return InferenceContext(
            server_url=client.base_url,
            image_size=image_size,
        )

    def _build_detection_context(self) -> DetectionContext | None:
        """検出クライアントからオーバーレイ用コンテキストを構築する.

        Returns:
            コンテキスト情報, またはクライアントがない場合は None.
        """
        client = self.detection_client
        if client is None:
            return None
        # DetectConfig はリサイズを持たないため image_size は常に None.
        return DetectionContext(server_url=client.base_url)

    @property
    def is_detect_mode(self) -> bool:
        """`detection_client` が設定されているかで detect モードかを判定する."""
        return self.detection_client is not None

    def _resize_for_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        アスペクト比を維持しつつ, preview_size に収まるようリサイズする.

        Args:
            frame: 入力フレーム.

        Returns:
            np.ndarray: リサイズされたフレーム.
        """
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            self.logger.warning(f"Invalid frame size: ({w}, {h}), skipping resize")
            return frame
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
        - 'i': classify モードでは推論実行, detect モードでは検出 ON/OFF トグル
        - 'd': ROI リセット (detect モードでは無効)
        - 'h': ヘルプオーバーレイ表示/非表示
        - 'q': 終了
        - マウスドラッグ: ROI 選択 (detect モードでは無効)

        Raises:
            VisionCaptureError: カメラ設定ダイアログが非対応のOSで呼び出された場合.
        """
        self.logger.info("Starting live preview with key controls:")
        self.logger.info(
            "Press 'c' to capture, 'r' to start recording, 't' to stop recording,"
        )
        self.logger.info(
            "'s' for camera settings, 'd' to clear ROI, 'h' for help, 'q' to quit."
        )

        try:
            cv2.namedWindow("Live View")
            # detect モードでは ROI 選択を無効化. mouse callback をワイヤしない.
            if not self.is_detect_mode:
                cv2.setMouseCallback("Live View", self.roi_selector.mouse_callback)

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

                # detect モードではスロットリングで非同期検出を起動する.
                # 結果は次以降の draw で overlay に反映される.
                if self.is_detect_mode:
                    self._maybe_detect(frame)

                # プレビュー表示 (リサイズ後にオーバーレイを描画)
                preview = self._resize_for_preview(frame)
                if not self.is_detect_mode:
                    self.roi_selector.set_preview_scale(
                        frame.shape[1], preview.shape[1]
                    )
                    self.roi_selector.draw(preview)
                    self.inference_overlay.draw(preview)
                else:
                    self.detection_overlay.draw(preview)
                self.help_overlay.draw(preview)
                cv2.imshow("Live View", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    # キャプチャ処理 (ROI でクロップ)
                    snapshot = self.roi_selector.crop(frame.copy())
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
                elif key == ord("i"):
                    # i キーはモード排他: detect → トグル, classify → 推論実行.
                    if self.is_detect_mode:
                        self._toggle_detection()
                    else:
                        self._run_inference(self.roi_selector.crop(frame))
                elif key == ord("d"):
                    # ROI リセット (detect モードでは無効)
                    if not self.is_detect_mode:
                        self.roi_selector.clear()
                        self.logger.info("ROI cleared")
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

    def _run_inference(self, frame: np.ndarray) -> None:
        """推論 API にフレームをバックグラウンドで送信する.

        推論中の二重送信を防止し, 別スレッドで実行する.

        Args:
            frame: 現在のフレーム.
        """
        if self.inference_client is None:
            self.logger.warning(
                "Inference is not available (--infer-config not loaded)"
            )
            return

        # Lock で check-then-act を保護し, 二重起動を確実に防止する.
        with self._inferring_lock:
            if self._inferring:
                self.logger.info("Inference already in progress, skipping")
                return
            self._inferring = True

        self.inference_overlay.set_inferring(True)
        snapshot = frame.copy()
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            args=(snapshot,),
            daemon=True,
        )
        self._inference_thread.start()

    def _inference_worker(self, frame: np.ndarray) -> None:
        """バックグラウンドで推論を実行するワーカー.

        Args:
            frame: 推論対象のフレーム.
        """
        try:
            client = self.inference_client
            assert client is not None  # _run_inference で None チェック済み
            resized = client.resize_frame(frame)
            result = client.predict(frame)
            self.inference_overlay.update(result)
            self.logger.info(
                f"Inference: {result.class_name} "
                f"({result.confidence * 100:.1f}%, "
                f"{result.e2e_time_ms:.1f}ms, "
                f"RTT: {result.rtt_ms:.1f}ms)"
            )
            image_file = self._save_inference_frame(resized)
            self._save_inference_csv(result, image_file)
        except InferenceConnectionError as e:
            self.inference_overlay.set_error("Connection failed")
            self.logger.error(f"Inference failed: {e}")
        except InferenceError as e:
            self.inference_overlay.set_error("Inference failed")
            self.logger.error(f"Inference failed: {e}")
        except Exception as e:
            self.inference_overlay.set_error("Unexpected error")
            self.logger.error(f"Unexpected inference error: {e}")
        finally:
            self.inference_overlay.set_inferring(False)
            with self._inferring_lock:
                self._inferring = False

    def _toggle_detection(self) -> None:
        """検出の ON / OFF をトグルする.

        OFF 時は overlay をクリアし, 次フレーム以降のリクエスト送信を停止する.
        In-flight 中のリクエストはキャンセルせず, 結果が返っても overlay に
        反映されない (OFF 側で clear 済み) ため UI 上は消える.
        """
        self._detection_enabled = not self._detection_enabled
        if not self._detection_enabled:
            self.detection_overlay.clear()
            self.detection_overlay.set_inferring(False)
            self.logger.info("Detection: OFF")
        else:
            self.logger.info("Detection: ON")

    def _maybe_detect(self, frame: np.ndarray, now: float | None = None) -> bool:
        """スロットリング + in-flight ガードを判定し, 検出ワーカーを起動する.

        以下のすべてを満たすときに限りワーカーを起動する:
        - 検出有効 (`_detection_enabled`)
        - 前回送信から `_detect_period_s` 以上経過
        - 前回リクエストが完了している (`_detecting is False`)

        スロットリング周期より API の応答が遅いケースでは, in-flight ガードで
        リクエストが重ならないよう自動的に実効レートが下がる.

        Args:
            frame: 送信候補のフレーム.
            now: 現在時刻 (テスト用). None のとき `time.perf_counter()` を使用.

        Returns:
            ワーカーを起動したら True, スキップしたら False.
        """
        if self.detection_client is None or not self._detection_enabled:
            return False

        current = now if now is not None else time.perf_counter()
        if current - self._last_detect_ts < self._detect_period_s:
            return False

        # check-then-act を lock で保護
        with self._detecting_lock:
            if self._detecting:
                return False
            self._detecting = True

        self._last_detect_ts = current
        self.detection_overlay.set_inferring(True)
        snapshot = frame.copy()
        self._detection_thread = threading.Thread(
            target=self._detection_worker,
            args=(snapshot,),
            daemon=True,
        )
        self._detection_thread.start()
        return True

    def _detection_worker(self, frame: np.ndarray) -> None:
        """バックグラウンドで検出 API を呼び出すワーカー.

        Args:
            frame: 検出対象のフレーム.
        """
        try:
            client = self.detection_client
            assert client is not None  # _maybe_detect で None チェック済み
            result = client.detect(frame)
            # OFF に切り替わった後は描画に使わないよう overlay.clear() 済みだが,
            # update で再設定されるのを避けるため enabled を確認してから反映.
            if self._detection_enabled:
                self.detection_overlay.update(result)
        except DetectionConnectionError as e:
            if self._detection_enabled:
                self.detection_overlay.set_error("Connection failed")
            self.logger.error(f"Detection failed: {e}")
        except DetectionError as e:
            if self._detection_enabled:
                self.detection_overlay.set_error("Detection failed")
            self.logger.error(f"Detection failed: {e}")
        except Exception as e:
            if self._detection_enabled:
                self.detection_overlay.set_error("Unexpected error")
            self.logger.error(f"Unexpected detection error: {e}")
        finally:
            self.detection_overlay.set_inferring(False)
            with self._detecting_lock:
                self._detecting = False

    def _save_inference_frame(self, frame: np.ndarray) -> str | None:
        """リサイズ済みフレームを画像ファイルとして保存する.

        Args:
            frame: リサイズ+パディング済みのフレーム.

        Returns:
            保存されたファイル名, または保存しなかった場合は None.
        """
        client = self.inference_client
        if client is None or not client.save_frame:
            return None

        try:
            inference_dir = Path(self.pipeline.output_dir) / "inference"
            inference_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"infer_{timestamp}.png"
            save_path = inference_dir / filename
            success = cv2.imwrite(str(save_path), frame)
            if success:
                self.logger.info(f"Inference frame saved: {save_path}")
                return filename
            else:
                self.logger.warning(f"Failed to save inference frame: {save_path}")
                return None
        except (OSError, cv2.error) as e:
            self.logger.warning(f"Error saving inference frame: {e}")
            return None

    def _save_inference_csv(
        self,
        result: PredictResponse,
        image_file: str | None,
    ) -> None:
        """推論結果を CSV ファイルに追記する.

        Args:
            result: 推論レスポンス.
            image_file: 保存された画像ファイル名 (None の場合は空文字).
        """
        client = self.inference_client
        if client is None or not client.save_csv:
            return

        try:
            inference_dir = Path(self.pipeline.output_dir) / "inference"
            writer = InferenceCsvWriter(inference_dir)
            writer.write_row(result, image_file)
            self.logger.info(f"Inference result saved to CSV: {writer.csv_path}")
        except OSError as e:
            self.logger.warning(f"Error saving inference CSV: {e}")

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

        # 推論ワーカースレッドの完了待機
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)

        # 検出ワーカースレッドの完了待機
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=5.0)

        # 推論クライアントのクリーンアップ
        if self.inference_client:
            self.inference_client.close()

        # 検出クライアントのクリーンアップ
        if self.detection_client:
            self.detection_client.close()

        # カメラ設定を結果フォルダに保存 (cap.release() の前に実行)
        self._save_camera_config()

        # カメラリソースの解放
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Camera resource and windows have been released.")

    def _save_camera_config(self) -> None:
        """カメラ設定を結果フォルダに JSON として保存する."""
        if self.camera_setup is None:
            return

        try:
            output_dir = Path(self.pipeline.output_dir)
            save_path = save_camera_config(
                cap=self.cap,
                output_dir=output_dir,
                camera_index=self.camera_setup.camera_index,
                profile_name=self.camera_setup.profile_name,
                requested_width=self.camera_setup.requested_width,
                requested_height=self.camera_setup.requested_height,
            )
            self.logger.info(f"Camera config saved: {save_path}")
        except OSError as e:
            self.logger.warning(f"Error saving camera config: {e}")
