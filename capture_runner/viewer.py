import cv2

from core import PipelineExecutor
from exceptions import VisionCaptureError
from capturelib.log_manager import LogManager


class LivePreviewRunner:
    """
    カメラプレビューとキャプチャ機能を統合したコントローラークラス。

    Attributes:
        cap (cv2.VideoCapture): カメラオブジェクト
        pipeline: キャプチャ後に処理を行うパイプラインインスタンス
    """

    def __init__(self, cap: cv2.VideoCapture, pipeline: PipelineExecutor) -> None:
        """
        LivePreviewRunner を初期化する。

        Args:
            cap: 初期化済みの cv2.VideoCapture オブジェクト
            pipeline: .run(image) を持つ画像処理パイプラインインスタンス
        """
        self.cap = cap
        self.pipeline = pipeline

    def run(self) -> None:
        """
        ライブビューを開始し、'c' でキャプチャ、'q' で終了。
        """
        print("Press 'c' to capture, 'q' to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                cv2.imshow("Live View", cv2.resize(frame, (640, 480)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    snapshot = frame.copy()
                    self.pipeline.run(snapshot)
                elif key == ord('q'):
                    break
        except VisionCaptureError as e:
            logger = LogManager().get_logger()
            logger.critical(
                f"Critical error occurred in pipeline: {e}\nCamera resource will be released.")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logger = LogManager().get_logger()
            logger.info("Camera resource and windows have been released.")
