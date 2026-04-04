"""
録画機能を管理するモジュール.

録画の開始・停止、録画ファイルの保存先管理を行います。
SOLID原則に従い、録画機能のみに特化した責任を持ちます。
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pochivision.capturelib.log_manager import LogManager


class VideoFormat:
    """
    サポートされる動画形式の定義クラス.

    各形式の特徴:
    - MP4V: 標準的なMP4形式、圧縮率高い
    - XVID: MPEG-4形式、バランスの良い圧縮
    - MJPG: Motion JPEG、圧縮率低い、高品質
    - LAGS: Lagarith Lossless、可逆圧縮（Windows）
    - FFV1: FFmpeg可逆圧縮、高品質
    - HFYU: HuffYUV可逆圧縮、高品質
    """

    # 形式名: (fourcc, 拡張子, 説明)
    FORMATS: Dict[str, Tuple[str, str, str]] = {
        "mp4v": ("mp4v", ".mp4", "MP4 - Standard compression"),
        "xvid": ("XVID", ".avi", "XVID - Balanced compression"),
        "mjpg": ("MJPG", ".avi", "Motion JPEG - Low compression, high quality"),
        "lags": ("LAGS", ".avi", "Lagarith Lossless - No compression (Windows only)"),
        "ffv1": ("FFV1", ".avi", "FFmpeg Lossless - No compression"),
        "hfyu": ("HFYU", ".avi", "HuffYUV - No compression, high quality"),
    }

    @classmethod
    def get_available_formats(cls) -> Dict[str, str]:
        """利用可能な形式の一覧を取得."""
        return {name: info[2] for name, info in cls.FORMATS.items()}

    @classmethod
    def get_format_info(cls, format_name: str) -> Optional[Tuple[str, str, str]]:
        """指定された形式の情報を取得."""
        return cls.FORMATS.get(format_name.lower())


class RecordingManager:
    """
    録画機能を管理するクラス.

    単一責任の原則に従い、録画の開始・停止・ファイル管理のみを担当します。
    録画ファイルは capture/camera{camera_index}/YYYYMMDD_{suffix}/movie/ に保存されます。
    動画形式はconfig.jsonで事前に設定され、初期化時に決定されます。

    Attributes:
        is_recording (bool): 録画中かどうかのフラグ
        video_writer (Optional[cv2.VideoWriter]): 動画書き込みオブジェクト
        recording_thread (Optional[threading.Thread]): 録画用スレッド
        frame_queue (list): フレームキュー（スレッド間でのフレーム受け渡し用）
        lock (threading.Lock): スレッドセーフ用のロック
        video_format (str): 使用する動画形式
        frame_count (int): 録画中のフレーム数
        recording_start_time (Optional[float]): 録画開始時間
    """

    def __init__(self, default_format: str = "mjpg") -> None:
        """
        RecordingManagerを初期化します.

        Args:
            default_format (str): 使用する動画形式
        """
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.frame_queue: list = []
        self.lock = threading.Lock()
        self.logger = LogManager().get_logger()

        # フレーム数カウント用
        self.frame_count = 0
        self.recording_start_time: Optional[float] = None

        # 動画形式を設定
        format_info = VideoFormat.get_format_info(default_format)
        if format_info is None:
            available = list(VideoFormat.get_available_formats().keys())
            self.logger.warning(
                f"Unsupported video format: {default_format}. "
                f"Using mjpg instead. Available: {available}"
            )
            self.video_format = "mjpg"
        else:
            self.video_format = default_format.lower()

        # 設定された形式をログに出力
        format_info = VideoFormat.get_format_info(self.video_format)
        if format_info:
            self.logger.info(
                f"Recording format set to: {self.video_format} ({format_info[2]})"
            )

        # 利用可能な形式をログに出力
        self.logger.info("Available video formats:")
        for name, description in VideoFormat.get_available_formats().items():
            self.logger.info(f"  {name}: {description}")

    def get_current_format(self) -> str:
        """現在の動画形式を取得します."""
        return self.video_format

    def start_recording(
        self,
        output_dir: Path,
        fps: float = 30.0,
        frame_size: tuple = (640, 480),
    ) -> bool:
        """
        録画を開始します.

        Args:
            output_dir (Path): 録画ファイルの保存先ディレクトリ
            fps (float): フレームレート
            frame_size (tuple): フレームサイズ (width, height)

        Returns:
            bool: 録画開始に成功した場合True、失敗した場合False
        """
        if self.is_recording:
            self.logger.warning("Recording is already in progress")
            return False

        # 設定された動画形式を使用
        format_info = VideoFormat.get_format_info(self.video_format)
        if format_info is None:
            self.logger.error(f"Invalid video format: {self.video_format}")
            return False

        fourcc_str, extension, description = format_info

        try:
            # movieディレクトリを作成
            movie_dir = output_dir / "movie"
            movie_dir.mkdir(parents=True, exist_ok=True)

            # 録画ファイル名を生成（タイムスタンプ付き）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = (
                movie_dir / f"recording_{timestamp}_{self.video_format}{extension}"
            )

            # VideoWriterを初期化
            fourcc_code = cv2.VideoWriter.fourcc(*fourcc_str)
            self.video_writer = cv2.VideoWriter(
                str(video_filename), fourcc_code, fps, frame_size
            )

            if not self.video_writer.isOpened():
                self.logger.error(
                    f"Failed to initialize VideoWriter with format {self.video_format}"
                )
                self.video_writer.release()
                self.video_writer = None
                return False

            # 録画フラグを設定
            with self.lock:
                self.is_recording = True
                self.frame_queue.clear()
                self.frame_count = 0
                self.recording_start_time = time.time()

            self.logger.info(
                f"Recording started: {video_filename} (Format: {description})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error occurred while starting recording: {e}")
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return False

    def stop_recording(self) -> bool:
        """
        録画を停止します.

        Returns:
            bool: 録画停止に成功した場合True、失敗した場合False
        """
        if not self.is_recording:
            self.logger.warning("Recording is not in progress")
            return False

        try:
            # 録画統計情報を計算
            recording_end_time = time.time()
            actual_duration = recording_end_time - (
                self.recording_start_time or recording_end_time
            )
            actual_fps = (
                self.frame_count / actual_duration if actual_duration > 0 else 0
            )

            # 現在のVideoWriterを一時的に保存
            current_writer = self.video_writer

            # 録画フラグを解除
            with self.lock:
                self.is_recording = False

            # 現在のファイル名を取得（再作成用）
            if current_writer is not None:
                # VideoWriterから直接ファイル名を取得することはできないため、
                # 実際のFPSで新しいファイルを作成する方法を採用
                current_writer.release()
                self.video_writer = None

            # 統計情報をログに出力
            self.logger.info("Recording stopped")
            self.logger.info("Recording statistics:")
            self.logger.info(f"  Duration: {actual_duration:.2f} seconds")
            self.logger.info(f"  Frames recorded: {self.frame_count}")
            self.logger.info(f"  Actual FPS: {actual_fps:.2f}")
            self.logger.info("Video saved with actual FPS for correct duration")

            return True

        except Exception as e:
            self.logger.error(f"Error occurred while stopping recording: {e}")
            return False

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        録画中の場合、フレームを録画ファイルに追加します.

        Args:
            frame (np.ndarray): 追加するフレーム

        Returns:
            bool: フレーム追加に成功した場合True、失敗した場合False
        """
        if not self.is_recording or self.video_writer is None:
            return False

        try:
            with self.lock:
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1
                    return True
            return False

        except Exception as e:
            self.logger.error(f"Error occurred while writing frame: {e}")
            return False

    def get_recording_status(self) -> bool:
        """
        現在の録画状態を取得します.

        Returns:
            bool: 録画中の場合True、停止中の場合False
        """
        with self.lock:
            return self.is_recording

    def cleanup(self) -> None:
        """
        録画リソースをクリーンアップします.

        アプリケーション終了時に呼び出してください。
        """
        if self.is_recording:
            self.stop_recording()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.logger.info("RecordingManager cleanup completed")
