"""ファイル命名規則を管理するユーティリティモジュール."""

from datetime import datetime
from typing import Dict, Tuple


class FileNamingManager:
    """
    画像ファイルの命名規則を管理するクラス.

    Attributes:
        image_counters (Dict[Tuple[int, str], int]): カメラとプレフィックスごとの画像カウンタ
        id_intervals (Dict[Tuple[int, str], int]): カメラとプレフィックスごとのID増加間隔
        labels (Dict[int, str]): カメラごとのラベル
    """

    def __init__(self) -> None:
        """FileNamingManagerのコンストラクタ."""
        self.image_counters: Dict[Tuple[int, str], int] = {}
        self.id_intervals: Dict[Tuple[int, str], int] = {}
        self.labels: Dict[int, str] = {}  # カメラごとのラベル

    def set_id_interval(
        self, prefix: str, camera_index: int, interval: int = 1
    ) -> None:
        """
        IDの増加間隔を設定します.

        Args:
            prefix (str): ファイル名のプレフィックス（'original'や'pipeline'など）
            camera_index (int): カメラのインデックス
            interval (int): IDが増加する画像数の間隔（デフォルトは1、画像ごとにIDが増加）
        """
        key = (camera_index, prefix)
        self.id_intervals[key] = max(1, interval)  # 最小値は1を保証

    def set_label(self, camera_index: int, label: str) -> None:
        """
        カメラごとのラベル（ファイル名のNA_NA部分）を設定します.

        Args:
            camera_index (int): カメラのインデックス
            label (str): ファイル名に挿入するラベル
        """
        self.labels[camera_index] = label

    def get_filename(
        self, prefix: str, camera_index: int, extension: str = "bmp"
    ) -> Tuple[str, int, int]:
        """
        指定されたプレフィックスと拡張子で新しいファイル名を生成します.

        命名規則：
        {prefix}_yyyymmddhhmmss_NA_NA_id{id_index}_image{image_index}.{extension}

        Args:
            prefix (str): ファイル名のプレフィックス（'original'や'pipeline'など）
            camera_index (int): カメラのインデックス
            extension (str): ファイルの拡張子（デフォルトは'bmp'）

        Returns:
            Tuple[str, int, int]: 生成されたファイル名、ID値、画像カウンタ値のタプル
        """
        # カメラとプレフィックスのセットをキーとして使用
        key = (camera_index, prefix)

        # カメラとプレフィックスごとのカウンターの初期化（存在しない場合）
        if key not in self.image_counters:
            self.image_counters[key] = 1  # 1から始まる

        # ID間隔の初期化（設定されていない場合）
        if key not in self.id_intervals:
            self.id_intervals[key] = 1  # デフォルトは1（画像ごとにIDが増加）

        # 現在の画像カウンタ値を取得
        image_index = self.image_counters[key]

        # ID値を計算（画像カウンタと間隔に基づく）
        interval = self.id_intervals[key]
        id_index = ((image_index - 1) // interval) + 1

        # 画像カウンタをインクリメント
        self.image_counters[key] += 1

        # 現在の日時を取得（yyyymmddhhmmss形式）
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # ラベルを取得（未設定の場合はNA_NA）
        label = self.labels.get(camera_index, "NA_NA")

        # ファイル名を生成
        filename = (
            f"{prefix}_{timestamp}_{label}_id{id_index}_image{image_index}.{extension}"
        )

        return filename, id_index, image_index


# シングルトンインスタンス
_file_naming_manager = FileNamingManager()


def get_file_naming_manager() -> FileNamingManager:
    """
    FileNamingManagerのシングルトンインスタンスを取得します.

    Returns:
        FileNamingManager: シングルトンインスタンス
    """
    return _file_naming_manager
