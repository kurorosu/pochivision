"""ファイル名からクラス名を抽出するユーティリティ."""

from pochivision.capturelib.log_manager import LogManager


def extract_class_from_filename(
    filename: str,
    delimiter: str = "_",
    position: int = 0,
) -> str:
    """ファイル名からクラス名を抽出する.

    Args:
        filename: ファイル名 (拡張子なし).
        delimiter: 区切り文字.
        position: クラス名の位置 (負のインデックスも可).

    Returns:
        抽出されたクラス名. 抽出できない場合は空文字列.
    """
    logger = LogManager().get_logger()

    try:
        parts = filename.split(delimiter)

        if 0 <= position < len(parts):
            return str(parts[position])
        elif position < 0 and abs(position) <= len(parts):
            return str(parts[position])
        else:
            logger.warning(
                f"ファイル名 '{filename}' の位置 {position} にクラス名が見つかりません"
            )
            return ""
    except Exception as e:
        logger.warning(f"ファイル名 '{filename}' からクラス名の抽出に失敗しました: {e}")
        return ""
