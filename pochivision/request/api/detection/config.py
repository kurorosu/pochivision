"""検出設定ファイルのモデルとローダーを定義するモジュール."""

from dataclasses import dataclass
from typing import Any

from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.constants import (
    DEFAULT_DETECTION_FORMAT,
    DEFAULT_DETECTION_JPEG_QUALITY,
    DEFAULT_DETECTION_SCORE_THRESHOLD,
    DEFAULT_DETECTION_TIMEOUT,
)
from pochivision.exceptions.config import ConfigValidationError

_VALID_FORMATS = {"raw", "jpeg"}


@dataclass(frozen=True)
class DetectConfig:
    """検出 API の設定.

    Note:
        `InferConfig` (分類用) と異なり, リサイズ設定 (`resize`) は**意図的に持たない**.
        pochidetection は内部でモデル入力サイズへのリサイズを行い, bbox を元画像
        座標系に逆変換して返すため, クライアント側でリサイズすると bbox の座標系が
        ずれる. アスペクト比維持のための letterbox パディングもサーバー側の
        responsibility として pochidetection#445 で扱う.

    Attributes:
        base_url: pochidetection 検出 API のベース URL.
        image_format: 画像送信形式 ("raw" or "jpeg"). raw は圧縮劣化なしで検出精度に有利,
            jpeg は転送量削減向き.
        score_threshold: 検出信頼度の下限しきい値 (0.0-1.0).
        timeout: リクエストタイムアウト (秒).
        jpeg_quality: JPEG 圧縮品質 (1-100). image_format="jpeg" のとき使用.
    """

    base_url: str
    image_format: str = DEFAULT_DETECTION_FORMAT
    score_threshold: float = DEFAULT_DETECTION_SCORE_THRESHOLD
    timeout: float = DEFAULT_DETECTION_TIMEOUT
    jpeg_quality: int = DEFAULT_DETECTION_JPEG_QUALITY


def load_detect_config(path: str) -> DetectConfig:
    """検出設定ファイルを読み込む.

    `ConfigHandler.load_json()` で JSON を読み込み,
    バリデーション後に DetectConfig を構築して返す.

    Args:
        path: 設定ファイルのパス.

    Returns:
        検出設定.

    Raises:
        ConfigLoadError: ファイルが見つからない, または JSON パースに失敗した場合.
        ConfigValidationError: 設定内容が不正な場合.
    """
    data = ConfigHandler.load_json(path)
    return _build_detect_config(data)


def _build_detect_config(data: dict[str, Any]) -> DetectConfig:
    """辞書から DetectConfig を構築する.

    Args:
        data: 設定辞書.

    Returns:
        検出設定.

    Raises:
        ConfigValidationError: 設定内容が不正な場合.
    """
    if "base_url" not in data:
        raise ConfigValidationError("検出設定に 'base_url' が必要です")
    base_url = data["base_url"]
    if not isinstance(base_url, str) or not base_url.startswith(
        ("http://", "https://")
    ):
        raise ConfigValidationError(
            f"'base_url' は http:// または https:// で始まる文字列必須: {base_url!r}"
        )

    image_format = data.get("image_format", DEFAULT_DETECTION_FORMAT)
    if image_format not in _VALID_FORMATS:
        raise ConfigValidationError(
            f"'image_format' は {_VALID_FORMATS} のいずれかである必要があります: "
            f"{image_format!r}"
        )

    score_threshold = data.get("score_threshold", DEFAULT_DETECTION_SCORE_THRESHOLD)
    if (
        not isinstance(score_threshold, (int, float))
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise ConfigValidationError(
            f"'score_threshold' は 0.0-1.0 の数値である必要があります: "
            f"{score_threshold!r}"
        )

    timeout = data.get("timeout", DEFAULT_DETECTION_TIMEOUT)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigValidationError(
            f"'timeout' は正の数値である必要があります: {timeout!r}"
        )

    jpeg_quality = data.get("jpeg_quality", DEFAULT_DETECTION_JPEG_QUALITY)
    if not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ConfigValidationError(
            f"'jpeg_quality' は 1-100 の整数である必要があります: {jpeg_quality!r}"
        )

    return DetectConfig(
        base_url=base_url,
        image_format=image_format,
        score_threshold=float(score_threshold),
        timeout=float(timeout),
        jpeg_quality=jpeg_quality,
    )
