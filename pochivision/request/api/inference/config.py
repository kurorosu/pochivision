"""推論設定ファイルのモデルとローダーを定義するモジュール."""

from dataclasses import dataclass
from typing import Any

from pochivision.capturelib.config_handler import ConfigHandler
from pochivision.exceptions.config import ConfigValidationError


@dataclass(frozen=True)
class ResizeConfig:
    """推論 API 送信前のリサイズ設定.

    Attributes:
        width: リサイズ後の幅.
        height: リサイズ後の高さ.
        padding_color: パディング色 (BGR).
    """

    width: int
    height: int
    padding_color: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class InferConfig:
    """推論 API の設定.

    Attributes:
        base_url: pochitrain 推論 API のベース URL.
        image_format: 画像送信形式 ("raw" or "jpeg").
        resize: リサイズ設定 (None の場合はリサイズなし).
        save_frame: 推論実行時にフレーム画像を保存するかどうか.
        save_csv: 推論結果を CSV ファイルに出力するかどうか.
    """

    base_url: str
    image_format: str = "jpeg"
    resize: ResizeConfig | None = None
    save_frame: bool = False
    save_csv: bool = False


_VALID_FORMATS = {"raw", "jpeg"}


def load_infer_config(path: str) -> InferConfig:
    """推論設定ファイルを読み込む.

    `ConfigHandler.load_json()` で JSON を読み込み,
    バリデーション後に InferConfig を構築して返す.

    Args:
        path: 設定ファイルのパス.

    Returns:
        推論設定.

    Raises:
        ConfigLoadError: ファイルが見つからない, または JSON パースに失敗した場合.
        ConfigValidationError: 設定内容が不正な場合.
    """
    data = ConfigHandler.load_json(path)
    return _build_infer_config(data)


def _build_infer_config(data: dict[str, Any]) -> InferConfig:
    """辞書から InferConfig を構築する.

    Args:
        data: 設定辞書.

    Returns:
        推論設定.

    Raises:
        ConfigValidationError: 設定内容が不正な場合.
    """
    if "base_url" not in data:
        raise ConfigValidationError("推論設定に 'base_url' が必要です")

    image_format = data.get("image_format", "jpeg")
    if image_format not in _VALID_FORMATS:
        raise ConfigValidationError(
            f"'image_format' は {_VALID_FORMATS} のいずれかである必要があります: "
            f"{image_format!r}"
        )

    resize = _build_resize_config(data.get("resize"))

    save_frame = data.get("save_frame", False)
    if not isinstance(save_frame, bool):
        raise ConfigValidationError(
            f"'save_frame' は bool である必要があります: {save_frame!r}"
        )

    save_csv = data.get("save_csv", False)
    if not isinstance(save_csv, bool):
        raise ConfigValidationError(
            f"'save_csv' は bool である必要があります: {save_csv!r}"
        )

    return InferConfig(
        base_url=data["base_url"],
        image_format=image_format,
        resize=resize,
        save_frame=save_frame,
        save_csv=save_csv,
    )


def _build_resize_config(
    resize_data: dict[str, Any] | None,
) -> ResizeConfig | None:
    """辞書から ResizeConfig を構築する.

    Args:
        resize_data: リサイズ設定辞書 (None 可).

    Returns:
        リサイズ設定, または None.

    Raises:
        ConfigValidationError: 設定内容が不正な場合.
    """
    if resize_data is None:
        return None

    if "width" not in resize_data or "height" not in resize_data:
        raise ConfigValidationError("resize 設定には 'width' と 'height' が必要です")

    width = resize_data["width"]
    height = resize_data["height"]

    if not isinstance(width, int) or width <= 0:
        raise ConfigValidationError(
            f"resize.width は正の整数である必要があります: {width!r}"
        )
    if not isinstance(height, int) or height <= 0:
        raise ConfigValidationError(
            f"resize.height は正の整数である必要があります: {height!r}"
        )

    padding_color_raw = resize_data.get("padding_color", [0, 0, 0])
    if (
        not isinstance(padding_color_raw, list)
        or len(padding_color_raw) != 3
        or not all(isinstance(v, int) and 0 <= v <= 255 for v in padding_color_raw)
    ):
        raise ConfigValidationError(
            f"resize.padding_color は [0-255, 0-255, 0-255] の形式である必要があります: "
            f"{padding_color_raw!r}"
        )

    return ResizeConfig(
        width=width,
        height=height,
        padding_color=(
            padding_color_raw[0],
            padding_color_raw[1],
            padding_color_raw[2],
        ),
    )
