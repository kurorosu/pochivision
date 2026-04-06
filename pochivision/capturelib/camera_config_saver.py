"""セッション終了時にカメラ設定を JSON として保存するモジュール."""

import json
from pathlib import Path

import cv2

_CAMERA_PROPERTIES: dict[str, int] = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gamma": cv2.CAP_PROP_GAMMA,
    "focus": cv2.CAP_PROP_FOCUS,
    "auto_focus": cv2.CAP_PROP_AUTOFOCUS,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
}

_CONFIG_FILENAME = "camera_config.json"


def save_camera_config(
    cap: cv2.VideoCapture,
    output_dir: Path,
    camera_index: int,
    profile_name: str,
    requested_width: int,
    requested_height: int,
) -> Path:
    """カメラの現在の設定を JSON ファイルに保存する.

    Args:
        cap: カメラキャプチャオブジェクト.
        output_dir: 保存先ディレクトリ.
        camera_index: カメラインデックス.
        profile_name: プロファイル名.
        requested_width: 設定ファイルで要求した幅.
        requested_height: 設定ファイルで要求した高さ.

    Returns:
        保存先のファイルパス.
    """
    config: dict[str, object] = {
        "camera_index": camera_index,
        "profile_name": profile_name,
        "backend": cap.getBackendName(),
        "requested_width": requested_width,
        "requested_height": requested_height,
        "actual_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "actual_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 1),
    }

    for name, prop_id in _CAMERA_PROPERTIES.items():
        value = cap.get(prop_id)
        if value != 0.0 or name in ("brightness", "exposure"):
            config[name] = value

    save_path = output_dir / _CONFIG_FILENAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return save_path
