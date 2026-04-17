"""`_build_detection_client` (--detect フラグ制御) のテスト."""

import json
import logging
from pathlib import Path

from pochivision.cli.commands.run import _build_detection_client
from pochivision.request.api.detection.client import DetectionClient


def _write_valid_config(tmp_path: Path, detect_fps: float = 5.0) -> Path:
    """有効な detect_config.json を書き出す."""
    path = tmp_path / "detect_config.json"
    path.write_text(
        json.dumps(
            {
                "base_url": "http://localhost:8000",
                "detect_fps": detect_fps,
            }
        ),
        encoding="utf-8",
    )
    return path


class TestBuildDetectionClient:
    """`--detect` フラグに応じた DetectionClient 構築の振る舞い."""

    def test_detect_flag_off_returns_none(self, tmp_path):
        """`--detect` 未指定なら config の有無にかかわらず DetectionClient を作らない."""
        path = _write_valid_config(tmp_path)
        logger = logging.getLogger("test")

        client, fps = _build_detection_client(str(path), False, logger)

        assert client is None
        # fps はデフォルト値 (config を読まないため)
        assert fps == 5.0

    def test_detect_flag_on_builds_client(self, tmp_path):
        """`--detect` 指定 + 有効 config で DetectionClient が構築される."""
        path = _write_valid_config(tmp_path, detect_fps=10.0)
        logger = logging.getLogger("test")

        client, fps = _build_detection_client(str(path), True, logger)

        try:
            assert isinstance(client, DetectionClient)
            assert client.base_url == "http://localhost:8000"
            assert fps == 10.0
        finally:
            if client is not None:
                client.close()

    def test_detect_flag_on_missing_config_warns_and_falls_back(self, tmp_path, caplog):
        """`--detect` 指定されているが config が無い場合は warning + classify fallback."""
        missing_path = tmp_path / "nonexistent.json"
        logger = logging.getLogger("test")

        with caplog.at_level("WARNING"):
            client, fps = _build_detection_client(str(missing_path), True, logger)

        assert client is None
        assert fps == 5.0  # デフォルト
        assert any(
            "--detect" in rec.message and "missing" in rec.message
            for rec in caplog.records
        )

    def test_detect_flag_on_invalid_config_warns_and_falls_back(self, tmp_path, caplog):
        """`--detect` 指定されているが config が不正なら warning + fallback."""
        path = tmp_path / "detect_config.json"
        path.write_text(json.dumps({"base_url": "not-a-url"}), encoding="utf-8")
        logger = logging.getLogger("test")

        with caplog.at_level("WARNING"):
            client, fps = _build_detection_client(str(path), True, logger)

        assert client is None
        assert fps == 5.0

    def test_legacy_mode_key_does_not_affect_behavior(self, tmp_path):
        """廃止された `mode` フィールドが JSON に残っていても `--detect` 無効なら構築しない."""
        path = tmp_path / "detect_config.json"
        path.write_text(
            json.dumps(
                {
                    "base_url": "http://localhost:8000",
                    "mode": "detect",  # legacy key, should be ignored
                }
            ),
            encoding="utf-8",
        )
        logger = logging.getLogger("test")

        # --detect 未指定: mode="detect" が残っていても classify のまま
        client, _ = _build_detection_client(str(path), False, logger)
        assert client is None
