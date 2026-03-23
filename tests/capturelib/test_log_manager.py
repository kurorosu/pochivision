"""LogManager のテスト."""

import logging

import pytest

from pochivision.capturelib.log_manager import LogManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """各テスト前に LogManager のシングルトンをリセットする."""
    LogManager._instance = None
    yield
    LogManager._instance = None


class TestLogManagerSingleton:
    """シングルトンパターンのテスト."""

    def test_same_instance(self):
        """同一インスタンスが返される."""
        lm1 = LogManager()
        lm2 = LogManager()
        assert lm1 is lm2

    def test_initialized_once(self):
        """初期化は一度だけ実行される."""
        lm = LogManager()
        assert lm._initialized is True


class TestLogManagerLogger:
    """get_logger のテスト."""

    def test_get_logger_returns_logger(self):
        """logging.Logger インスタンスが返される."""
        lm = LogManager()
        logger = lm.get_logger()
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        """ロガー名が 'vcc' である."""
        lm = LogManager()
        logger = lm.get_logger()
        assert logger.name == "vcc"

    def test_logger_level_debug(self):
        """ロガーのレベルが DEBUG に設定されている."""
        lm = LogManager()
        logger = lm.get_logger()
        assert logger.level == logging.DEBUG

    def test_console_handler_exists(self):
        """コンソールハンドラが追加されている."""
        lm = LogManager()
        logger = lm.get_logger()
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) >= 1


class TestLogManagerFileLogging:
    """setup_file_logging のテスト."""

    def test_file_logging_creates_handler(self, tmp_path):
        """ファイルハンドラが追加される."""
        lm = LogManager()
        log_file = tmp_path / "test.log"
        lm.setup_file_logging(log_file)

        assert lm._file_handler is not None
        assert isinstance(lm._file_handler, logging.FileHandler)

    def test_file_logging_writes_log(self, tmp_path):
        """ログファイルにメッセージが書き込まれる."""
        lm = LogManager()
        log_file = tmp_path / "test.log"
        lm.setup_file_logging(log_file)

        logger = lm.get_logger()
        logger.info("test message")

        # ハンドラをフラッシュ
        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "test message" in content

    def test_file_logging_replaces_handler(self, tmp_path):
        """2回呼ぶと古いハンドラが置き換わる."""
        lm = LogManager()
        log1 = tmp_path / "log1.log"
        log2 = tmp_path / "log2.log"

        lm.setup_file_logging(log1)
        first_handler = lm._file_handler

        lm.setup_file_logging(log2)
        second_handler = lm._file_handler

        assert first_handler is not second_handler
        assert second_handler in lm.get_logger().handlers
        assert first_handler not in lm.get_logger().handlers


class TestLogManagerSystemInfo:
    """log_system_info のテスト."""

    def test_log_system_info_no_error(self, tmp_path):
        """log_system_info がエラーなく実行できる."""
        lm = LogManager()
        log_file = tmp_path / "test.log"
        lm.setup_file_logging(log_file)

        lm.log_system_info()

        for handler in lm.get_logger().handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "System:" in content
        assert "Python:" in content
        assert "OpenCV:" in content
