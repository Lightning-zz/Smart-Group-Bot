from __future__ import annotations

import io
import logging
import os
import queue
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bot.utils.logging_setup import (
    _BoundedQueueListener,
    _DEFAULT_FORMAT,
    _DroppingQueueHandler,
    _FlushBarrier,
    _PriorityLogQueue,
    _SecureRotatingFileHandler,
    ChineseLogFormatter,
    LogContextFilter,
    attach_exception_log_context,
    configure_logging,
    flush_logging,
    logging_resource_health_snapshot,
    reset_log_context,
    set_log_context,
    sanitize_log_text,
    shutdown_logging,
)


def _record(level: int, message: str = "event") -> logging.LogRecord:
    return logging.LogRecord("bot.test", level, __file__, 1, message, (), None)


class LogFormattingTests(unittest.TestCase):
    def test_redacts_credentials_and_neutralizes_control_characters(self) -> None:
        samples = [
            'api_key="provider-secret"',
            "password='a secret with spaces'",
            "Authorization: Bearer bearer-secret",
            "https://service.test/x?key=query-secret&view=plain",
            "https://name:user-secret@service.test/",
            "https://api.telegram.org/bot123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/getMe",
            "sk-abcdefghijklmnopqrstuvwxyz0123456789",
            '{"refresh_token": "refresh-secret"}',
            "CONFIG_MASTER_KEY=master-secret",
            "TURNSTILE_SECRET=turnstile-secret",
        ]
        secrets = [
            "provider-secret", "a secret with spaces", "bearer-secret", "query-secret",
            "user-secret", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "sk-abcdefghijklmnopqrstuvwxyz0123456789", "refresh-secret", "master-secret",
            "turnstile-secret",
        ]
        for source, secret in zip(samples, secrets):
            with self.subTest(source=source):
                safe = sanitize_log_text(source)
                self.assertNotIn(secret, safe)
                self.assertIn("REDACTED", safe)
                self.assertEqual(sanitize_log_text(safe), safe)
        safe = sanitize_log_text("name\r\nFORGED\x1b[31m\x00\u202e")
        self.assertNotIn("\n", safe)
        self.assertNotIn("\r", safe)
        self.assertNotIn("\x1b", safe)
        self.assertIn(r"\n", safe)
        self.assertIn(r"\u001b", safe)
        # A batch's tuple key is correlation metadata, not a credential.
        self.assertEqual(sanitize_log_text("key=(-1001, 42)"), "key=(-1001, 42)")

    def test_formatter_includes_all_ids_and_explicit_shanghai_offset(self) -> None:
        record = _record(logging.INFO)
        record.created = 0.125
        tokens = set_log_context(
            req_id="req123", flow_id="f123", update_id=901, chat_id=-1001, user_id=42, message_id=7
        )
        try:
            LogContextFilter().filter(record)
        finally:
            reset_log_context(tokens)
        rendered = ChineseLogFormatter(_DEFAULT_FORMAT, use_color=False).format(record)
        self.assertIn("1970-01-01 08:00:00.125+08:00", rendered)
        for expected in ("流=f123", "请求=req123", "更新=901", "群=-1001", "用户=42", "消息=7", "任务=MainThread"):
            self.assertIn(expected, rendered)

    def test_queue_preparation_redacts_traceback_but_preserves_stack_lines(self) -> None:
        try:
            raise ValueError("api_key=trace-secret\x1b[31m")
        except ValueError:
            record = _record(logging.ERROR, "operation failed")
            record.exc_info = sys.exc_info()
        prepared = _DroppingQueueHandler(queue.Queue()).prepare(record)
        self.assertIsNone(prepared.exc_info)
        self.assertIn("Traceback", prepared.exc_text)
        self.assertIn("\n", prepared.exc_text)
        rendered = ChineseLogFormatter(_DEFAULT_FORMAT, use_color=False).format(prepared)
        self.assertNotIn("trace-secret", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("ValueError", rendered)

    def test_malformed_format_arguments_cannot_bypass_redaction_to_stderr(self) -> None:
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        stderr = io.StringIO()
        try:
            with patch("sys.stderr", stderr):
                configure_logging(force=True, config=SimpleNamespace(to_file=False))
                logging.getLogger("malformed-format").info(
                    "bad count=%d api_key=%s", "not-a-number", "TEST_SECRET_DO_NOT_LOG"
                )
                self.assertTrue(flush_logging(timeout=2.0))
            self.assertNotIn("TEST_SECRET_DO_NOT_LOG", stderr.getvalue())
            self.assertNotIn("not-a-number", stderr.getvalue())
            snapshot = logging_resource_health_snapshot()
            self.assertGreaterEqual(snapshot["format_errors"], 1)
        finally:
            shutdown_logging(timeout=2.0)
            root.handlers = old_handlers
            root.setLevel(old_level)

    def test_exception_context_survives_scope_reset_and_wrapping(self) -> None:
        tokens = set_log_context(update_id=901, chat_id=-1001, user_id=42, message_id=7)
        try:
            original = RuntimeError("failed")
            attach_exception_log_context(original)
        finally:
            reset_log_context(tokens)
        wrapped = RuntimeError("boundary failure")
        wrapped.__cause__ = original
        record = _record(logging.ERROR)
        record.exc_info = (RuntimeError, wrapped, None)
        LogContextFilter().filter(record)
        self.assertEqual(record.update_id, "901")
        self.assertEqual(record.message_id, "7")


class LogQueuePriorityTests(unittest.TestCase):
    def test_reserve_and_replacement_keep_high_severity_without_blocking(self) -> None:
        log_queue = _PriorityLogQueue(maxsize=4)
        handler = _DroppingQueueHandler(log_queue)
        for _ in range(4):
            handler.handle(_record(logging.INFO))
        self.assertEqual(log_queue.qsize(), 3)
        handler.handle(_record(logging.WARNING))
        handler.handle(_record(logging.ERROR))
        handler.handle(_record(logging.CRITICAL))
        levels = [log_queue.get_nowait().levelno for _ in range(log_queue.qsize())]
        self.assertEqual(levels, [logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
        self.assertEqual(handler.dropped_records, 3)

    def test_when_all_records_are_critical_drops_are_bounded_and_counted(self) -> None:
        log_queue = _PriorityLogQueue(maxsize=2)
        handler = _DroppingQueueHandler(log_queue)
        for _ in range(3):
            handler.handle(_record(logging.CRITICAL))
        self.assertEqual(log_queue.qsize(), 2)
        self.assertEqual(handler.drop_snapshot(), {"by_level": {"CRITICAL": 1}, "unreported": 1})

    def test_replacement_chooses_lowest_level_not_oldest_warning(self) -> None:
        log_queue = _PriorityLogQueue(maxsize=4)
        handler = _DroppingQueueHandler(log_queue)
        for level in (logging.WARNING, logging.INFO, logging.INFO, logging.WARNING):
            handler.handle(_record(level))
        handler.handle(_record(logging.ERROR))
        levels = [log_queue.get_nowait().levelno for _ in range(log_queue.qsize())]
        self.assertEqual(levels, [logging.WARNING, logging.INFO, logging.WARNING, logging.ERROR])
        self.assertEqual(handler.drop_snapshot()["by_level"], {"INFO": 1})

    def test_overflow_never_discards_flush_barriers(self) -> None:
        import threading

        log_queue = _PriorityLogQueue(maxsize=2)
        barrier = _FlushBarrier(threading.Event())
        log_queue.put_nowait(barrier)
        log_queue.put_nowait(None)
        handler = _DroppingQueueHandler(log_queue)
        handler.handle(_record(logging.CRITICAL))
        self.assertIs(log_queue.get_nowait(), barrier)
        self.assertIsNone(log_queue.get_nowait())

    def test_listener_emits_one_recovery_summary_without_recursive_enqueue(self) -> None:
        log_queue = _PriorityLogQueue(maxsize=4)
        handler = _DroppingQueueHandler(log_queue)
        for _ in range(6):
            handler.handle(_record(logging.INFO))
        stream = io.StringIO()
        sink = logging.StreamHandler(stream)
        sink.setFormatter(ChineseLogFormatter(_DEFAULT_FORMAT, use_color=False))
        listener = _BoundedQueueListener(log_queue, sink)
        listener.drop_handler = handler
        while not log_queue.empty():
            listener.handle(log_queue.get_nowait())
        self.assertEqual(stream.getvalue().count("日志队列已恢复"), 1)
        self.assertIn("丢弃=3", stream.getvalue())
        self.assertEqual(handler.drop_snapshot()["unreported"], 0)
        self.assertTrue(log_queue.empty())


class LoggingFilePermissionTests(unittest.TestCase):
    def test_default_color_auto_does_not_write_ansi_to_non_tty(self) -> None:
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        stream = io.StringIO()
        try:
            # Missing color uses the same default as newly-created runtime
            # settings. File output is deliberately disabled in this fixture.
            with patch("sys.stdout", stream):
                configure_logging(force=True, config=SimpleNamespace(to_file=False))
                logging.getLogger("color-test").warning("plain warning")
                self.assertTrue(flush_logging(timeout=2.0))
            self.assertIn("plain warning", stream.getvalue())
            self.assertNotIn("\x1b", stream.getvalue())
        finally:
            shutdown_logging(timeout=2.0)
            root.handlers = old_handlers
            root.setLevel(old_level)

    def test_default_file_log_is_written_under_persistent_data_directory(self) -> None:
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        old_cwd = Path.cwd()
        old_log_to_file = os.environ.get("LOG_TO_FILE")
        old_log_file_path = os.environ.get("LOG_FILE_PATH")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                os.environ["LOG_TO_FILE"] = "true"
                os.environ.pop("LOG_FILE_PATH", None)
                configure_logging(force=True)
                logging.getLogger("default-log-path-test").info("persist me")
                self.assertTrue(flush_logging(timeout=2.0))
                snapshot = logging_resource_health_snapshot()
                self.assertTrue(snapshot["listener_alive"])
                self.assertEqual(snapshot["dropped_records"], 0)
                self.assertTrue((Path(tmpdir) / "data" / "bot.log").is_file())
            finally:
                shutdown_logging(timeout=2.0)
                root.handlers = old_handlers
                root.setLevel(old_level)
                os.chdir(old_cwd)
                if old_log_to_file is None:
                    os.environ.pop("LOG_TO_FILE", None)
                else:
                    os.environ["LOG_TO_FILE"] = old_log_to_file
                if old_log_file_path is None:
                    os.environ.pop("LOG_FILE_PATH", None)
                else:
                    os.environ["LOG_FILE_PATH"] = old_log_file_path

    def test_context_is_captured_before_record_crosses_listener_thread(self) -> None:
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        old_log_to_file = os.environ.get("LOG_TO_FILE")
        old_log_file_path = os.environ.get("LOG_FILE_PATH")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.environ["LOG_TO_FILE"] = "true"
                os.environ["LOG_FILE_PATH"] = str(Path(tmpdir) / "context.log")
                configure_logging(force=True)
                tokens = set_log_context(flow_id="flow-123")
                try:
                    logging.getLogger("context-capture-test").info("captured")
                finally:
                    reset_log_context(tokens)
                self.assertTrue(flush_logging(timeout=2.0))
                text = (Path(tmpdir) / "context.log").read_text(encoding="utf-8")
                self.assertIn("流=flow-123", text)
            finally:
                shutdown_logging(timeout=2.0)
                root.handlers = old_handlers
                root.setLevel(old_level)
                for key, value in (("LOG_TO_FILE", old_log_to_file), ("LOG_FILE_PATH", old_log_file_path)):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_log_file_and_rollover_remain_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bot.log"
            handler = _SecureRotatingFileHandler(
                path,
                maxBytes=1,
                backupCount=1,
                encoding="utf-8",
            )
            logger = logging.getLogger(f"test-secure-log-{id(self)}")
            logger.handlers = [handler]
            logger.propagate = False
            logger.setLevel(logging.INFO)
            try:
                logger.info("first")
                handler.doRollover()
                logger.info("second")
            finally:
                handler.close()
                logger.handlers.clear()

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            backup = Path(f"{path}.1")
            self.assertTrue(backup.exists())
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
