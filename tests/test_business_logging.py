"""Operation logs must describe confirmed outcomes, not intended actions."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import PinChatMessage, SendMessage

from bot.config import Settings
from bot.db.engine import init_db
from bot.db.models import ScheduledMessage
from bot.middlewares.verification_gate import PendingVerificationGateMiddleware
from bot.services.keyword_reply import send_keyword_reply
from bot.services.scheduled_messages import ScheduledMessageService
from bot.services.verify_web import (
    _QueuedWebhookUpdate,
    _WebhookUpdateDeadlineExceeded,
    _WebhookUpdateQueue,
)
from bot.services.welcome import send_group_welcome


class BusinessLoggingTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self) -> Settings:
        settings = Settings(_env_file=None)
        settings.bot.auto_delete_categories = []
        return settings

    async def test_gate_reports_deleted_only_after_delete_success(self) -> None:
        event = SimpleNamespace(delete=AsyncMock(return_value=True), message_id=3)
        with patch(
            "bot.middlewares.verification_gate.schedule_message_auto_delete_durable",
            new=AsyncMock(),
        ) as schedule:
            result = await PendingVerificationGateMiddleware._delete_or_schedule(event, -1)
        self.assertEqual(result, "deleted")
        schedule.assert_not_awaited()

    async def test_gate_distinguishes_queued_from_rejected_cleanup(self) -> None:
        for accepted, expected in ((True, "queued"), (False, "failed")):
            with self.subTest(accepted=accepted):
                event = SimpleNamespace(
                    delete=AsyncMock(side_effect=TimeoutError("request timed out")),
                    message_id=3,
                )
                with patch(
                    "bot.middlewares.verification_gate.schedule_message_auto_delete_durable",
                    new=AsyncMock(return_value=accepted),
                ), self.assertLogs("bot.middlewares.verification_gate", "WARNING") as logs:
                    result = await PendingVerificationGateMiddleware._delete_or_schedule(event, -1)
                self.assertEqual(result, expected)
                self.assertIn("TimeoutError", "\n".join(logs.output))
                self.assertNotIn("outcome=deleted", "\n".join(logs.output))

    async def test_gate_false_delete_result_is_not_reported_as_deleted(self) -> None:
        event = SimpleNamespace(delete=AsyncMock(return_value=False), message_id=3)
        with patch(
            "bot.middlewares.verification_gate.schedule_message_auto_delete_durable",
            new=AsyncMock(return_value=False),
        ) as schedule, self.assertLogs("bot.middlewares.verification_gate", "WARNING"):
            result = await PendingVerificationGateMiddleware._delete_or_schedule(event, -1)
        self.assertEqual(result, "failed")
        # Logging changes must not add new retry behavior.
        schedule.assert_not_awaited()

    async def test_gate_fallback_exception_reports_failed(self) -> None:
        event = SimpleNamespace(delete=AsyncMock(side_effect=TimeoutError()), message_id=3)
        with patch(
            "bot.middlewares.verification_gate.schedule_message_auto_delete_durable",
            new=AsyncMock(side_effect=RuntimeError("queue unavailable")),
        ), self.assertLogs("bot.middlewares.verification_gate", "WARNING") as logs:
            result = await PendingVerificationGateMiddleware._delete_or_schedule(event, -1)
        self.assertEqual(result, "failed")
        self.assertIn("fallback_error_type=RuntimeError", "\n".join(logs.output))

    async def test_scheduled_send_log_does_not_claim_failed_pin_succeeded(self) -> None:
        entry = ScheduledMessage(
            id=1, group_id=-1, text="notice", buttons=[],
            auto_delete=False, pin_message=True, unpin_previous=False,
        )
        failure = TelegramBadRequest(
            method=PinChatMessage(chat_id=-1, message_id=3), message="not enough rights"
        )
        bot = SimpleNamespace(
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=3)),
            pin_chat_message=AsyncMock(side_effect=failure),
        )
        service = ScheduledMessageService(
            bot=bot, settings=self._settings(), session_factory=None,
        )
        with self.assertLogs("bot.services.scheduled_messages", "INFO") as logs:
            result = await service._deliver(entry)
        self.assertTrue(result)  # Message delivery is independent of its pin.
        output = "\n".join(logs.output)
        self.assertIn("pinned=False", output)
        self.assertNotIn("pinned=True", output)
        self.assertIn("error_type=TelegramBadRequest", output)
        self.assertTrue(all(not record.exc_info for record in logs.records))

    async def test_scheduled_pin_returns_false_for_explicit_api_failure(self) -> None:
        service = ScheduledMessageService(
            bot=SimpleNamespace(pin_chat_message=AsyncMock(return_value=False)),
            settings=self._settings(), session_factory=None,
        )
        entry = ScheduledMessage(id=1, group_id=-1, unpin_previous=False)
        with self.assertLogs("bot.services.scheduled_messages", "WARNING"):
            self.assertFalse(await service._pin(entry, 3))

    async def test_scheduled_pin_is_true_even_if_bookkeeping_fails(self) -> None:
        service = ScheduledMessageService(
            bot=SimpleNamespace(pin_chat_message=AsyncMock(return_value=True)),
            settings=self._settings(),
            session_factory=lambda: (_ for _ in ()).throw(RuntimeError("db unavailable")),
        )
        entry = ScheduledMessage(id=1, group_id=-1, unpin_previous=False)
        with self.assertLogs("bot.services.scheduled_messages", "ERROR"):
            self.assertTrue(await service._pin(entry, 3))

    async def test_welcome_api_failure_has_reason_without_expected_traceback(self) -> None:
        failure = TelegramBadRequest(
            method=SendMessage(chat_id=-1, text="welcome"), message="chat not found"
        )
        session = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(settings={"welcome_message": "welcome"})),
            commit=AsyncMock(),
        )
        with patch(
            "bot.services.welcome.send_template_with_fallback",
            new=AsyncMock(side_effect=failure),
        ), self.assertLogs("bot.services.welcome", "WARNING") as logs:
            result = await send_group_welcome(
                SimpleNamespace(), session, self._settings(),
                group_id=-1, user_id=2, display_name="name",
            )
        self.assertFalse(result)
        self.assertIn("error_type=TelegramBadRequest", "\n".join(logs.output))
        self.assertIn("chat not found", "\n".join(logs.output))
        self.assertFalse(logs.records[-1].exc_info)

    async def test_keyword_pin_failure_has_reason_without_expected_traceback(self) -> None:
        failure = TelegramBadRequest(
            method=PinChatMessage(chat_id=-1, message_id=3), message="not enough rights"
        )
        message = SimpleNamespace(
            message_id=1, chat=SimpleNamespace(id=-1),
            bot=SimpleNamespace(pin_chat_message=AsyncMock(side_effect=failure)),
        )
        rule = SimpleNamespace(
            id=4, group_id=-1, auto_delete=False, reply_text="reply", pin_message=True,
        )
        with patch(
            "bot.services.keyword_reply.answer_with_auto_delete",
            new=AsyncMock(return_value=SimpleNamespace(message_id=3)),
        ), self.assertLogs("bot.services.keyword_reply", "WARNING") as logs:
            self.assertTrue(await send_keyword_reply(message, rule, self._settings()))
        self.assertIn("error_type=TelegramBadRequest", "\n".join(logs.output))
        self.assertFalse(logs.records[-1].exc_info)

    async def test_engine_hides_parameters_at_construction(self) -> None:
        # Stop at engine creation: no database, production config or files touched.
        with patch(
            "bot.db.engine.create_async_engine", side_effect=RuntimeError("test sentinel")
        ) as create:
            with self.assertRaisesRegex(RuntimeError, "test sentinel"):
                await init_db("sqlite+aiosqlite:///:memory:")
        self.assertIs(create.call_args.kwargs["hide_parameters"], True)

    async def test_webhook_exception_contains_update_and_phase(self) -> None:
        dispatcher = SimpleNamespace(feed_raw_update=AsyncMock(side_effect=ValueError("bad update")))
        queue = _WebhookUpdateQueue(
            dispatcher=dispatcher, bot=SimpleNamespace(), session_factory=None,
            secret_token="test", worker_count=1,
        )
        queued = _QueuedWebhookUpdate(
            update={"update_id": 87}, update_id=87,
            enqueued_at=asyncio.get_running_loop().time(),
            result=asyncio.get_running_loop().create_future(),
        )
        with self.assertLogs("bot.services.verify_web", "ERROR") as logs:
            self.assertFalse(await queue._process_update(queued))
        output = "\n".join(logs.output)
        self.assertIn("update=87", output)
        self.assertIn("phase=dispatch", output)
        self.assertIn("error_type=ValueError", output)
        self.assertIsNotNone(logs.records[-1].exc_info)

    async def test_webhook_timeout_uses_summary_not_traceback(self) -> None:
        dispatcher = SimpleNamespace(
            feed_raw_update=AsyncMock(side_effect=_WebhookUpdateDeadlineExceeded("deadline exceeded"))
        )
        queue = _WebhookUpdateQueue(
            dispatcher=dispatcher, bot=SimpleNamespace(), session_factory=None,
            secret_token="test", worker_count=1,
        )
        queued = _QueuedWebhookUpdate(
            update={"update_id": 88}, update_id=88,
            enqueued_at=asyncio.get_running_loop().time(),
            result=asyncio.get_running_loop().create_future(),
        )
        with self.assertLogs("bot.services.verify_web", "WARNING") as logs:
            self.assertFalse(await queue._process_update(queued))
        self.assertIn("update=88", "\n".join(logs.output))
        self.assertTrue(all(record.exc_info is None for record in logs.records))
