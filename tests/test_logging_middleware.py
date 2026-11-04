from __future__ import annotations

import asyncio
import unittest

from aiogram.types import Update

from bot.middlewares.logging_mw import LoggingMiddleware
from bot.services.update_completion import (
    UpdateCompletionReceipt,
    bind_update_completion,
    reset_update_completion,
)
from bot.utils.logging_setup import LogContextFilter, get_log_context


_USER = {"id": 42, "is_bot": False, "first_name": "private-name"}
_CHAT = {"id": -1001, "type": "supergroup", "title": "private-title"}
_MESSAGE = {
    "message_id": 7,
    "date": 1,
    "chat": _CHAT,
    "from": _USER,
    "text": "private-message api_key=never-log-me",
}


class LoggingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_update_types_receive_scoped_context_without_content_logs(self) -> None:
        cases = {
            "message": (_MESSAGE, "-1001", "42", "7"),
            "edited_message": (_MESSAGE, "-1001", "42", "7"),
            "callback_query": ({
                "id": "callback", "from": _USER, "chat_instance": "instance",
                "message": {**_MESSAGE, "from": {**_USER, "id": 99}},
                "data": "private-callback",
            }, "-1001", "42", "7"),
            "inline_query": ({
                "id": "inline", "from": _USER, "query": "private-query", "offset": "",
            }, "-", "42", "-"),
            "poll_answer": ({
                "poll_id": "poll", "user": _USER, "option_ids": [1],
                "option_persistent_ids": [],
            }, "-", "42", "-"),
            "chat_member": ({
                "chat": _CHAT, "from": _USER, "date": 1,
                "old_chat_member": {"status": "member", "user": _USER},
                "new_chat_member": {"status": "left", "user": _USER},
            }, "-1001", "42", "-"),
        }
        middleware = LoggingMiddleware()
        for update_type, (payload, chat, user, message) in cases.items():
            with self.subTest(update_type=update_type):
                update = Update.model_validate({"update_id": 901, update_type: payload})

                async def handler(event: Update, data: dict) -> dict:
                    self.assertIs(event, update)
                    return get_log_context()

                with self.assertLogs("bot.middlewares.logging_mw", level="INFO") as captured:
                    context = await middleware(handler, update, {})
                self.assertEqual(context["update_id"], "901")
                self.assertEqual(context["chat_id"], chat)
                self.assertEqual(context["user_id"], user)
                self.assertEqual(context["message_id"], message)
                self.assertNotEqual(context["flow_id"], "-")
                rendered = "\n".join(captured.output)
                self.assertIn(update_type, rendered)
                self.assertIn("分发结束", rendered)
                self.assertNotIn("成功", rendered)
                self.assertNotIn("private-", rendered)
                self.assertNotIn("never-log-me", rendered)
                self.assertTrue(all(value == "-" for value in get_log_context().values()))

    async def test_deferred_work_is_not_logged_as_completed(self) -> None:
        receipt = UpdateCompletionReceipt()
        token = bind_update_completion(receipt)
        update = Update.model_validate({"update_id": 901, "message": _MESSAGE})

        async def handler(_event: Update, _data: dict) -> str:
            receipt.defer()
            return "queued"

        try:
            with self.assertLogs("bot.middlewares.logging_mw", level="INFO") as captured:
                result = await LoggingMiddleware()(handler, update, {})
            self.assertEqual(result, "queued")
            self.assertIn("已转后台/等待终态", "\n".join(captured.output))
            self.assertNotIn("完成", "\n".join(captured.output))
        finally:
            receipt.finish(True)
            receipt.seal()
            reset_update_completion(token)

    async def test_failure_keeps_exception_context_without_duplicate_traceback(self) -> None:
        update = Update.model_validate({"update_id": 901, "edited_message": _MESSAGE})
        error = RuntimeError("operation failed")

        async def handler(_event: Update, _data: dict) -> None:
            raise error

        with self.assertLogs("bot.middlewares.logging_mw", level="INFO") as captured:
            with self.assertRaises(RuntimeError) as caught:
                await LoggingMiddleware()(handler, update, {})
        self.assertIs(caught.exception, error)
        self.assertTrue(all(record.exc_info is None for record in captured.records))
        self.assertEqual(error._bot_log_context["update_id"], "901")
        self.assertTrue(all(value == "-" for value in get_log_context().values()))

    async def test_cancelled_dispatch_restores_context(self) -> None:
        update = Update.model_validate({"update_id": 901, "message": _MESSAGE})

        async def handler(_event: Update, _data: dict) -> None:
            raise asyncio.CancelledError

        with self.assertLogs("bot.middlewares.logging_mw", level="INFO") as captured:
            with self.assertRaises(asyncio.CancelledError):
                await LoggingMiddleware()(handler, update, {})
        self.assertIn("分发取消", "\n".join(captured.output))
        self.assertTrue(all(value == "-" for value in get_log_context().values()))

    async def test_context_filter_records_current_task_name(self) -> None:
        import logging

        task = asyncio.current_task()
        previous_name = task.get_name()
        task.set_name("cleanup-job:123")
        try:
            record = logging.LogRecord("bot.test", logging.INFO, __file__, 1, "event", (), None)
            LogContextFilter().filter(record)
            self.assertEqual(record.task_name, "cleanup-job:123")
        finally:
            task.set_name(previous_name)


if __name__ == "__main__":
    unittest.main()
