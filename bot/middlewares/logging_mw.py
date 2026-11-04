from __future__ import annotations

import asyncio
import logging
import time
import zlib
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Update

from bot.services.update_completion import current_update_completion
from bot.utils.logging_setup import (
    attach_exception_log_context,
    reset_log_context,
    set_log_context,
)

log = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Trace dispatch for every update without claiming business completion."""

    @staticmethod
    def _build_flow_id(update_id: int, chat_id: int, message_id: int) -> str:
        seed = f"{update_id}:{chat_id}:{message_id}".encode("utf-8")
        return f"{zlib.crc32(seed) & 0xFFFFFFFF:08x}"

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        update_id = int(getattr(event, "update_id", 0) or 0)
        try:
            # Aiogram's accessor covers future event types as well as messages,
            # callbacks, member updates, inline queries and poll answers.
            payload = event.event
            update_type = event.event_type
        except (LookupError, AttributeError):
            payload = None
            update_type = "unknown"
        message = getattr(payload, "message", None)
        source = getattr(payload, "source", None) or getattr(
            getattr(payload, "boost", None), "source", None
        )
        user = (
            getattr(payload, "from_user", None)
            or getattr(payload, "user", None)
            or getattr(source, "user", None)
        )
        chat = getattr(payload, "chat", None) or getattr(message, "chat", None)
        chat_id = getattr(chat, "id", 0) or getattr(payload, "user_chat_id", 0) or 0
        user_id = getattr(user, "id", 0) or 0
        message_id = getattr(payload, "message_id", 0) or 0
        if not message_id:
            message_id = getattr(message, "message_id", 0) or 0
        req_id = f"{update_id}:{chat_id}:{message_id or update_type}"
        flow_id = self._build_flow_id(update_id, chat_id, message_id)
        tokens = set_log_context(
            req_id=req_id,
            flow_id=flow_id,
            update_id=update_id or "-",
            chat_id=chat_id or "-",
            user_id=user_id or "-",
            message_id=message_id or "-",
        )

        started = time.perf_counter()
        try:
            # Names, message text and callback_data may contain private data or
            # credentials. The correlation columns identify the event instead.
            log.info(
                "【入口】收到更新 | 类型=%s",
                update_type,
            )
            result = await handler(event, data)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            receipt = current_update_completion()
            state = (
                "已转后台/等待终态"
                if receipt is not None and receipt.deferred
                else "分发结束"
            )
            log.info("【出口】更新分发返回 | 状态=%s | 耗时=%dms", state, elapsed_ms)
            return result
        except asyncio.CancelledError:
            log.info(
                "【出口】更新分发取消 | 耗时=%dms",
                int((time.perf_counter() - started) * 1000),
            )
            raise
        except Exception as exc:
            attach_exception_log_context(exc)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            # The delivery boundary owns the traceback. Its LogRecord can
            # recover the saved IDs after this ContextVar scope is reset.
            log.error(
                "【出口】更新分发失败 | 异常=%s | 原因=%s | 耗时=%dms",
                type(exc).__name__,
                exc,
                elapsed_ms,
            )
            raise
        finally:
            reset_log_context(tokens)
