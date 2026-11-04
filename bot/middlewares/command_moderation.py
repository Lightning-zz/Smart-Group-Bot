"""Moderate matched group commands before their handler can cause side effects."""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.services.authz import is_group_authorized
from bot.services.moderation_revision import (
    is_current_moderation_revision,
    moderation_revision_scope,
)

log = logging.getLogger(__name__)


class CommandModerationMiddleware(BaseMiddleware):
    """Inner middleware: only matched command/admin handlers, after DB setup.

    Unmatched messages retain the normal group pipeline. Private and currently
    unauthorized chats retain the command's own access checks, so `/authgroup`
    remains usable. Once review starts in an authorized group, revocation or a
    newer edit blocks execution rather than allowing an obsolete command.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def _authorized(self, group_id: int) -> bool:
        # Avoid the handler session's identity map and potentially old WAL
        # snapshot after a slow model call.
        async with self.session_factory() as session:
            return await is_group_authorized(session, group_id)

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if not event.chat or event.chat.type not in {"group", "supergroup"}:
            return await handler(event, data)

        from bot.handlers.group import on_group_message

        with moderation_revision_scope(event):
            if not is_current_moderation_revision():
                return None
            authorized = await self._authorized(int(event.chat.id))
            if not is_current_moderation_revision():
                return None
            if not authorized:
                return await handler(event, data)

            allowed = await on_group_message(
                event,
                session=data["session"],
                settings=data["settings"],
                session_factory=self.session_factory,
                _moderation_only=True,
            )
            if allowed is not True or not is_current_moderation_revision():
                log.info(
                    "命令执行已阻断 | group=%s message=%s reason=moderation_or_stale",
                    event.chat.id, event.message_id,
                )
                return None
            if not await self._authorized(int(event.chat.id)) or not is_current_moderation_revision():
                log.info(
                    "命令执行已阻断 | group=%s message=%s reason=authorization_or_stale",
                    event.chat.id, event.message_id,
                )
                return None
            return await handler(event, data)
