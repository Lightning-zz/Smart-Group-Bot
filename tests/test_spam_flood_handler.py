"""Handler-level tests for the flood-ad screening integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bot.handlers import group
from bot.services import spam_flood


def _settings(**overrides):
    moderation = SimpleNamespace(
        enabled=True,
        spam_flood_enabled=True,
        spam_flood_window_seconds=300,
        spam_flood_min_messages=4,
        spam_flood_similarity_min_hits=3,
        spam_flood_similarity_threshold=0.85,
        spam_flood_strong_score=0.85,
        spam_flood_weak_score=0.5,
        spam_flood_notify_admin=True,
    )
    for key, value in overrides.items():
        setattr(moderation, key, value)
    return SimpleNamespace(moderation=moderation)


def _message(*, text="ad", bot=None):
    return SimpleNamespace(
        bot=bot or SimpleNamespace(send_message=AsyncMock()),
        chat=SimpleNamespace(id=-100, title="测试群"),
        message_id=1,
        text=text,
        delete=AsyncMock(),
    )


class FloodHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        spam_flood.reset_state()

    async def test_disabled_feature_does_not_observe(self) -> None:
        session = SimpleNamespace(add=MagicMock(), flush=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())
        handled = await group._screen_message_for_flood(
            session=session,
            message=_message(),
            settings=_settings(spam_flood_enabled=False),
            group_id=-100,
            user_id=7,
            message_text="ad",
            sender_identity=SimpleNamespace(display_name="x", username="x"),
        )
        self.assertFalse(handled)

    async def test_strong_hit_deletes_and_records(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(return_value=[]),
            send_message=AsyncMock(),
        )
        session = SimpleNamespace(add=MagicMock(), flush=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())
        message = _message(bot=bot)
        identity = SimpleNamespace(display_name="Spammer", username="spamguy")
        settings = _settings()

        delete_recent = AsyncMock(return_value=3)
        ban = AsyncMock(return_value=True)
        with patch.object(group, "delete_recent_member_messages", new=delete_recent), \
             patch.object(group, "moderation_challenge_ready", return_value=False), \
             patch.object(group, "ban_member", new=ban):
            handled = False
            for index in range(4):
                handled = await group._screen_message_for_flood(
                    session=session,
                    message=message,
                    settings=settings,
                    group_id=-100,
                    user_id=7,
                    message_text="限时优惠 加@xinbi888 领取 https://t.me/abc",
                    sender_identity=identity,
                )
        self.assertTrue(handled)
        message.delete.assert_awaited()
        delete_recent.assert_awaited()
        ban.assert_awaited_once()

    async def test_strong_hit_issues_challenge_when_available(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(return_value=[]),
            send_message=AsyncMock(),
        )
        session = SimpleNamespace(add=MagicMock(), flush=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())
        message = _message(bot=bot)
        identity = SimpleNamespace(display_name="Spammer", username="spamguy")
        settings = _settings()

        challenge = AsyncMock(return_value=True)
        ban = AsyncMock(return_value=True)
        with patch.object(group, "delete_recent_member_messages", new=AsyncMock(return_value=2)), \
             patch.object(group, "moderation_challenge_ready", return_value=True), \
             patch.object(group, "begin_moderation_challenge", new=challenge), \
             patch.object(group, "ban_member", new=ban):
            handled = False
            for index in range(4):
                handled = await group._screen_message_for_flood(
                    session=session,
                    message=message,
                    settings=settings,
                    group_id=-100,
                    user_id=7,
                    message_text="限时优惠 加@xinbi888 领取 https://t.me/abc",
                    sender_identity=identity,
                )
        self.assertTrue(handled)
        challenge.assert_awaited_once()
        self.assertEqual(challenge.await_args.kwargs["rule_action"], "ban")
        ban.assert_not_awaited()

    async def test_weak_hit_notifies_but_does_not_handle(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(
                return_value=[SimpleNamespace(user=SimpleNamespace(id=11, is_bot=False))]
            ),
            send_message=AsyncMock(),
        )
        session = SimpleNamespace(add=MagicMock(), flush=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())
        message = _message(bot=bot)
        identity = SimpleNamespace(display_name="u", username="u")
        settings = _settings()

        handled = False
        for _ in range(4):
            handled = await group._screen_message_for_flood(
                session=session,
                message=message,
                settings=settings,
                group_id=-100,
                user_id=7,
                message_text="出二手手机联系我",
                sender_identity=identity,
            )
        self.assertFalse(handled)
        message.delete.assert_not_awaited()
        self.assertGreaterEqual(bot.send_message.await_count, 1)


if __name__ == "__main__":
    unittest.main()
