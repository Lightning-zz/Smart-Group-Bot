"""Matched commands must not bypass the group content review."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message, Update, User
from sqlalchemy import select

from bot.config import Settings
from bot.db.engine import init_db
from bot.db.models import BotScreening, Group, ModerationExemption, ModerationRule, Violation
from bot.handlers import group
from bot.middlewares.command_moderation import CommandModerationMiddleware
from bot.middlewares.db import DbSessionMiddleware
from bot.services.authz import authorize_group, deauthorize_group
from bot.services.bot_screening import record_bot_screening_pass
from bot.services.moderation import ModerationVerdict
from bot.services.moderation_revision import _reset_for_tests, moderation_revision_scope


GROUP_ID = -10001
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _message(**overrides):
    values = dict(
        chat=NS(id=GROUP_ID, type="supergroup", title="test"),
        from_user=NS(id=42, is_bot=False, username="sender", full_name="Sender"),
        sender_chat=None, text="/help", caption=None, photo=None, video=None,
        animation=None, document=None, audio=None, contact=None, sticker=None,
        voice=None, video_note=None, location=None, reply_markup=None, quote=None,
        reply_to_message=None, external_reply=None, entities=None,
        date=NOW, edit_date=None, message_id=701,
        delete=AsyncMock(return_value=True),
        bot=NS(me=AsyncMock(return_value=NS(id=1, username="selfbot"))),
    )
    values.update(overrides)
    return NS(**values)


class CommandModerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.engine, self.factory = await init_db(
            f"sqlite+aiosqlite:///{Path(self.tmp.name) / 'commands.db'}"
        )
        self.settings = Settings(_env_file=None)
        self.settings.super_admin_id = 1
        self.settings.moderation.enabled = True
        self.settings.moderation.bot_screening_enabled = True
        self.settings.moderation.bot_screening_message_count = 3
        self.settings.bot.auto_delete_categories = []
        async with self.factory() as session:
            session.add(Group(id=GROUP_ID, title="test", settings={}))
            await session.flush()
            await authorize_group(session, GROUP_ID)
            session.add(ModerationRule(
                group_id=GROUP_ID, rule_type="keyword", pattern="advertising",
                action="delete", enabled=True,
            ))
            await session.commit()
        self.middleware = CommandModerationMiddleware(self.factory)
        self.handler = AsyncMock(return_value="command executed")
        self.memory = NS(archive_message=AsyncMock(), session_factory=self.factory)
        self.notice = AsyncMock(return_value=True)
        self.activity = Mock()
        self.archive_activity = AsyncMock()
        self.reply = AsyncMock()
        self.admin = AsyncMock(return_value=False)
        self.llm = NS(moderation=AsyncMock(return_value='{"violated":false}'))
        self.stack = ExitStack()
        for item in (
            patch("bot.handlers.group.memory_holder.get", return_value=self.memory),
            patch("bot.handlers.group.memory_holder.get_optional", return_value=self.memory),
            patch("bot.handlers.group._is_user_admin_cached", self.admin),
            patch("bot.handlers.group._append_image_context", AsyncMock(side_effect=lambda _m, _l, t, _mt: (t, ""))),
            patch("bot.handlers.group._reference_vision", AsyncMock(return_value=("", ""))),
            patch("bot.handlers.group.LLMService", return_value=self.llm),
            patch("bot.handlers.group.answer_with_auto_delete", self.notice),
            patch("bot.handlers.group.note_group_activity", self.activity),
            patch("bot.handlers.group._record_group_activity_cas", self.archive_activity),
            patch("bot.handlers.group._enqueue_pending_reply", self.reply),
        ):
            self.stack.enter_context(item)

    async def asyncTearDown(self):
        self.stack.close()
        await self.engine.dispose()
        self.tmp.cleanup()
        _reset_for_tests()

    async def run_command(self, message=None):
        async with self.factory() as session:
            result = await self.middleware(self.handler, message or _message(), {
                "session": session, "settings": self.settings,
                "session_factory": self.factory,
            })
            await session.commit()
            return result

    async def test_advertising_in_every_command_evidence_field_blocks_command(self):
        cases = (
            {"text": "/help advertising"},
            {"quote": NS(text="advertising")},
            {"reply_to_message": NS(text="advertising")},
            {"external_reply": NS(text="advertising")},
            {"reply_markup": NS(inline_keyboard=[[NS(text="advertising")]])},
            {"entities": [NS(url="https://advertising.example")]},
        )
        for i, evidence in enumerate(cases):
            with self.subTest(evidence=evidence):
                message = _message(message_id=701 + i, **evidence)
                self.assertIsNone(await self.run_command(message))
                message.delete.assert_awaited_once()
        self.handler.assert_not_awaited()
        async with self.factory() as session:
            self.assertEqual(len((await session.scalars(select(Violation))).all()), len(cases))

    async def test_clean_command_has_no_group_conversation_side_effects(self):
        self.assertEqual(await self.run_command(), "command executed")
        self.handler.assert_awaited_once()
        self.memory.archive_message.assert_not_awaited()
        self.archive_activity.assert_not_awaited()
        self.activity.assert_not_called()
        self.reply.assert_not_awaited()

    async def test_disabled_moderation_allows_command(self):
        self.settings.moderation.enabled = False
        self.assertEqual(await self.run_command(_message(text="/help advertising")), "command executed")

    async def test_admin_exemption_allows_command(self):
        self.admin.return_value = True
        self.assertEqual(await self.run_command(_message(text="/help advertising")), "command executed")

    async def test_manual_exemption_allows_command(self):
        async with self.factory() as session:
            session.add(ModerationExemption(group_id=GROUP_ID, user_id=42, created_by=1))
            await session.commit()
        self.assertEqual(await self.run_command(_message(text="/help advertising")), "command executed")

    async def test_private_command_preserves_its_original_authorization(self):
        message = _message(chat=NS(id=42, type="private"), text="/help advertising")
        with patch("bot.handlers.group.on_group_message", AsyncMock()) as review:
            self.assertEqual(await self.run_command(message), "command executed")
        review.assert_not_awaited()

    async def test_unauthorized_group_preserves_authorization_commands(self):
        async with self.factory() as session:
            await deauthorize_group(session, GROUP_ID)
            await session.commit()
        with patch("bot.handlers.group.on_group_message", AsyncMock()) as review:
            self.assertEqual(await self.run_command(_message(text="/authgroup")), "command executed")
        review.assert_not_awaited()

    async def test_authorization_revocation_during_review_blocks_command(self):
        async def review(*args, **kwargs):
            async with self.factory() as session:
                await deauthorize_group(session, GROUP_ID)
                await session.commit()
            return True
        with patch("bot.handlers.group.on_group_message", AsyncMock(side_effect=review)):
            self.assertIsNone(await self.run_command())
        self.handler.assert_not_awaited()

    async def test_rule_change_during_clean_review_blocks_command(self):
        async with self.factory() as session:
            rule = await session.scalar(select(ModerationRule))
            rule.rule_type = "llm"
            rule.pattern = "No advertising"
            await session.commit()

        async def change_policy(*args, **kwargs):
            async with self.factory() as session:
                session.add(ModerationRule(
                    group_id=GROUP_ID, rule_type="keyword", pattern="help",
                    action="delete", enabled=True,
                ))
                await session.commit()
            return '{"violated":false}'

        self.llm.moderation.side_effect = change_policy
        self.assertIsNone(await self.run_command())
        self.handler.assert_not_awaited()

    async def test_clean_human_review_releases_claim_before_handler(self):
        async def handler(_message, data):
            self.assertFalse(data["session"].in_transaction())
            return "command executed"

        self.handler.side_effect = handler
        self.assertEqual(await self.run_command(), "command executed")

    async def test_no_enabled_rules_clean_fingerprint_still_allows_command(self):
        async with self.factory() as session:
            rule = await session.scalar(select(ModerationRule))
            rule.enabled = False
            await session.commit()
        self.assertEqual(await self.run_command(), "command executed")

    async def test_newer_edit_during_review_blocks_obsolete_command(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def review(*args, **kwargs):
            entered.set()
            await release.wait()
            return True
        with patch("bot.handlers.group.on_group_message", AsyncMock(side_effect=review)):
            pending = asyncio.create_task(self.run_command())
            await asyncio.wait_for(entered.wait(), timeout=2)
            with moderation_revision_scope(_message(edit_date=NOW + timedelta(seconds=1))):
                pass
            release.set()
            self.assertIsNone(await pending)
        self.handler.assert_not_awaited()

    async def test_no_violation_inconclusive_result_allows_without_bot_credit(self):
        verdict = ModerationVerdict(False, "", None, False)
        message = _message(from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"))
        with patch("bot.services.moderation.ModerationService.evaluate", AsyncMock(return_value=verdict)):
            self.assertEqual(await self.run_command(message), "command executed")
        async with self.factory() as session:
            self.assertIsNone(await session.scalar(select(BotScreening)))

    async def test_inconclusive_violation_blocks_command_without_punishment(self):
        verdict = ModerationVerdict(True, "possible ad", None, False)
        message = _message()
        with patch("bot.services.moderation.ModerationService.evaluate", AsyncMock(return_value=verdict)):
            self.assertIsNone(await self.run_command(message))
        self.handler.assert_not_awaited()
        message.delete.assert_not_awaited()

    async def test_bot_new_command_earns_exactly_one_clean_message(self):
        message = _message(from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"))
        self.assertEqual(await self.run_command(message), "command executed")
        async with self.factory() as session:
            row = await session.scalar(select(BotScreening))
            self.assertIsNotNone(row)
            self.assertEqual(row.passed_count, 1)

    async def test_failed_current_or_reference_ocr_cannot_credit_clean_bot_command(self):
        for i, fields in enumerate((
            {"photo": [NS(file_id="image")]},
            {"reply_to_message": NS(photo=[NS(file_id="image")])},
            {"external_reply": NS(document=NS(mime_type="image/png"))},
        )):
            message = _message(message_id=701 + i, from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"), **fields)
            self.assertEqual(await self.run_command(message), "command executed")
        async with self.factory() as session:
            self.assertIsNone(await session.scalar(select(BotScreening)))

    async def test_incomplete_bot_screening_releases_transaction_before_handler(self):
        message = _message(
            from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"),
            reply_to_message=NS(photo=[NS(file_id="image")]),
        )
        async def handler(_message, data):
            self.assertFalse(data["session"].in_transaction())
            # A second session must be able to acquire SQLite's writer lock
            # while the command handler is still running.
            async with self.factory() as concurrent:
                from bot.services.group_settings import acquire_group_settings_write_intent
                await asyncio.wait_for(acquire_group_settings_write_intent(concurrent, GROUP_ID), timeout=2)
                await concurrent.rollback()
            return "command executed"

        self.handler.side_effect = handler
        self.assertEqual(await self.run_command(message), "command executed")

    async def test_clean_bot_edit_releases_uncredited_claim_transaction(self):
        message = _message(
            from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"),
            edit_date=NOW + timedelta(seconds=1),
        )
        async with self.factory() as session:
            with moderation_revision_scope(message):
                allowed = await group._screen_bot_sender_message(
                    moderation=group.ModerationService(self.settings.moderation, self.llm),
                    session=session, settings=self.settings, message=message,
                    group_id=GROUP_ID, bot_id=77, input_text="/help", warn_target="Guest",
                    flow_started=asyncio.get_running_loop().time(), credit_clean_message=False,
                )
            self.assertIs(allowed, True)
            self.assertFalse(session.in_transaction())
        async with self.factory() as session:
            self.assertIsNone(await session.scalar(select(BotScreening)))

    async def test_failed_ocr_does_not_exempt_visible_advertising(self):
        message = _message(text="/help advertising", photo=[NS(file_id="image")], from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"))
        self.assertIsNone(await self.run_command(message))
        message.delete.assert_awaited_once()
        self.handler.assert_not_awaited()

    async def test_whitelisted_bot_new_command_keeps_existing_exemption(self):
        async with self.factory() as session:
            await record_bot_screening_pass(session, GROUP_ID, 77, threshold=1)
            await session.commit()
        message = _message(text="/help advertising", from_user=NS(id=77, is_bot=True, username="guest", full_name="Guest"))
        self.assertEqual(await self.run_command(message), "command executed")
        message.delete.assert_not_awaited()

    async def test_aiogram_matched_command_runs_db_then_review_exactly_once(self):
        bot = Bot("1:TEST_TOKEN")
        dp = Dispatcher(settings=self.settings)
        dp.message.middleware(DbSessionMiddleware(self.factory))
        router = Router()
        router.message.middleware(self.middleware)
        command = AsyncMock(return_value="handled")
        async def command_handler(message):
            return await command(message)
        router.message.register(command_handler, Command("help"))
        fallback = AsyncMock()
        async def fallback_handler(message):
            return await fallback(message)
        fallback_router = Router()
        fallback_router.message.register(fallback_handler)
        dp.include_router(router)
        dp.include_router(fallback_router)
        message = Message.model_validate({
            "message_id": 701, "date": NOW, "chat": {"id": GROUP_ID, "type": "supergroup"},
            "from": {"id": 42, "is_bot": False, "first_name": "Sender"}, "text": "/help",
        })
        try:
            with patch.object(bot, "me", AsyncMock(return_value=User(id=1, is_bot=True, first_name="Self", username="selfbot"))):
                with patch("bot.handlers.group.on_group_message", wraps=group.on_group_message) as review:
                    await dp.feed_update(bot, Update(update_id=1, message=message))
            review.assert_awaited_once()
            self.assertIsNotNone(review.await_args.kwargs["session"])
            command.assert_awaited_once()
            fallback.assert_not_awaited()
        finally:
            await bot.session.close()


if __name__ == "__main__":
    unittest.main()
