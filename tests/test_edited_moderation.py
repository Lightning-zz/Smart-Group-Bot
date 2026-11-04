"""End-to-end edited-message moderation with durable, temporary state."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from bot.config import Settings
from bot.db.engine import init_db
from bot.db.models import BotScreening, Group, ModerationRule, UserWarning, Violation
from bot.handlers import group
from bot.services.authz import authorize_group
from bot.services.bot_screening import is_bot_whitelisted, record_bot_screening_pass
from bot.services.moderation import ModerationService, ModerationVerdict
from bot.services.moderation_revision import _reset_for_tests
from bot.utils.telegram import extract_message_text as real_extract_message_text


GROUP_ID = -10001
USER_ID = 42
BOT_ID = 777000123
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _message(*, bot: bool = False, text: str = "edited text", message_id: int = 701):
    return SimpleNamespace(
        chat=SimpleNamespace(id=GROUP_ID, type="supergroup", title="test"),
        from_user=SimpleNamespace(
            id=BOT_ID if bot else USER_ID,
            is_bot=bot,
            username="sender",
            full_name="Sender",
        ),
        sender_chat=None,
        text=text,
        caption=None,
        photo=None,
        video=None,
        animation=None,
        document=None,
        audio=None,
        contact=None,
        sticker=None,
        voice=None,
        video_note=None,
        location=None,
        date=NOW,
        edit_date=NOW + timedelta(seconds=1),
        message_id=message_id,
        delete=AsyncMock(return_value=True),
        bot=SimpleNamespace(me=AsyncMock(return_value=SimpleNamespace(id=1, username="selfbot"))),
    )


class EditedModerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _reset_for_tests()
        self.tempdir = tempfile.TemporaryDirectory()
        self.engine, self.session_factory = await init_db(
            f"sqlite+aiosqlite:///{Path(self.tempdir.name) / 'edited.db'}"
        )
        self.settings = Settings(_env_file=None)
        self.settings.super_admin_id = 1
        self.settings.moderation.enabled = True
        self.settings.moderation.bot_screening_enabled = True
        self.settings.moderation.bot_screening_message_count = 3
        self.settings.bot.auto_delete_categories = []
        async with self.session_factory() as session:
            session.add(Group(id=GROUP_ID, title="test", settings={}))
            await session.flush()
            await authorize_group(session, GROUP_ID)
            self.rule = ModerationRule(
                group_id=GROUP_ID, rule_type="keyword", pattern="advertising",
                action="delete", enabled=True,
            )
            session.add(self.rule)
            await session.commit()

        self.moderation = ModerationService(self.settings.moderation, object())
        self.moderation.evaluate = AsyncMock()
        self.memory = SimpleNamespace(
            archive_message=AsyncMock(), session_factory=self.session_factory,
        )
        self.notice = AsyncMock(return_value=True)
        self.reply = AsyncMock()
        self.stack = ExitStack()
        for patcher in (
            patch("bot.handlers.group.ensure_group_authorized", new=AsyncMock(return_value=True)),
            patch("bot.handlers.group.memory_holder.get", return_value=self.memory),
            patch("bot.handlers.group.memory_holder.get_optional", return_value=self.memory),
            patch("bot.handlers.group.extract_message_text", side_effect=lambda msg: (msg.text, "text")),
            patch("bot.handlers.group._is_user_admin_cached", new=AsyncMock(return_value=False)),
            patch("bot.handlers.group._append_image_context", new=AsyncMock(side_effect=lambda _m, _l, t, _mt: (t, ""))),
            patch("bot.handlers.group.LLMService", return_value=object()),
            patch("bot.handlers.group.ModerationService", return_value=self.moderation),
            patch("bot.handlers.group.answer_with_auto_delete", new=self.notice),
            patch("bot.handlers.group._enqueue_pending_reply", new=self.reply),
        ):
            self.stack.enter_context(patcher)

    async def asyncTearDown(self) -> None:
        self.stack.close()
        await self.engine.dispose()
        self.tempdir.cleanup()
        _reset_for_tests()

    def _verdict(self, violated: bool) -> ModerationVerdict:
        return ModerationVerdict(
            violated=violated, reason="advertising" if violated else "",
            rule=self.rule if violated else None, conclusive=True, confidence=0.99,
        )

    async def _edit(self, message) -> None:
        async with self.session_factory() as session:
            await group.on_group_message_edited(
                message, session=session, settings=self.settings,
                session_factory=self.session_factory,
            )

    async def _violations(self) -> list[Violation]:
        async with self.session_factory() as session:
            return list((await session.scalars(select(Violation))).all())

    async def test_human_edit_violation_deletes_and_records_notice(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message()
        await self._edit(message)
        message.delete.assert_awaited_once()
        self.notice.assert_awaited_once()
        self.assertEqual(len(await self._violations()), 1)
        self.assertIn("edited text", self.moderation.evaluate.await_args.args[2])

    async def test_clean_edit_never_runs_reply_pipeline(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(False)
        message = _message()
        with patch("bot.handlers.group.find_keyword_reply", new=AsyncMock()) as keyword, patch(
            "bot.handlers.group.handle_call_admin", new=AsyncMock()
        ) as call_admin:
            await self._edit(message)
        self.moderation.evaluate.assert_awaited_once()
        message.delete.assert_not_awaited()
        self.notice.assert_not_awaited()
        self.reply.assert_not_awaited()
        keyword.assert_not_awaited()
        call_admin.assert_not_awaited()

    async def test_whitelisted_bot_edit_violation_is_reviewed_and_revokes_trust(self) -> None:
        async with self.session_factory() as session:
            await record_bot_screening_pass(session, GROUP_ID, BOT_ID, threshold=1)
            await session.commit()
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(bot=True)
        await self._edit(message)
        self.moderation.evaluate.assert_awaited_once()
        message.delete.assert_awaited_once()
        self.notice.assert_awaited_once()
        async with self.session_factory() as session:
            self.assertFalse(await is_bot_whitelisted(session, GROUP_ID, BOT_ID))
            screening = await session.scalar(select(BotScreening).where(BotScreening.bot_id == BOT_ID))
            self.assertEqual(screening.passed_count, 0)

    async def test_clean_bot_edit_does_not_credit_screening_pass(self) -> None:
        async with self.session_factory() as session:
            await record_bot_screening_pass(session, GROUP_ID, BOT_ID, threshold=3)
            await session.commit()
        self.moderation.evaluate.return_value = self._verdict(False)
        message = _message(bot=True)
        await self._edit(message)
        self.moderation.evaluate.assert_awaited_once()
        async with self.session_factory() as session:
            screening = await session.scalar(select(BotScreening).where(BotScreening.bot_id == BOT_ID))
            self.assertEqual(screening.passed_count, 1)
            self.assertFalse(screening.whitelisted)
        message.delete.assert_not_awaited()
        self.reply.assert_not_awaited()

    async def test_archive_failure_does_not_suppress_review(self) -> None:
        self.memory.archive_message.side_effect = RuntimeError("archive unavailable")
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message()
        with self.assertLogs("bot.handlers.group", "ERROR"):
            await self._edit(message)
        self.moderation.evaluate.assert_awaited_once()
        message.delete.assert_awaited_once()
        self.assertEqual(len(await self._violations()), 1)

    async def test_caption_edit_uses_real_media_extractor_and_is_reviewed(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(text=None)
        message.video = SimpleNamespace(file_id="video-file")
        message.caption = "advertising caption"
        with patch("bot.handlers.group.extract_message_text", new=real_extract_message_text):
            await self._edit(message)
        self.assertIn("advertising caption", self.moderation.evaluate.await_args.args[2])
        message.delete.assert_awaited_once()
        self.reply.assert_not_awaited()

    async def test_hidden_link_edit_uses_real_extractor_and_reviews_destination(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(text="safe label")
        message.entities = [SimpleNamespace(
            type="text_link", offset=0, length=10, url="https://advertising.example",
        )]
        with patch("bot.handlers.group.extract_message_text", new=real_extract_message_text):
            await self._edit(message)
        submitted = self.moderation.evaluate.await_args.args[2]
        self.assertIn("safe label", submitted)
        self.assertIn("https://advertising.example", submitted)
        message.delete.assert_awaited_once()

    async def test_edit_reviews_quote_and_reply_ads_without_replaying_chat(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(text="check this")
        message.reply_to_message = _message(text="replied advertising", message_id=700)
        message.quote = SimpleNamespace(text="quoted advertising")

        await self._edit(message)

        submitted = self.moderation.evaluate.await_args.args[2]
        self.assertIn("replied advertising", submitted)
        self.assertIn("quoted advertising", submitted)
        self.assertIn("[reply-to-message]", submitted)
        self.assertIn("[quote]", submitted)
        message.delete.assert_awaited_once()
        self.reply.assert_not_awaited()

    async def test_captionless_video_edit_reviews_keyboard_via_real_extractor(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(bot=True, text=None)
        message.video = SimpleNamespace(file_id="video-file")
        message.reply_markup = SimpleNamespace(inline_keyboard=[[
            SimpleNamespace(text="advertising button", url="https://offer.example"),
        ]])
        with patch("bot.handlers.group.extract_message_text", new=real_extract_message_text):
            await self._edit(message)
        submitted = self.moderation.evaluate.await_args.args[2]
        self.assertIn("advertising button", submitted)
        self.assertIn("https://offer.example", submitted)
        self.assertNotEqual(submitted, "[video]")
        message.delete.assert_awaited_once()
        self.reply.assert_not_awaited()

    async def test_captionless_bot_video_edit_reviews_reply_image(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(bot=True, text=None)
        message.video = SimpleNamespace(file_id="video-file")
        message.reply_to_message = SimpleNamespace(
            photo=[SimpleNamespace(file_id="referenced-image")],
        )

        async def vision(_image, _llm, text, kind):
            return (text, "advertising in reply image" if kind == "photo" else "")

        with (
            patch("bot.handlers.group.extract_message_text", new=real_extract_message_text),
            patch("bot.handlers.group._append_image_context", new=AsyncMock(side_effect=vision)),
        ):
            await self._edit(message)
        submitted = self.moderation.evaluate.await_args.args[2]
        self.assertIn("advertising in reply image", submitted.deterministic_parts)
        self.assertIn("advertising in reply image", dict(submitted.semantic_sections)["reply-to-message"])
        message.delete.assert_awaited_once()
        self.reply.assert_not_awaited()

    async def test_reference_only_edit_reviews_available_external_image(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message(text=None)
        message.external_reply = SimpleNamespace(
            photo=[SimpleNamespace(file_id="external-image")],
        )

        async def vision(_image, _llm, text, kind):
            return (text, "advertising in external image" if kind == "photo" else "")

        with (
            patch("bot.handlers.group.extract_message_text", new=real_extract_message_text),
            patch("bot.handlers.group._append_image_context", new=AsyncMock(side_effect=vision)),
        ):
            await self._edit(message)
        submitted = self.moderation.evaluate.await_args.args[2]
        self.assertIn("advertising in external image", dict(submitted.semantic_sections)["external-reply"])
        message.delete.assert_awaited_once()
        self.reply.assert_not_awaited()

    async def test_replayed_edit_does_not_repeat_violation_or_notice(self) -> None:
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message()
        await self._edit(message)
        await self._edit(message)
        violations = await self._violations()
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].source_message_id, message.message_id)
        self.notice.assert_awaited_once()

    async def test_replayed_edit_does_not_increment_warning_count_twice(self) -> None:
        async with self.session_factory() as session:
            rule = await session.get(ModerationRule, self.rule.id)
            rule.action = "ban"
            await session.commit()
        self.rule.action = "ban"
        self.settings.moderation.warn_threshold = 3
        self.moderation.evaluate.return_value = self._verdict(True)
        message = _message()
        await self._edit(message)
        await self._edit(message)
        async with self.session_factory() as session:
            warning = await session.scalar(select(UserWarning).where(
                UserWarning.group_id == GROUP_ID,
                UserWarning.user_id == USER_ID,
            ))
            self.assertEqual(warning.count, 1)
            self.assertFalse(warning.is_banned)
        self.assertEqual(len(await self._violations()), 1)
        self.notice.assert_awaited_once()

    async def test_new_ban_edit_after_warn_updates_notice_and_warns_once(self) -> None:
        async with self.session_factory() as session:
            rule = await session.get(ModerationRule, self.rule.id)
            rule.action = "warn"
            await session.commit()
            self.rule = rule
        self.moderation.evaluate.return_value = self._verdict(True)
        original = _message(text="first violation")
        await self._edit(original)
        self.notice.assert_awaited_once()

        old_rule_id = self.rule.id
        async with self.session_factory() as session:
            rule = ModerationRule(
                group_id=GROUP_ID, rule_type="keyword", pattern="stronger violation",
                action="ban", enabled=True,
            )
            session.add(rule)
            await session.commit()
            self.rule = rule
        self.moderation.evaluate.return_value = self._verdict(True)
        edited = _message(text="stronger violation")
        edited.edit_date = original.edit_date + timedelta(seconds=1)
        await self._edit(edited)
        async with self.session_factory() as session:
            warning = await session.scalar(select(UserWarning).where(
                UserWarning.group_id == GROUP_ID,
                UserWarning.user_id == USER_ID,
            ))
            self.assertEqual(warning.count, 1)
        violations = await self._violations()
        self.assertEqual(len(violations), 1)
        self.assertIn("stronger violation", violations[0].message_text)
        self.assertEqual(violations[0].rule_id, self.rule.id)
        self.assertNotEqual(violations[0].rule_id, old_rule_id)
        self.assertEqual(self.notice.await_count, 2)
        await self._edit(edited)
        async with self.session_factory() as session:
            warning = await session.scalar(select(UserWarning).where(
                UserWarning.group_id == GROUP_ID,
                UserWarning.user_id == USER_ID,
            ))
            self.assertEqual(warning.count, 1)
        self.assertEqual(self.notice.await_count, 2)

    async def test_new_delete_edit_after_warn_updates_evidence_and_notice_once(self) -> None:
        async with self.session_factory() as session:
            rule = await session.get(ModerationRule, self.rule.id)
            rule.action = "warn"
            await session.commit()
            self.rule = rule
        self.moderation.evaluate.return_value = self._verdict(True)
        original = _message(text="first warning content")
        await self._edit(original)
        original.delete.assert_not_awaited()
        self.notice.assert_awaited_once()

        old_rule_id = self.rule.id
        async with self.session_factory() as session:
            rule = ModerationRule(
                group_id=GROUP_ID, rule_type="keyword", pattern="delete violation",
                action="delete", enabled=True,
            )
            session.add(rule)
            await session.commit()
            self.rule = rule
        self.moderation.evaluate.return_value = self._verdict(True)
        edited = _message(text="delete violation")
        edited.edit_date = original.edit_date + timedelta(seconds=1)
        await self._edit(edited)
        edited.delete.assert_awaited_once()
        violations = await self._violations()
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].action_taken, "delete")
        self.assertEqual(violations[0].rule_id, self.rule.id)
        self.assertNotEqual(violations[0].rule_id, old_rule_id)
        self.assertIn("delete violation", violations[0].message_text)
        self.assertEqual(self.notice.await_count, 2)
        await self._edit(edited)
        self.assertEqual(len(await self._violations()), 1)
        self.assertEqual(self.notice.await_count, 2)
        async with self.session_factory() as session:
            self.assertIsNone(await session.scalar(select(UserWarning).where(
                UserWarning.group_id == GROUP_ID, UserWarning.user_id == USER_ID,
            )))
    async def test_older_in_flight_verdict_is_discarded_after_new_clean_edit(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def evaluate(_session, _group_id, text):
            if "old content" in text:
                started.set()
                await release.wait()
                return self._verdict(True)
            return self._verdict(False)

        self.moderation.evaluate.side_effect = evaluate
        old = _message(text="old content")
        new = _message(text="new content")
        new.edit_date = old.edit_date + timedelta(seconds=1)
        task = asyncio.create_task(self._edit(old))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            await self._edit(new)
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=5)
        self.assertEqual(self.moderation.evaluate.await_count, 2)
        old.delete.assert_not_awaited()
        new.delete.assert_not_awaited()
        self.notice.assert_not_awaited()
        self.assertEqual(await self._violations(), [])
