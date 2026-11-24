"""Tests for admin notifications of human challenges + the 解封 button."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bot.handlers import group
from bot.services import violation_notice


class BuildChallengeNoticeTests(unittest.TestCase):
    def test_notice_includes_numeric_user_id(self) -> None:
        text = violation_notice.build_challenge_admin_notice(
            sender_id=987654321,
            display_name="Spammer",
            username="spamguy",
            message_text="buy now",
            reason="疑似刷屏广告",
            group_title="测试群",
            timeout_seconds=600,
            kind="moderation",
        )
        self.assertIn("987654321", text)
        self.assertIn("buy now", text)
        self.assertIn("疑似刷屏广告", text)
        self.assertIn("600", text)
        self.assertIn("封禁", text)

    def test_notice_escapes_html(self) -> None:
        text = violation_notice.build_challenge_admin_notice(
            sender_id=1,
            display_name="<b>x</b>",
            username="",
            message_text="<script>alert(1)</script>",
            reason="<i>r</i>",
        )
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_unban_keyboard_carries_target(self) -> None:
        markup = violation_notice.build_challenge_unban_keyboard(-100, 42)
        button = markup.inline_keyboard[0][0]
        self.assertEqual(button.text, "解封")
        self.assertEqual(button.callback_data, "chun:unban:-100:42")


class NotifyChallengeTests(unittest.IsolatedAsyncioTestCase):
    async def test_每管理员发送一次且带键盘(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(
                return_value=[
                    SimpleNamespace(user=SimpleNamespace(id=11, is_bot=False)),
                    SimpleNamespace(user=SimpleNamespace(id=999, is_bot=True)),
                ]
            ),
            send_message=AsyncMock(),
        )
        delivered = await violation_notice.notify_admins_of_challenge(
            bot,
            SimpleNamespace(),
            group_id=-100,
            sender_id=42,
            message_text="ad",
            reason="spam",
            timeout_seconds=600,
        )
        self.assertEqual(delivered, 1)
        kwargs = bot.send_message.await_args.kwargs
        self.assertEqual(kwargs["chat_id"], 11)
        self.assertIsNotNone(kwargs["reply_markup"])


class ChallengeUnbanCallbackTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_valid_and_invalid(self) -> None:
        self.assertEqual(group._parse_challenge_unban_callback("chun:unban:-100:42"), (-100, 42))
        self.assertIsNone(group._parse_challenge_unban_callback("chun:unban:-100"))
        self.assertIsNone(group._parse_challenge_unban_callback("chun:other:-100:42"))
        self.assertIsNone(group._parse_challenge_unban_callback("x:unban:-100:42"))
        self.assertIsNone(group._parse_challenge_unban_callback("chun:unban:-100:0"))

    async def test_rejects_non_admin(self) -> None:
        callback = SimpleNamespace(
            data="chun:unban:-100:42",
            from_user=SimpleNamespace(id=7),
            bot=SimpleNamespace(),
            message=SimpleNamespace(chat=SimpleNamespace(id=-100)),
            answer=AsyncMock(),
        )
        session = SimpleNamespace(commit=AsyncMock(), in_transaction=lambda: False)
        with patch.object(group, "is_group_authorized", new=AsyncMock(return_value=True)), \
             patch.object(group, "is_group_admin_or_higher", new=AsyncMock(return_value=False)):
            await group.on_challenge_unban(callback, SimpleNamespace(), session=session)
        callback.answer.assert_awaited()
        self.assertIn("仅群管理员", callback.answer.await_args.args[0])

    async def test_global_ban_is_not_released(self) -> None:
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        with patch.object(group, "is_globally_banned", new=AsyncMock(return_value=True)):
            outcome = await group._release_challenge_restriction(
                bot=SimpleNamespace(), session=session, group_id=-100, user_id=42
            )
        self.assertIn("全局封禁", outcome)

    async def test_no_pending_record(self) -> None:
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        with patch.object(group, "is_globally_banned", new=AsyncMock(return_value=False)), \
             patch.object(group, "get_join_verification", new=AsyncMock(return_value=None)):
            outcome = await group._release_challenge_restriction(
                bot=SimpleNamespace(), session=session, group_id=-100, user_id=42
            )
        self.assertIn("没有进行中", outcome)

    async def test_releases_and_deletes_record(self) -> None:
        record = SimpleNamespace(id=5)
        recovery = SimpleNamespace(verification_id=5, group_id=-100, user_id=42, lease_until=None)
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        with patch.object(group, "is_globally_banned", new=AsyncMock(return_value=False)), \
             patch.object(group, "get_join_verification", new=AsyncMock(return_value=record)), \
             patch.object(group, "lease_join_verification_for_unban", new=AsyncMock(return_value=recovery)), \
             patch.object(group, "release_moderation_restriction_after_exemption", new=AsyncMock(return_value=True)), \
             patch.object(group, "activate_manual_unban_recovery"), \
             patch.object(group, "delete_join_verification", new=AsyncMock(return_value=True)):
            outcome = await group._release_challenge_restriction(
                bot=SimpleNamespace(), session=session, group_id=-100, user_id=42
            )
        self.assertIn("已解除", outcome)


class ReleaseChallengeDbTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from bot.db.engine import init_db
        from bot.services.authz import authorize_group

        fd, self._path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.engine, self.session_factory = await init_db(
            f"sqlite+aiosqlite:///{self._path}"
        )
        async with self.session_factory() as session:
            await authorize_group(session, -100, 1)
            await session.commit()

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(f"{self._path}{suffix}")
            except OSError:
                pass

    async def test_releases_restriction_and_clears_record(self) -> None:
        from datetime import timedelta

        from bot.services.join_verification import (
            get_join_verification,
            upsert_join_verification,
        )
        from bot.utils.timezone import now_shanghai_naive

        async with self.session_factory() as session:
            await upsert_join_verification(
                session,
                group_id=-100,
                user_id=42,
                deadline_at=now_shanghai_naive() + timedelta(minutes=5),
                kind="patrol",
                reason="资料命中群规",
                prompt_message_id=321,
            )
            await session.commit()

        bot = SimpleNamespace(
            restrict_chat_member=AsyncMock(return_value=True),
            ban_chat_member=AsyncMock(return_value=True),
            unban_chat_member=AsyncMock(return_value=True),
            get_chat_member=AsyncMock(
                return_value=SimpleNamespace(status="member", can_send_messages=False)
            ),
            edit_message_reply_markup=AsyncMock(return_value=True),
            delete_message=AsyncMock(return_value=True),
            edit_message_text=AsyncMock(return_value=True),
        )

        async with self.session_factory() as session:
            outcome = await group._release_challenge_restriction(
                bot=bot, session=session, group_id=-100, user_id=42
            )
            await session.commit()
        self.assertIn("已解除", outcome)

        async with self.session_factory() as session:
            self.assertIsNone(await get_join_verification(session, -100, 42))

    async def test_missing_record_does_not_touch_telegram(self) -> None:
        bot = SimpleNamespace(
            unban_chat_member=AsyncMock(),
            restrict_chat_member=AsyncMock(),
        )
        async with self.session_factory() as session:
            outcome = await group._release_challenge_restriction(
                bot=bot, session=session, group_id=-100, user_id=999
            )
            await session.commit()
        self.assertIn("没有进行中", outcome)
        bot.unban_chat_member.assert_not_awaited()


class LookupGroupTitleTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_group_title(self) -> None:
        from bot.db.engine import init_db
        from bot.db.models import Group

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        engine, session_factory = await init_db(f"sqlite+aiosqlite:///{path}")
        try:
            async with session_factory() as session:
                session.add(Group(id=-100, title="测试群", settings={}))
                await session.commit()
                title = await violation_notice.lookup_group_title(session, -100)
            self.assertEqual(title, "测试群")
            async with session_factory() as session:
                missing = await violation_notice.lookup_group_title(session, -999)
            self.assertEqual(missing, "")
        finally:
            await engine.dispose()
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(f"{path}{suffix}")
                except OSError:
                    pass


class RaidBatchWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_raid_batch_notify_called_with_ids(self) -> None:
        from contextlib import asynccontextmanager
        from datetime import datetime

        from bot.services import raid_guard
        from bot.services.raid_guard import RaidGuardService, RaidSuspect

        service = RaidGuardService.__new__(RaidGuardService)
        service.bot = SimpleNamespace()
        service.settings = SimpleNamespace(
            moderation=SimpleNamespace(notify_admin_on_challenge=True)
        )

        @asynccontextmanager
        async def _session():
            yield SimpleNamespace(commit=AsyncMock())

        service.session_factory = _session

        sent = []

        async def fake_batch(bot, session, **kwargs):
            sent.append(kwargs)
            return 1

        async def fake_title(_session, _group_id):
            return "G"

        with patch.object(raid_guard, "notify_admins_of_challenge_batch", new=fake_batch), \
             patch.object(raid_guard, "lookup_group_title", new=fake_title):
            await service._notify_admins_of_raid_challenge(
                -100,
                [
                    RaidSuspect(
                        user_id=9,
                        full_name="a",
                        username="u",
                        joined_at=datetime(2026, 1, 1),
                    )
                ],
                timeout_seconds=600,
            )

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["kind"], "raid")
        self.assertEqual(sent[0]["members"][0][0], 9)


if __name__ == "__main__":
    unittest.main()


class ChallengeToggleTests(unittest.TestCase):
    def test_toggle_reads_moderation_flag(self) -> None:
        self.assertTrue(
            violation_notice.challenge_notice_enabled(
                SimpleNamespace(moderation=SimpleNamespace(notify_admin_on_challenge=True))
            )
        )
        self.assertFalse(
            violation_notice.challenge_notice_enabled(
                SimpleNamespace(moderation=SimpleNamespace(notify_admin_on_challenge=False))
            )
        )
        # A missing moderation block or missing field must not raise.
        self.assertFalse(violation_notice.challenge_notice_enabled(SimpleNamespace()))
        self.assertFalse(
            violation_notice.challenge_notice_enabled(
                SimpleNamespace(moderation=SimpleNamespace())
            )
        )


class BatchChallengeNoticeTests(unittest.TestCase):
    def test_batch_notice_lists_ids_and_reasons(self) -> None:
        text = violation_notice.build_challenge_batch_admin_notice(
            members=[
                (111, "卖广告的", "ad_guy", "昵称含广告"),
                (222, "", "", "资料命中群规"),
            ],
            group_title="测试群",
            timeout_seconds=600,
            kind="patrol",
        )
        self.assertIn("111", text)
        self.assertIn("222", text)
        self.assertIn("昵称含广告", text)
        self.assertIn("资料命中群规", text)
        self.assertIn("资料巡检质询", text)
        self.assertIn("移出群聊", text)

    def test_batch_notice_escapes_html_in_reason(self) -> None:
        text = violation_notice.build_challenge_batch_admin_notice(
            members=[(1, "<b>x</b>", "", "<i>r</i>")],
        )
        self.assertNotIn("<i>r</i>", text)
        self.assertIn("&lt;i&gt;r&lt;/i&gt;", text)

    def test_batch_keyboard_has_one_button_per_member(self) -> None:
        markup = violation_notice.build_challenge_unban_keyboard_batch(-100, [1, 2])
        self.assertEqual(len(markup.inline_keyboard), 2)
        self.assertEqual(markup.inline_keyboard[0][0].callback_data, "chun:unban:-100:1")
        self.assertEqual(markup.inline_keyboard[1][0].callback_data, "chun:unban:-100:2")


class JoinVerificationIsNotNotifiedTests(unittest.TestCase):
    """入群验证不应通知管理员：每个新成员都会触发，加入量不可控。"""

    def test_membership_has_no_challenge_notice_wiring(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "bot" / "handlers" / "membership.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("notify_admins_of_challenge", source)
        self.assertNotIn("_notify_admins_of_join_challenge", source)

    def test_join_kind_is_not_a_notifiable_label(self) -> None:
        self.assertNotIn("join", violation_notice._CHALLENGE_KIND_LABELS)


class BatchChallengeNotifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_notify_sends_to_each_admin(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(
                return_value=[
                    SimpleNamespace(user=SimpleNamespace(id=11, is_bot=False)),
                    SimpleNamespace(user=SimpleNamespace(id=12, is_bot=False)),
                ]
            ),
            send_message=AsyncMock(),
        )
        delivered = await violation_notice.notify_admins_of_challenge_batch(
            bot,
            SimpleNamespace(),
            group_id=-100,
            members=[(42, "u", "h", "原因")],
            group_title="G",
            timeout_seconds=600,
            kind="raid",
        )
        self.assertEqual(delivered, 2)
        self.assertEqual(bot.send_message.await_count, 2)
        self.assertIsNotNone(bot.send_message.await_args.kwargs["reply_markup"])

    async def test_batch_notify_no_members_is_noop(self) -> None:
        bot = SimpleNamespace(send_message=AsyncMock())
        delivered = await violation_notice.notify_admins_of_challenge_batch(
            bot, SimpleNamespace(), group_id=-100, members=[]
        )
        self.assertEqual(delivered, 0)
        bot.send_message.assert_not_awaited()


class RemoveUnbanButtonTests(unittest.TestCase):
    def test_keeps_other_members_buttons(self) -> None:
        from aiogram.types import InlineKeyboardMarkup

        markup = violation_notice.build_challenge_unban_keyboard_batch(-100, [1, 2])
        remaining = group._remove_challenge_unban_button(markup, "chun:unban:-100:1")
        self.assertIsInstance(remaining, InlineKeyboardMarkup)
        self.assertEqual(len(remaining.inline_keyboard), 1)
        self.assertEqual(
            remaining.inline_keyboard[0][0].callback_data, "chun:unban:-100:2"
        )

    def test_last_button_returns_none(self) -> None:
        markup = violation_notice.build_challenge_unban_keyboard(-100, 1)
        self.assertIsNone(group._remove_challenge_unban_button(markup, "chun:unban:-100:1"))

    def test_unrelated_data_keeps_all_buttons(self) -> None:
        markup = violation_notice.build_challenge_unban_keyboard_batch(-100, [1, 2])
        remaining = group._remove_challenge_unban_button(markup, "chun:unban:-100:99")
        self.assertEqual(len(remaining.inline_keyboard), 2)


class BatchWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_patrol_batch_notify_called_with_ids(self) -> None:
        from contextlib import asynccontextmanager

        from bot.services import patrol
        from bot.services.patrol import PatrolService, PatrolViolator

        service = PatrolService.__new__(PatrolService)
        service.bot = SimpleNamespace()
        service.settings = SimpleNamespace(
            moderation=SimpleNamespace(notify_admin_on_challenge=True)
        )

        @asynccontextmanager
        async def _session():
            yield SimpleNamespace(commit=AsyncMock())

        service.session_factory = _session

        sent = []

        async def fake_batch(bot, session, **kwargs):
            sent.append(kwargs)
            return 1

        async def fake_title(_session, _group_id):
            return "G"

        with patch.object(patrol, "notify_admins_of_challenge_batch", new=fake_batch), \
             patch.object(patrol, "lookup_group_title", new=fake_title):
            await service._notify_admins_of_patrol_challenge(
                -100,
                [PatrolViolator(user_id=7, full_name="a", username="u", reason="原因")],
                timeout_seconds=600,
            )

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["kind"], "patrol")
        self.assertEqual(sent[0]["members"][0][0], 7)
        self.assertEqual(sent[0]["members"][0][3], "原因")


if __name__ == "__main__":
    unittest.main()
