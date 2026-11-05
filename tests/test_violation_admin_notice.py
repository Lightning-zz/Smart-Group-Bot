"""Tests for the private admin notification of moderation violations."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.services import violation_notice


class BuildNoticeTests(unittest.TestCase):
    def test_notice_contains_sender_message_and_rule(self) -> None:
        rule = SimpleNamespace(id=9, rule_type="keyword", pattern="spam-token")
        text = violation_notice.build_violation_admin_notice(
            sender_id=555,
            display_name="张三",
            username="zhangsan",
            message_text="buy spam-token now",
            rule=rule,
            action_taken="ban",
            reason="广告刷屏",
            group_title="测试群",
        )
        self.assertIn("违规通知", text)
        self.assertIn("@zhangsan", text)
        self.assertIn("测试群", text)
        self.assertIn("已删除违规消息并封禁用户", text)
        self.assertIn("#9", text)
        self.assertIn("spam-token", text)
        self.assertIn("buy spam-token now", text)
        self.assertIn("广告刷屏", text)

    def test_notice_escapes_html_in_message(self) -> None:
        text = violation_notice.build_violation_admin_notice(
            sender_id=1,
            display_name="<b>evil</b>",
            username="",
            message_text="<script>alert(1)</script>",
            rule=None,
            action_taken="delete",
        )
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_missing_rule_uses_ai_fallback_label(self) -> None:
        text = violation_notice.build_violation_admin_notice(
            sender_id=1,
            display_name="",
            username="",
            message_text="",
            rule=None,
            action_taken="warn",
        )
        self.assertIn("AI 语义判定", text)
        self.assertIn("无文本内容", text)


class NotifyDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_delivers_to_each_human_admin_excluding_bot(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(
                return_value=[
                    SimpleNamespace(user=SimpleNamespace(id=999, is_bot=True, username="self")),
                    SimpleNamespace(user=SimpleNamespace(id=11, is_bot=False, username="a")),
                    SimpleNamespace(user=SimpleNamespace(id=22, is_bot=False, username="b")),
                    SimpleNamespace(user=SimpleNamespace(id=33, is_bot=True, username="otherbot")),
                ]
            ),
            send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
        )
        session = SimpleNamespace()
        delivered = await violation_notice.notify_admins_of_violation(
            bot,
            session,
            group_id=-100,
            sender_id=555,
            message_text="bad",
            rule=None,
            action_taken="ban",
            display_name="u",
            username="u",
        )
        self.assertEqual(delivered, 2)
        sent_ids = {call.kwargs["chat_id"] for call in bot.send_message.await_args_list}
        self.assertEqual(sent_ids, {11, 22})

    async def test_send_failure_is_swallowed(self) -> None:
        bot = SimpleNamespace(
            me=AsyncMock(return_value=SimpleNamespace(id=999)),
            get_chat_administrators=AsyncMock(
                return_value=[SimpleNamespace(user=SimpleNamespace(id=11, is_bot=False))]
            ),
            send_message=AsyncMock(side_effect=RuntimeError("blocked")),
        )
        delivered = await violation_notice.notify_admins_of_violation(
            bot,
            SimpleNamespace(),
            group_id=-100,
            sender_id=555,
            message_text="bad",
            rule=None,
            action_taken="ban",
        )
        self.assertEqual(delivered, 0)


if __name__ == "__main__":
    unittest.main()
