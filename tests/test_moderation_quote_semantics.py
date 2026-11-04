"""Regression tests for quote provenance without a source-based exemption."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from bot.config import ModerationConfig
from bot.db.models import ModerationRule
from bot.services.moderation import ModerationService
from bot.services.moderation_input import build_moderation_input


class _NoAutoflush:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


class QuoteSemanticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rule = ModerationRule(
            id=7, group_id=-100, rule_type="llm", pattern="No advertising",
            action="ban", enabled=True,
        )
        self.session = NS(
            no_autoflush=_NoAutoflush(),
            execute=AsyncMock(return_value=_Rows([self.rule])),
            commit=AsyncMock(),
        )

    def service(self, response):
        return ModerationService(
            ModerationConfig(), NS(moderation=AsyncMock(return_value=response)),
        )

    async def test_quote_only_advertising_is_submitted_and_enforced(self):
        service = self.service(
            '{"violated":true,"confidence":0.99,"rule_id":7,"reason":"引用传播广告"}'
        )
        message = NS(quote=NS(text="低价出售，请联系 @offer"))

        verdict = await service.evaluate(
            self.session, -100, build_moderation_input(message),
        )

        self.assertTrue(verdict.violated)
        self.assertTrue(service.is_high_confidence(verdict))
        system, user = service.llm.moderation.await_args.args
        self.assertIn("[quote]", user)
        self.assertIn("低价出售，请联系 @offer", user)
        self.assertIn("they do not exempt", system)
        self.assertIn("Never dismiss evidence merely because", system)

    async def test_reply_context_keeps_authorship_and_current_words(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(
            text="请管理员处理，这个链接是诈骗，不要点击。",
            reply_to_message=NS(
                text="立即购买优惠商品",
                entities=[NS(type="text_link", url="https://offer.example")],
            ),
        )

        verdict = await service.evaluate(
            self.session, -100, build_moderation_input(message),
        )

        self.assertFalse(verdict.violated)
        self.assertTrue(verdict.conclusive)
        system, user = service.llm.moderation.await_args.args
        self.assertIn("[current-message]", user)
        self.assertIn("[reply-to-message]", user)
        self.assertIn(message.text, user)
        self.assertIn("立即购买优惠商品", user)
        self.assertIn("https://offer.example", user)
        self.assertIn("distinguish the original author", system)
        self.assertIn("claiming to report", system)

    async def test_reporting_claim_does_not_override_semantic_violation(self):
        service = self.service(
            '{"violated":true,"confidence":0.99,"rule_id":7}'
        )
        message = NS(
            text="举报一下这个广告，感兴趣赶紧加他。",
            quote=NS(text="加 @offer 可领优惠"),
        )

        verdict = await service.evaluate(
            self.session, -100, build_moderation_input(message),
        )

        self.assertTrue(verdict.violated)
        self.assertTrue(service.is_high_confidence(verdict))
        self.assertIn("加 @offer 可领优惠", service.llm.moderation.await_args.args[1])

    async def test_long_reply_preserves_provenance_and_current_message_on_every_part(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(
            text="请管理员处理，不要点击。",
            reply_to_message=NS(text="普通内容" * 1500 + "广告尾部"),
        )

        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertTrue(verdict.conclusive)
        reference_calls = [
            call.args[1] for call in service.llm.moderation.await_args_list
            if "[reply-to-message]" in call.args[1]
        ]
        self.assertGreater(len(reference_calls), 1)
        for user in reference_calls:
            self.assertIn("[current-message]", user)
            self.assertIn(message.text, user)
        self.assertIn("广告尾部", reference_calls[-1])

    async def test_long_quote_advertising_tail_can_still_trigger_penalty(self):
        async def review(system, user):
            if "广告尾部" in user:
                return '{"violated":true,"confidence":0.99,"rule_id":7}'
            return '{"violated":false,"confidence":0.99}'

        service = self.service("")
        service.llm.moderation.side_effect = review
        message = NS(text="分享一下", quote=NS(text="普通内容" * 1500 + "广告尾部"))

        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertTrue(verdict.violated)
        self.assertTrue(service.is_high_confidence(verdict))
        user = service.llm.moderation.await_args.args[1]
        self.assertIn("[quote]", user)
        self.assertIn("分享一下", user)

    async def test_truncated_current_context_cannot_trigger_direct_punishment(self):
        async def review(system, user):
            if "[reply-to-message]" in user:
                return '{"violated":true,"confidence":0.99,"rule_id":7}'
            return '{"violated":false,"confidence":0.99}'

        service = self.service("")
        service.llm.moderation.side_effect = review
        message = NS(text="x" * 8100, reply_to_message=NS(text="广告"))

        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertTrue(verdict.violated)
        self.assertFalse(verdict.conclusive)
        self.assertFalse(service.is_high_confidence(verdict))
        self.assertIn("[current-message context incomplete]", service.llm.moderation.await_args.args[1])

    async def test_truncated_current_context_cannot_be_cached_as_clean(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(text="x" * 8100, quote=NS(text="引用"))

        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertFalse(verdict.violated)
        self.assertFalse(verdict.conclusive)

    async def test_unseen_quote_tail_never_earns_a_clean_cache_entry(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(text="查看引用", quote=NS(text="x" * 40000 + "尾部广告"))

        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertFalse(verdict.violated)
        self.assertFalse(verdict.conclusive)
        self.assertEqual(service.llm.moderation.await_count, 8)
        self.assertNotIn("尾部广告", service.llm.moderation.await_args.args[1])

    async def test_long_reply_context_is_present_when_reviewing_current_message(self):
        async def review(_system, user):
            if "请管理员处理" in user and "[reply-to-message]" not in user:
                return '{"violated":true,"confidence":0.99,"rule_id":7}'
            return '{"violated":false,"confidence":0.99}'

        service = self.service("")
        service.llm.moderation.side_effect = review
        message = NS(text="请管理员处理，禁止此广告", reply_to_message=NS(text="广告" * 2500))
        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))

        self.assertFalse(verdict.violated)
        self.assertTrue(verdict.conclusive)
        first = service.llm.moderation.await_args_list[0].args[1]
        self.assertIn("[current-message]", first)
        self.assertIn("[reply-to-message]", first)
        self.assertIn("请管理员处理", first)

    async def test_incomplete_reference_context_cannot_punish_or_cache_clean(self):
        message = NS(text="请管理员处理，禁止此广告", reply_to_message=NS(text="广告" * 5000))
        evidence = build_moderation_input(message)
        service = self.service('{"violated":true,"confidence":0.99,"rule_id":7}')

        verdict = await service.evaluate(self.session, -100, evidence)

        self.assertTrue(verdict.violated)
        self.assertFalse(verdict.conclusive)
        self.assertFalse(service.is_high_confidence(verdict))
        self.assertTrue(all(
            "[reference context incomplete]" in call.args[1]
            for call in service.llm.moderation.await_args_list
        ))
        service = self.service('{"violated":false,"confidence":0.99}')
        clean = await service.evaluate(self.session, -100, evidence)
        self.assertFalse(clean.conclusive)

    async def test_quote_context_is_present_during_other_reference_review(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(text="请注意", quote=NS(text="这条广告不要相信"),
                     reply_to_message=NS(text="广告内容" * 1200))
        await service.evaluate(self.session, -100, build_moderation_input(message))
        for call in service.llm.moderation.await_args_list:
            self.assertIn("这条广告不要相信", call.args[1])

    async def test_current_message_tail_is_preserved_on_its_earlier_chunks(self):
        service = self.service('{"violated":false,"confidence":0.99}')
        message = NS(text="举例" * 2400 + "以上都是诈骗，请管理员删除",
                     quote=NS(text="被引用的内容"))
        verdict = await service.evaluate(self.session, -100, build_moderation_input(message))
        self.assertTrue(verdict.conclusive)
        for call in service.llm.moderation.await_args_list:
            self.assertIn("以上都是诈骗，请管理员删除", call.args[1])


if __name__ == "__main__":
    unittest.main()
