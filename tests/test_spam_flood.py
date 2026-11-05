"""Tests for repetition-based flood-ad detection."""

from __future__ import annotations

import time
import unittest

from bot.services import spam_flood


def _score(messages, *, window=300, min_messages=4, min_hits=3, threshold=0.85,
           strong=0.85, weak=0.5, gap=4.0, user_id=42):
    spam_flood.reset_state()
    base = time.monotonic()
    for index, text in enumerate(messages):
        spam_flood.observe_message(-100, user_id, text, now=base + index * gap)
    return spam_flood.score_window(
        -100,
        user_id,
        window_seconds=window,
        min_messages=min_messages,
        similarity_min_hits=min_hits,
        similarity_threshold=threshold,
        strong_score=strong,
        weak_score=weak,
    )


class FloodScoringTests(unittest.TestCase):
    def test_repeated_advert_with_link_is_strong(self) -> None:
        verdict = _score([
            "限时优惠 加@xinbi888 领取 https://t.me/abc",
            "限时钜惠 加@xinbi888 速来 https://t.me/abc",
            "限时福利 加@xinbi888 领取 https://t.me/abc",
            "限时优惠 加@xinbi888 领取 https://t.me/abc",
        ])
        self.assertTrue(verdict.is_strong)
        self.assertGreaterEqual(verdict.similar_count, 3)

    def test_repeated_advert_without_link_is_weak(self) -> None:
        verdict = _score(["出二手手机联系我"] * 4)
        self.assertEqual(verdict.tier, "weak")
        self.assertFalse(verdict.is_strong)

    def test_slow_spammer_within_long_window_is_detected(self) -> None:
        verdict = _score(
            ["限时优惠 加@xinbi888 领取 https://t.me/abc"] * 4,
            window=600,
            gap=90,
        )
        self.assertTrue(verdict.is_candidate)

    def test_normal_varied_chat_is_ignored(self) -> None:
        verdict = _score(["今天天气不错", "晚上吃啥", "有没有人打游戏", "明天开会几点"])
        self.assertEqual(verdict.tier, "none")

    def test_group_chain_style_replies_are_ignored(self) -> None:
        verdict = _score(["接龙1 张三", "接龙2 李四", "接龙3 王五", "接龙4 赵六"])
        self.assertEqual(verdict.tier, "none")

    def test_frequent_distinct_messages_are_ignored(self) -> None:
        verdict = _score(["哈哈", "我同意", "+1", "好的", "收到", "明白"], min_messages=5)
        self.assertEqual(verdict.tier, "none")

    def test_below_minimum_messages_is_none(self) -> None:
        verdict = _score(["广告一条"], min_messages=4)
        self.assertEqual(verdict.tier, "none")


class FloodWindowTests(unittest.TestCase):
    def test_messages_outside_window_are_excluded(self) -> None:
        spam_flood.reset_state()
        base = time.monotonic()
        # Four spam messages, but only the last two are within the 300s window.
        for index, text in enumerate(["ad link https://t.me/x"] * 4):
            spam_flood.observe_message(-100, 42, text, now=base + index * 400)
        verdict = spam_flood.score_window(
            -100, 42,
            window_seconds=300, min_messages=4, similarity_min_hits=3,
            similarity_threshold=0.85, strong_score=0.85, weak_score=0.5,
            now=base + 3 * 400,
        )
        self.assertEqual(verdict.tier, "none")
        self.assertLess(verdict.message_count, 4)

    def test_forget_member_clears_window(self) -> None:
        spam_flood.reset_state()
        spam_flood.observe_message(-100, 42, "x")
        spam_flood.forget_member(-100, 42)
        verdict = spam_flood.score_window(
            -100, 42,
            window_seconds=300, min_messages=2, similarity_min_hits=2,
            similarity_threshold=0.85, strong_score=0.85, weak_score=0.5,
        )
        self.assertEqual(verdict.message_count, 0)


if __name__ == "__main__":
    unittest.main()
