import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch

from bot.services import moderation_revision as revision
from bot.utils.logging_setup import reset_log_context, set_log_context


BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def message(*, message_id=10, chat_id=-100, edited_at=None, text="original"):
    return NS(
        chat=NS(id=chat_id), message_id=message_id, date=BASE_TIME,
        edit_date=edited_at, text=text,
    )


class ModerationRevisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        revision._reset_for_tests()
        self.tokens = set_log_context(update_id="-")

    def tearDown(self):
        revision._reset_for_tests()
        reset_log_context(self.tokens)

    async def test_new_edit_immediately_invalidates_original_scope(self):
        original = message()
        edited = message(edited_at=BASE_TIME + timedelta(seconds=1), text="edited")
        with revision.moderation_revision_scope(original):
            self.assertTrue(revision.is_current_moderation_revision())
            with revision.moderation_revision_scope(edited):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertFalse(revision.is_current_moderation_revision())
        self.assertTrue(revision.is_current_moderation_revision())

    async def test_late_original_cannot_supersede_completed_edit(self):
        with revision.moderation_revision_scope(message(edited_at=BASE_TIME + timedelta(seconds=2))):
            pass
        with revision.moderation_revision_scope(message()):
            self.assertFalse(revision.is_current_moderation_revision())

    async def test_same_second_updates_are_ordered_by_update_id(self):
        first = message(edited_at=BASE_TIME, text="first")
        second = message(edited_at=BASE_TIME, text="second")
        first_token = set_log_context(update_id=100)
        try:
            with revision.moderation_revision_scope(first):
                second_token = set_log_context(update_id=101)
                try:
                    with revision.moderation_revision_scope(second):
                        self.assertTrue(revision.is_current_moderation_revision())
                finally:
                    reset_log_context(second_token)
                self.assertFalse(revision.is_current_moderation_revision())
            with revision.moderation_revision_scope(first):
                self.assertFalse(revision.is_current_moderation_revision())
        finally:
            reset_log_context(first_token)

    async def test_same_task_nested_scope_is_not_registered_twice(self):
        item = message()
        with revision.moderation_revision_scope(item):
            with revision.moderation_revision_scope(item):
                self.assertEqual(revision._LATEST[(-100, 10)].active, 1)
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertTrue(revision.is_current_moderation_revision())
        self.assertEqual(revision._LATEST[(-100, 10)].active, 0)

    async def test_concurrent_newer_task_invalidates_waiting_old_task(self):
        old_started = asyncio.Event()
        edit_finished = asyncio.Event()

        async def old_worker():
            with revision.moderation_revision_scope(message()):
                old_started.set()
                await edit_finished.wait()
                return revision.is_current_moderation_revision()

        async def edit_worker():
            await old_started.wait()
            with revision.moderation_revision_scope(message(edited_at=BASE_TIME + timedelta(seconds=1))):
                self.assertTrue(revision.is_current_moderation_revision())
            edit_finished.set()

        stale, _ = await asyncio.gather(old_worker(), edit_worker())
        self.assertFalse(stale)

    async def test_inherited_child_registers_its_own_active_scope(self):
        item = message()
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def child():
            with revision.moderation_revision_scope(item):
                entered.set()
                await finish.wait()
                self.assertTrue(revision.is_current_moderation_revision())

        with revision.moderation_revision_scope(item):
            task = asyncio.create_task(child())
            await entered.wait()
            self.assertEqual(revision._LATEST[(-100, 10)].active, 2)
        self.assertEqual(revision._LATEST[(-100, 10)].active, 1)
        finish.set()
        await task

    async def test_active_keys_survive_cache_pressure_then_cache_shrinks(self):
        with patch.object(revision, "_MAX_INACTIVE_KEYS", 2):
            with revision.moderation_revision_scope(message(message_id=1)):
                with revision.moderation_revision_scope(message(message_id=2)):
                    with revision.moderation_revision_scope(message(message_id=3)):
                        self.assertEqual(len(revision._LATEST), 3)
                        self.assertEqual(revision._LATEST[(-100, 1)].active, 1)
                    self.assertTrue(revision.is_current_moderation_revision())
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertLessEqual(len(revision._LATEST), 2)

    async def test_unknown_update_id_uses_changed_payload_as_revision(self):
        with revision.moderation_revision_scope(message(edited_at=BASE_TIME, text="first")):
            with revision.moderation_revision_scope(message(edited_at=BASE_TIME, text="second")):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertFalse(revision.is_current_moderation_revision())

    async def test_missing_dates_still_orders_known_update_ids(self):
        original = message()
        original.date = None
        edited = message(text="edited")
        edited.date = None
        first_token = set_log_context(update_id=200)
        try:
            with revision.moderation_revision_scope(original):
                second_token = set_log_context(update_id=201)
                try:
                    with revision.moderation_revision_scope(edited):
                        self.assertTrue(revision.is_current_moderation_revision())
                finally:
                    reset_log_context(second_token)
                self.assertFalse(revision.is_current_moderation_revision())
        finally:
            reset_log_context(first_token)

    async def test_missing_dates_and_update_id_use_arrival_order(self):
        original = message()
        original.date = None
        edited = message(text="edited")
        edited.date = None
        with revision.moderation_revision_scope(original):
            with revision.moderation_revision_scope(edited):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertFalse(revision.is_current_moderation_revision())

    async def test_expired_inactive_entries_pruned_but_active_retained(self):
        with patch.object(revision, "_INACTIVE_TTL_SECONDS", 1.0):
            with patch.object(revision.time, "monotonic", return_value=10.0):
                with revision.moderation_revision_scope(message(message_id=1)):
                    with revision.moderation_revision_scope(message(message_id=2)):
                        pass
                    with patch.object(revision.time, "monotonic", return_value=12.0):
                        with revision.moderation_revision_scope(message(message_id=3)):
                            self.assertIn((-100, 1), revision._LATEST)
                            self.assertNotIn((-100, 2), revision._LATEST)

    async def test_entity_only_change_is_a_revision_without_update_id(self):
        first = message(edited_at=BASE_TIME)
        first.entities = [NS(url="https://old.example")]
        second = message(edited_at=BASE_TIME)
        second.entities = [NS(url="https://new.example")]
        with revision.moderation_revision_scope(first):
            with revision.moderation_revision_scope(second):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertFalse(revision.is_current_moderation_revision())

    async def test_quote_only_change_is_a_revision_without_update_id(self):
        first = message(edited_at=BASE_TIME)
        first.quote = NS(text="old quote")
        second = message(edited_at=BASE_TIME)
        second.quote = NS(text="new quote")
        with revision.moderation_revision_scope(first):
            with revision.moderation_revision_scope(second):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertFalse(revision.is_current_moderation_revision())

    async def test_reference_fingerprint_does_not_follow_nested_reply_chain(self):
        first = message(edited_at=BASE_TIME)
        first.reply_to_message = NS(text="immediate", reply_to_message=NS(text="old nested"))
        second = message(edited_at=BASE_TIME)
        second.reply_to_message = NS(text="immediate", reply_to_message=NS(text="new nested"))
        self.assertEqual(revision._fingerprint(first), revision._fingerprint(second))

    async def test_reference_fingerprint_handles_circular_reply_context(self):
        item = message(edited_at=BASE_TIME)
        item.reply_to_message = item
        item.external_reply = item
        with revision.moderation_revision_scope(item):
            self.assertTrue(revision.is_current_moderation_revision())

    async def test_structured_payload_changes_update_revision_fingerprint(self):
        for field in ("game", "invoice", "venue", "checklist"):
            with self.subTest(field=field):
                first = message(edited_at=BASE_TIME)
                setattr(first, field, NS(title="old content"))
                second = message(edited_at=BASE_TIME)
                setattr(second, field, NS(title="advertising content"))
                self.assertNotEqual(revision._fingerprint(first), revision._fingerprint(second))
                first_reply = message(edited_at=BASE_TIME)
                first_reply.reply_to_message = first
                second_reply = message(edited_at=BASE_TIME)
                second_reply.reply_to_message = second
                self.assertNotEqual(
                    revision._fingerprint(first_reply), revision._fingerprint(second_reply),
                )

    async def test_unkeyed_synthetic_message_does_not_contaminate_cache(self):
        with revision.moderation_revision_scope(NS()):
            self.assertTrue(revision.is_current_moderation_revision())
        self.assertEqual(len(revision._LATEST), 0)

    async def test_different_chat_and_message_scopes_are_independent(self):
        with revision.moderation_revision_scope(message()):
            with revision.moderation_revision_scope(message(chat_id=-200)):
                self.assertTrue(revision.is_current_moderation_revision())
            with revision.moderation_revision_scope(message(message_id=11)):
                self.assertTrue(revision.is_current_moderation_revision())
            self.assertTrue(revision.is_current_moderation_revision())


if __name__ == "__main__":
    unittest.main()
