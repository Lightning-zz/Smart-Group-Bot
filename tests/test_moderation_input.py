import unittest
from types import SimpleNamespace as NS

from bot.services.moderation_input import build_moderation_input


class ModerationInputTests(unittest.TestCase):
    def test_keyboard_and_hidden_links_are_included(self) -> None:
        message = NS(
            text="普通文案",
            caption=None,
            entities=[NS(type="text_link", url="https://hidden.example/ad")],
            caption_entities=[],
            reply_markup=NS(
                inline_keyboard=[
                    [
                        NS(
                            text="立即领取",
                            url="https://button.example/offer",
                            callback_data="offer:广告",
                            web_app=NS(url="https://web.example"),
                            login_url=NS(url="https://login.example"),
                            copy_text=NS(text="复制广告码"),
                            switch_inline_query="广告查询",
                            switch_inline_query_current_chat="广告查询2",
                        )
                    ]
                ]
            ),
        )

        result = build_moderation_input(message, text="普通文案", msg_type="text")

        for value in (
            "普通文案",
            "https://hidden.example/ad",
            "立即领取",
            "https://button.example/offer",
            "offer:广告",
            "https://web.example",
            "https://login.example",
            "复制广告码",
            "广告查询",
            "广告查询2",
        ):
            self.assertIn(value, result)

    def test_keyboard_makes_placeholder_media_screenable(self) -> None:
        message = NS(
            text=None,
            caption=None,
            entities=[],
            caption_entities=[],
            reply_markup=NS(inline_keyboard=[[NS(text="打开广告", url="https://x.example")]]),
        )

        result = build_moderation_input(message, text="[image]", msg_type="photo")

        self.assertIn("打开广告", result)
        self.assertNotIn("[image]", result)
        self.assertIsInstance(result, str)
        self.assertEqual(result.deterministic_parts, ("打开广告", "https://x.example"))

    def test_reply_and_quote_are_attributed_and_screened(self) -> None:
        message = NS(
            text="正常回复",
            entities=[],
            caption_entities=[],
            reply_markup=None,
            reply_to_message=NS(text="原消息广告", entities=[NS(url="https://reply.example")], reply_markup=NS(inline_keyboard=[[NS(text="广告按钮")]])),
            quote=NS(text="引用广告", entities=[NS(url="https://quote.example")]),
            external_reply=NS(text="外部广告"),
        )

        result = build_moderation_input(message, text="正常回复", msg_type="text")

        for value in ("[current-message]", "[reply-to-message]", "[quote]", "[external-reply]",
                      "原消息广告", "广告按钮", "引用广告", "外部广告",
                      "https://reply.example", "https://quote.example"):
            self.assertIn(value, result)
        self.assertIn("https://quote.example", result.deterministic_parts)
        self.assertNotIn("[quote]", result.deterministic_parts)
        self.assertEqual(result.current_message_context, "正常回复")

    def test_external_structured_content_and_direct_reference_only(self) -> None:
        nested = NS(text="不应递归")
        reply = NS(caption="图片广告", caption_entities=[NS(url="https://caption.example")], reply_to_message=nested)
        message = NS(reply_to_message=reply, external_reply=NS(
            document=NS(file_name="广告.pdf"),
            poll=NS(question="广告问题", options=[NS(text="广告选项")]),
        ))
        nested.reply_to_message = message
        result = build_moderation_input(message, reply_vision_text="回复图广告", external_reply_vision_text="外部图广告")
        for value in ("图片广告", "https://caption.example", "广告.pdf", "广告问题", "广告选项", "回复图广告", "外部图广告"):
            self.assertIn(value, result.deterministic_parts)
        self.assertNotIn("不应递归", result)
        self.assertIn("回复图广告", dict(result.semantic_sections)["reply-to-message"])
        self.assertIn("外部图广告", dict(result.semantic_sections)["external-reply"])

    def test_long_quote_and_reference_only_are_screenable(self) -> None:
        quote = "普通" * 5000 + "尾部广告"
        result = build_moderation_input(NS(quote=NS(text=quote)), text="[image]", msg_type="photo")
        self.assertIn("尾部广告", result)
        self.assertIn(quote, result.deterministic_parts)

    def test_plain_command_is_not_prefixed_or_changed(self) -> None:
        self.assertEqual(
            build_moderation_input(NS(text="/spam@guestbot")),
            "/spam@guestbot",
        )

    def test_caption_entity_links_and_chosen_chat_queries(self) -> None:
        message = {
            "caption": "图片说明",
            "caption_entities": [{"type": "text_link", "url": "https://caption.example/ad"}],
            "reply_markup": {"inline_keyboard": [[{
                "text": "选群", "switch_inline_query_chosen_chat": {"query": "广告频道"},
            }]]},
        }

        result = build_moderation_input(message)

        self.assertIn("图片说明", result)
        self.assertIn("https://caption.example/ad", result)
        self.assertIn("广告频道", result)

    def test_placeholder_without_keyboard_stays_empty(self) -> None:
        for kind in ("photo", "voice", "video", "video_note", "audio", "sticker"):
            with self.subTest(kind=kind):
                self.assertEqual(
                    build_moderation_input(NS(), text=f"[{kind}]", msg_type=kind),
                    "",
                )

    def test_vision_filename_and_contact_are_evidence(self) -> None:
        self.assertIn("视觉广告", build_moderation_input(NS(), text="[image]", msg_type="photo", vision_text="视觉广告"))
        self.assertIn("广告.pdf", build_moderation_input(NS(document=NS(file_name="广告.pdf"))))
        self.assertIn("123456", build_moderation_input(NS(contact=NS(first_name="联系人", phone_number="123456"))))

    def test_extractor_wrappers_do_not_enter_deterministic_fields(self) -> None:
        cases = (
            (NS(caption="正常内容"), "[image]\n正常内容", "photo_caption", ("正常内容",)),
            (NS(document=NS(file_name="手册.pdf"), caption="正常内容"),
             "[document: 手册.pdf]\n正常内容", "document_caption", ("正常内容", "手册.pdf")),
            (NS(document=NS(file_name="手册.pdf")), "[document: 手册.pdf]", "document", ("手册.pdf",)),
            (NS(contact=NS(first_name="张三", phone_number="123456", user_id=888)),
             "[contact]\nname: 张三\nphone: 123456\nuser_id: 888", "contact", ("张三", "123456")),
        )
        for message, text, msg_type, values in cases:
            with self.subTest(msg_type=msg_type):
                result = build_moderation_input(message, text=text, msg_type=msg_type)
                self.assertEqual(result.deterministic_parts, values)

    def test_extracted_text_fallback_remains_for_missing_raw_fields(self) -> None:
        for kind in (None, "text", "photo_caption", "document", "contact"):
            with self.subTest(kind=kind):
                result = build_moderation_input(NS(), text="调用者提供的内容", msg_type=kind)
                self.assertEqual(result.deterministic_parts, ("调用者提供的内容",))

    def test_long_body_does_not_drop_tail_buttons_or_links(self) -> None:
        message = NS(text="正常" * 2048, reply_markup=NS(inline_keyboard=[[NS(text="尾部广告")]]))

        result = build_moderation_input(message)

        self.assertIn("尾部广告", result)
        self.assertGreater(len(result), 4096)


if __name__ == "__main__":
    unittest.main()
