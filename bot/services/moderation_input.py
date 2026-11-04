"""Build attributed moderation evidence from a message and direct references."""
from __future__ import annotations

from collections.abc import Mapping


class ModerationInput(str):
    """Rendered evidence plus the values deterministic rules should inspect.

    Section labels are useful to the semantic model but must not themselves
    satisfy a group's keyword/regex rule (for example a rule named ``url``).
    """

    deterministic_parts: tuple[str, ...]
    semantic_sections: tuple[tuple[str, str], ...]
    current_message_context: str

    def __new__(
        cls, rendered: str, deterministic_parts: tuple[str, ...],
        semantic_sections: tuple[tuple[str, str], ...] = (),
    ) -> "ModerationInput":
        value = super().__new__(cls, rendered)
        value.deterministic_parts = deterministic_parts
        value.semantic_sections = semantic_sections
        value.current_message_context = next(
            (text for name, text in semantic_sections if name == "current-message"), "",
        )
        return value


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _text(value: object) -> str:
    # Telegram fields are strings.  Do not accidentally stringify model dumps,
    # arbitrary objects or test mocks into evidence.
    return value if isinstance(value, str) else ""


def _default_body(message: object) -> str:
    return _text(_field(message, "text")) or _text(_field(message, "caption"))


def _collect_evidence(
    message: object,
    *,
    text: str | None = None,
    msg_type: str | None = None,
    vision_text: str = "",
) -> tuple[list[str], list[str]]:
    """Collect one object only; deliberately do not follow its own references."""
    # extract_message_text adds presentation labels such as [image], name:
    # and user_id:. Prefer the original fields so local rules cannot match
    # those program-generated labels or contact identity metadata.
    body = _default_body(message)
    has_structured_body = (
        msg_type in {"contact", "document"}
        and _field(message, msg_type) is not None
    )
    accepts_extracted_text = msg_type is None or (
        msg_type == "text" or "caption" in msg_type or msg_type in {"contact", "document"}
    )
    if not body and not has_structured_body and text is not None and accepts_extracted_text:
        body = text
    parts = [body] if body else []
    deterministic_parts = [body] if body else []
    if vision_text:
        parts.append(f"[image-vision]\n{vision_text}")
        deterministic_parts.append(vision_text)

    entity_sources = [(message, ("entities", "caption_entities"))]
    for outer, names in (
        ("document", ("file_name",)),
        ("contact", ("first_name", "last_name", "phone_number", "vcard")),
        ("poll", ("question", "explanation")), ("game", ("title", "description", "text")),
        ("invoice", ("title", "description")), ("venue", ("title", "address")),
        ("audio", ("title", "performer")), ("checklist", ("title",)),
    ):
        item = _field(message, outer)
        entity_sources.append((item, ("question_entities", "explanation_entities", "title_entities", "text_entities")))
        for name in names:
            value = _text(_field(item, name))
            if value:
                parts.append(f"[{outer} {name}]\n{value}")
                deterministic_parts.append(value)
        if outer in {"poll", "checklist"}:
            entries = _field(item, "options" if outer == "poll" else "tasks")
            if isinstance(entries, (list, tuple)):
                for entry in entries:
                    entity_sources.append((entry, ("text_entities",)))
                    value = _text(_field(entry, "text"))
                    if value:
                        parts.append(f"[{outer} item]\n{value}")
                        deterministic_parts.append(value)

    seen_urls: set[str] = set()
    for source, fields in entity_sources:
        for entity_field in fields:
            entities = _field(source, entity_field)
            if not isinstance(entities, (list, tuple)):
                continue
            for entity in entities:
                # Visible URLs already occur in text; hidden targets do not.
                url = _text(_field(entity, "url"))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    parts.append(f"[hidden-link url]\n{url}")
                    deterministic_parts.append(url)

    rows = _field(_field(message, "reply_markup"), "inline_keyboard")
    if isinstance(rows, (list, tuple)):
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, (list, tuple)):
                continue
            for column_index, button in enumerate(row, start=1):
                prefix = f"button {row_index}.{column_index}"
                for name in (
                    "text", "url", "callback_data", "switch_inline_query",
                    "switch_inline_query_current_chat",
                ):
                    value = _text(_field(button, name))
                    if value:
                        parts.append(f"[{prefix} {name}]\n{value}")
                        deterministic_parts.append(value)
                for outer, inner in (
                    ("web_app", "url"), ("login_url", "url"),
                    ("copy_text", "text"), ("switch_inline_query_chosen_chat", "query"),
                ):
                    value = _text(_field(_field(button, outer), inner))
                    if value:
                        parts.append(f"[{prefix} {outer}.{inner}]\n{value}")
                        deterministic_parts.append(value)
    return parts, deterministic_parts


def build_moderation_input(
    message: object,
    *,
    text: str | None = None,
    msg_type: str | None = None,
    vision_text: str = "",
    reply_vision_text: str = "",
    external_reply_vision_text: str = "",
) -> str:
    """Collect current content and available direct references without truncation.

    ``text``/``msg_type`` accept the existing current-message text extractor's
    result. Placeholder-only media is excluded, but attached content is not.
    Referenced evidence is included in both deterministic and semantic rules;
    source labels allow semantic assessment of how the sender uses that content
    rather than assuming either guilt or an automatic reply/quote exemption.
    Only Telegram-provided content is read; unavailable external text is not
    inferred from origin metadata, and nested reply chains are never followed.
    """
    current, deterministic_parts = _collect_evidence(
        message, text=text, msg_type=msg_type, vision_text=vision_text,
    )
    sections = [("current-message", "\n".join(current))] if current else []
    for field, label, vision in (
        ("quote", "quote", ""),
        ("reply_to_message", "reply-to-message", reply_vision_text),
        ("external_reply", "external-reply", external_reply_vision_text),
    ):
        reference = _field(message, field)
        if reference is None and not vision:
            continue
        parts, values = _collect_evidence(reference, vision_text=vision)
        if parts:
            sections.append((label, "\n".join(parts)))
            deterministic_parts.extend(values)
    if len(sections) == 1 and sections[0][0] == "current-message":
        rendered = sections[0][1]
    else:
        rendered = "\n".join(f"[{label}]\n{content}" for label, content in sections)
    return ModerationInput(rendered, tuple(deterministic_parts), tuple(sections))


def has_screenable_moderation_content(
    message: object, *, vision_text: str = ""
) -> bool:
    return bool(build_moderation_input(message, vision_text=vision_text))
