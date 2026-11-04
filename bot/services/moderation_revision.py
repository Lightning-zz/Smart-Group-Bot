"""In-process ordering guard for moderation side effects on edited messages.

Telegram can deliver an edit while a previous, slow moderation call is still
waiting on an LLM.  The newer revision must invalidate the older call before
it can delete, warn, ban, or otherwise publish a moderation result.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from bot.utils.logging_setup import get_log_context

_MAX_INACTIVE_KEYS = 4096
_INACTIVE_TTL_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class _Revision:
    event_time: float
    edited: bool
    update_id: int
    fingerprint: str


@dataclass(slots=True)
class _Entry:
    revision: _Revision
    generation: int
    active: int = 0
    last_seen: float = 0.0


@dataclass(slots=True)
class _ScopeState:
    key: tuple[int, int]
    revision: _Revision
    generation: int
    owner: object
    closed: bool = False


_LATEST: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
_CURRENT: ContextVar[_ScopeState | None] = ContextVar(
    "moderation_revision_scope", default=None
)


def _number(value: object) -> float:
    if isinstance(value, datetime):
        value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            return value.timestamp()
        except (ValueError, OSError):
            return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else 0.0
    return 0.0


def _update_id() -> int:
    raw = get_log_context().get("update_id", "-")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


_FINGERPRINT_FIELDS = (
    "text", "caption", "edit_date", "reply_markup", "photo", "video",
    "document", "animation", "audio", "voice", "sticker", "video_note",
    "entities", "caption_entities", "contact", "poll", "location",
    "game", "invoice", "venue", "checklist",
)


def _fingerprint_values(message: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in _FINGERPRINT_FIELDS:
        value = message.get(name) if isinstance(message, dict) else getattr(message, name, None)
        if name not in {"text", "caption", "edit_date"}:
            dump = getattr(value, "model_dump", None)
            if callable(dump):
                try:
                    value = dump(mode="json", exclude_none=True)
                except (TypeError, ValueError):
                    value = repr(value)
            elif value is not None:
                value = repr(value)
        values[name] = value
    return values


def _fingerprint(message: object) -> str:
    values = _fingerprint_values(message)
    # Capture only the immediate referenced content, never nested reply chains.
    for name in ("quote", "reply_to_message", "external_reply"):
        context = getattr(message, name, None)
        values[name] = _fingerprint_values(context) if context is not None else None
    try:
        encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = repr(values)
    return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()[:24]


def _revision(message: object) -> _Revision:
    edit_date = getattr(message, "edit_date", None)
    edited = edit_date is not None
    event_time = _number(edit_date if edited else getattr(message, "date", None))
    return _Revision(event_time, edited, _update_id(), _fingerprint(message))


def _newer(candidate: _Revision, current: _Revision) -> bool:
    if candidate.event_time != current.event_time:
        return candidate.event_time > current.event_time
    if candidate.edited != current.edited:
        return candidate.edited and not current.edited
    if candidate.update_id or current.update_id:
        return candidate.update_id > current.update_id
    # When Telegram/log middleware supplies no update id, distinct payloads
    # arriving at the same timestamp are considered newer by arrival order.
    return candidate.fingerprint != current.fingerprint


def _key(message: object) -> tuple[int, int]:
    chat = getattr(message, "chat", None)
    try:
        return int(getattr(chat, "id", 0) or 0), int(getattr(message, "message_id", 0) or 0)
    except (TypeError, ValueError):
        return 0, 0


def _scope_owner() -> object:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return task if task is not None else threading.get_ident()


def _prune(now: float) -> None:
    for key, entry in tuple(_LATEST.items()):
        if entry.active == 0 and now - entry.last_seen >= _INACTIVE_TTL_SECONDS:
            _LATEST.pop(key, None)
    if len(_LATEST) <= _MAX_INACTIVE_KEYS:
        return
    for key, entry in tuple(_LATEST.items()):
        if len(_LATEST) <= _MAX_INACTIVE_KEYS:
            break
        if entry.active == 0:
            _LATEST.pop(key, None)


@contextmanager
def moderation_revision_scope(message: object) -> Iterator[None]:
    """Register one message revision and invalidate older in-flight scopes."""
    key = _key(message)
    if not key[0] or key[1] <= 0:
        # Synthetic/manual messages cannot safely share an ordering key.
        token = _CURRENT.set(None)
        try:
            yield
        finally:
            _CURRENT.reset(token)
        return
    candidate = _revision(message)
    owner = _scope_owner()
    inherited = _CURRENT.get()
    if (
        inherited is not None and not inherited.closed and inherited.owner == owner
        and inherited.key == key and inherited.revision == candidate
    ):
        yield
        return

    now = time.monotonic()
    entry = _LATEST.get(key)
    if entry is None:
        entry = _Entry(candidate, generation=1, active=0, last_seen=now)
        _LATEST[key] = entry
    elif _newer(candidate, entry.revision):
        entry.revision = candidate
        entry.generation += 1
        entry.last_seen = now
    else:
        entry.last_seen = now
    entry.active += 1
    _LATEST.move_to_end(key)
    _prune(now)
    state = _ScopeState(key, candidate, entry.generation, owner=owner)
    token: Token[_ScopeState | None] = _CURRENT.set(state)
    try:
        yield
    finally:
        state.closed = True
        _CURRENT.reset(token)
        current = _LATEST.get(key)
        if current is not None:
            current.active = max(0, current.active - 1)
            current.last_seen = time.monotonic()
            _prune(current.last_seen)


def is_current_moderation_revision() -> bool:
    """Whether the current scope still represents the latest known revision."""
    state = _CURRENT.get()
    if state is None:
        return True
    entry = _LATEST.get(state.key)
    return bool(
        entry is not None
        and not state.closed
        and entry.generation == state.generation
        and entry.revision == state.revision
    )


def _reset_for_tests() -> None:
    """Reset process state for isolated tests; not part of runtime API."""
    _LATEST.clear()
    _CURRENT.set(None)
