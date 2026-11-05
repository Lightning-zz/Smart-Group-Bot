"""Private admin notification for messages that triggered moderation rules.

When ``moderation.notify_admin_on_violation`` is enabled, every recorded
violation (ban / delete / warn, including bot-screening hits) is pushed to the
group's human administrators as a private message describing the sender, the
offending message and the rule that fired.

The notification is best-effort and must never block or fail the moderation
pipeline: every send is wrapped and failures are only logged.  Duplicate delivery
across durable retries is avoided by the same per-violation ``notice_sent_at``
style guard the group-visible notice uses — callers pass the freshly recorded
``Violation`` and the helper claims it once.
"""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.models import Admin, GroupMember
from bot.services.message_templates import card_field, render_summary_notice

log = logging.getLogger(__name__)

# Telegram URL length budget for a t.me profile link; usernames are far shorter.
_MAX_DISPLAY_UNITS = 128
_MAX_MESSAGE_UNITS = 1200
_ADMIN_NOTIFY_CONCURRENCY = 4

_ACTION_LABELS = {
    "warn": "已警告",
    "delete": "已删除违规消息",
    "ban": "已删除违规消息并封禁用户",
    "ban_applied": "已封禁用户",
    "ban_warning": "已警告（未达封禁阈值）",
    "bot_ban": "已封禁 bot",
    "bot_delete": "已删除违规消息",
}


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _sender_label(user_id: int, display_name: str, username: str) -> str:
    """Render an HTML-safe sender identity line with a clickable profile link."""
    name = _truncate(display_name, _MAX_DISPLAY_UNITS)
    handle = str(username or "").strip().lstrip("@")
    if handle:
        mention = f"@{html.escape(handle)}"
        link = f"https://t.me/{html.escape(handle)}"
        return f'<a href="{link}">{mention}</a>'
    if name:
        # tg://user deep link opens the profile without needing a username.
        return f'<a href="tg://user?id={int(user_id)}">{html.escape(name)}</a>'
    return f"<code>{int(user_id)}</code>"


def _rule_reference(rule: Any) -> str:
    if rule is None:
        return "未定位具体规则（AI 语义判定）"
    rule_type = str(getattr(rule, "rule_type", "") or "").strip().lower()
    type_labels = {"keyword": "关键词", "regex": "正则", "llm": "语义"}
    label = type_labels.get(rule_type, rule_type or "未知")
    rule_id = getattr(rule, "id", None)
    preview = _truncate(getattr(rule, "pattern", "") or "", 80)
    prefix = f"#{int(rule_id)}" if rule_id else "规则"
    if preview:
        return f"{prefix}（{html.escape(label)}）{html.escape(preview)}"
    return f"{prefix}（{html.escape(label)}）"


def build_violation_admin_notice(
    *,
    sender_id: int,
    display_name: str,
    username: str,
    message_text: str,
    rule: Any,
    action_taken: str,
    reason: str = "",
    group_title: str = "",
    group_id: int = 0,
) -> str:
    """Render the private admin alert as Telegram HTML."""
    action_label = _ACTION_LABELS.get(
        str(action_taken or "").strip().lower(), str(action_taken or "已处理")
    )
    message_body = _truncate(message_text, _MAX_MESSAGE_UNITS)
    message_value = (
        f"<blockquote>{html.escape(message_body)}</blockquote>"
        if message_body
        else "<i>（无文本内容）</i>"
    )
    summary = [card_field("发送人", _sender_label(sender_id, display_name, username))]
    if group_title:
        summary.append(card_field("群组", html.escape(_truncate(group_title, _MAX_DISPLAY_UNITS))))
    summary.append(card_field("处理结果", html.escape(action_label)))

    details = [
        card_field("触发规则", _rule_reference(rule)),
        card_field("消息内容", message_value),
    ]
    clean_reason = str(reason or "").strip()
    if clean_reason:
        details.append(card_field("判定原因", html.escape(_truncate(clean_reason, 400))))
    return render_summary_notice("违规通知", summary, details=details)


async def _resolve_group_admin_ids(
    bot: Bot,
    session: AsyncSession,
    group_id: int,
) -> list[int]:
    """Human admin user ids of ``group_id``, excluding the bot itself.

    Live Telegram administrators are the source of truth; when the lookup fails
    the local authorized-admin table is used instead.
    """
    try:
        me_id = int((await bot.me()).id)
    except Exception:
        me_id = 0

    ids: list[int] = []
    try:
        for member in await bot.get_chat_administrators(int(group_id)) or []:
            user = getattr(member, "user", None)
            if user is None or getattr(user, "is_bot", False):
                continue
            user_id = int(user.id)
            if user_id == me_id:
                continue
            ids.append(user_id)
        return ids
    except Exception:
        log.warning(
            "violation notice: get_chat_administrators failed | group=%s",
            group_id,
            exc_info=True,
        )
    try:
        rows = (
            await session.execute(
                select(Admin.user_id).where(Admin.group_id == int(group_id))
            )
        ).scalars().all()
        return [int(user_id) for user_id in rows if int(user_id) != me_id]
    except Exception:
        log.warning(
            "violation notice: admin table fallback failed | group=%s",
            group_id,
            exc_info=True,
        )
        return []


async def notify_admins_of_violation(
    bot: Bot,
    session: AsyncSession,
    *,
    group_id: int,
    sender_id: int,
    message_text: str,
    rule: Any,
    action_taken: str,
    reason: str = "",
    display_name: str = "",
    username: str = "",
    group_title: str = "",
) -> int:
    """Private-message every human admin in ``group_id`` about one violation.

    Returns the number of successfully delivered messages.  Never raises.
    """
    try:
        admin_ids = await _resolve_group_admin_ids(bot, session, group_id)
    except Exception:
        log.warning(
            "violation notice: admin resolution failed | group=%s", group_id, exc_info=True
        )
        return 0
    if not admin_ids:
        return 0

    per_admin_text = build_violation_admin_notice(
        sender_id=int(sender_id),
        display_name=display_name,
        username=username,
        message_text=message_text,
        rule=rule,
        action_taken=action_taken,
        reason=reason,
        group_title=group_title,
        group_id=group_id,
    )

    delivered = 0
    semaphore = asyncio.Semaphore(_ADMIN_NOTIFY_CONCURRENCY)

    async def _send(admin_id: int) -> bool:
        async with semaphore:
            try:
                await bot.send_message(
                    chat_id=admin_id,
                    text=per_admin_text,
                    disable_web_page_preview=True,
                )
                return True
            except Exception as exc:
                # A blocked bot / never-started chat is expected and not fatal.
                log.info(
                    "violation notice: admin send failed | group=%s admin=%s error=%s",
                    group_id,
                    admin_id,
                    type(exc).__name__,
                )
                return False

    results = await asyncio.gather(
        *(_send(admin_id) for admin_id in set(admin_ids)),
        return_exceptions=True,
    )
    delivered = sum(1 for item in results if item is True)
    return delivered


async def lookup_sender_identity(
    session: AsyncSession,
    group_id: int,
    user_id: int,
) -> tuple[str, str]:
    """Best-effort ``(display_name, username)`` from the local roster."""
    try:
        row = await session.scalar(
            select(GroupMember.full_name, GroupMember.username).where(
                GroupMember.group_id == int(group_id),
                GroupMember.user_id == int(user_id),
            )
        )
    except Exception:
        return "", ""
    if not row:
        return "", ""
    full_name, username = row
    return str(full_name or ""), str(username or "").lstrip("@")


def build_flood_admin_notice(
    *,
    sender_id: int,
    display_name: str,
    username: str,
    message_text: str,
    tier: str,
    score: float,
    message_count: int,
    similar_count: int,
    link_count: int,
    group_title: str = "",
) -> str:
    """Render a private admin alert for a suspected flood-ad behaviour hit."""
    tier_label = "疑似刷屏广告（强）" if str(tier) == "strong" else "疑似刷屏广告（弱）"
    message_body = _truncate(message_text, _MAX_MESSAGE_UNITS)
    message_value = (
        f"<blockquote>{html.escape(message_body)}</blockquote>"
        if message_body
        else "<i>（无文本内容）</i>"
    )
    summary = [card_field("发送人", _sender_label(sender_id, display_name, username))]
    if group_title:
        summary.append(card_field("群组", html.escape(_truncate(group_title, _MAX_DISPLAY_UNITS))))
    summary.append(card_field("判定", html.escape(tier_label)))

    details = [
        card_field(
            "行为信号",
            f"窗口内 <code>{int(message_count)}</code> 条 · "
            f"相似 <code>{int(similar_count)}</code> 条 · "
            f"链接 <code>{int(link_count)}</code> 个",
        ),
        card_field("综合评分", f"<code>{float(score):.2f}</code>"),
        card_field("最近消息", message_value),
    ]
    return render_summary_notice("刷屏广告告警", summary, details=details)


__all__ = [
    "build_violation_admin_notice",
    "build_flood_admin_notice",
    "notify_admins_of_violation",
    "lookup_sender_identity",
]
