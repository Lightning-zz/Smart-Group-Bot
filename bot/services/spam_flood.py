"""Repetition-based flood-ad detection for content that passes group rules.

Group moderation rules judge one message's text against keyword/regex/semantic
policy.  A spammer that posts a *slightly reworded* advert every few seconds or
minutes slips past every rule while still flooding the group.  This module adds
a behaviour layer on top of the rule engine: it keeps a short, bounded,
in-process window of each member's recent messages and scores the *pattern* of
posting (burst rhythm + self-similarity + shared links) rather than the text
alone.

Design constraints that shape the implementation:

* The hot path is every inbound group message, so per-message work is O(1)
  amortised: the window is a fixed-size ring and similarity uses 64-bit
  SimHash + character shingles, never an O(N^2) pairwise scan of history.
* State is intentionally in-process and bounded (like ``recent_messages``):
  this is a fast behavioural heuristic, not a durable record.  A restart loses
  the window, which only costs a few minutes of recall.
* Scoring is tiered so callers can auto-act on strong evidence and merely warn
  admins on weak evidence, keeping false positives away from destructive
  actions.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

_MAX_TRACKED_MEMBERS = 8192
_MAX_WINDOW_MESSAGES = 24
# Keep distinct windows from interacting across a very long idle gap.
_MAX_WINDOW_AGE_SECONDS = 6 * 3600

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|t\.me/[^\s<>\"']+", re.IGNORECASE)
# Telegram usernames / channel invites are common advert payloads.
_HANDLE_RE = re.compile(r"@[A-Za-z][A-Za-z0-9_]{3,31}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-]{7,}\d)(?!\d)")

_TIER_NONE = "none"
_TIER_WEAK = "weak"
_TIER_STRONG = "strong"


@dataclass(slots=True)
class FloodSample:
    """One observed message reduced to the few features the scorer needs."""

    at: float
    text: str
    simhash: int
    shingles: frozenset[int]
    link_key: str
    link_count: int


@dataclass(slots=True)
class FloodVerdict:
    """Result of scoring a member's recent window."""

    tier: str = _TIER_NONE
    score: float = 0.0
    message_count: int = 0
    similar_count: int = 0
    link_count: int = 0
    window_seconds: float = 0.0
    reason: str = ""

    @property
    def is_strong(self) -> bool:
        return self.tier == _TIER_STRONG

    @property
    def is_candidate(self) -> bool:
        return self.tier in {_TIER_WEAK, _TIER_STRONG}


_WINDOWS: dict[tuple[int, int], deque[FloodSample]] = {}


def _prune_members(now: float) -> None:
    stale_floor = now - _MAX_WINDOW_AGE_SECONDS
    for key in [
        key
        for key, bucket in _WINDOWS.items()
        if not bucket or bucket[-1].at < stale_floor
    ]:
        _WINDOWS.pop(key, None)
    while len(_WINDOWS) >= _MAX_TRACKED_MEMBERS:
        _WINDOWS.pop(next(iter(_WINDOWS)), None)


def reset_state() -> None:
    """Clear all tracked windows (used by tests and on config reload)."""
    _WINDOWS.clear()


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    # Strip URLs/handles so reworded adverts with rotating links still compare
    # on their body copy; links are scored separately via the link fingerprint.
    lowered = _URL_RE.sub(" ", lowered)
    lowered = _HANDLE_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _char_shingles(text: str, size: int = 4) -> frozenset[int]:
    normalized = _normalize_text(text)
    if len(normalized) < size:
        return frozenset({hash(normalized)} if normalized else set())
    return frozenset(
        hash(normalized[i : i + size]) for i in range(len(normalized) - size + 1)
    )


def _simhash(shingles: Iterable[int]) -> int:
    bits = [0] * 64
    for value in shingles:
        for bit in range(64):
            if value & (1 << bit):
                bits[bit] += 1
            else:
                bits[bit] -= 1
    result = 0
    for bit, weight in enumerate(bits):
        if weight > 0:
            result |= 1 << bit
    return result


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _link_key(text: str) -> tuple[str, int]:
    """Return a canonical fingerprint for the first link plus total count."""
    matches = _URL_RE.findall(text or "")
    urls = list(matches)
    urls.extend(_HANDLE_RE.findall(text or ""))
    if not urls:
        return "", 0

    def canon(url: str) -> str:
        value = url.strip().lower()
        value = re.sub(r"^https?://", "", value)
        value = value.rstrip("/")
        # Collapse an advert's rotating tracking/query noise to host+path.
        value = value.split("?", 1)[0].split("#", 1)[0]
        return value

    return canon(urls[0]), len(urls)


def _features(text: str) -> FloodSample:
    shingles = _char_shingles(text)
    link_key, link_count = _link_key(text)
    return FloodSample(
        at=time.monotonic(),
        text=(text or "")[:500],
        simhash=_simhash(shingles),
        shingles=shingles,
        link_key=link_key,
        link_count=link_count,
    )


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def observe_message(
    group_id: int,
    user_id: int,
    text: str,
    *,
    now: float | None = None,
) -> None:
    """Record one inbound message into the member's sliding window."""
    key = (int(group_id), int(user_id))
    current = time.monotonic() if now is None else now
    bucket = _WINDOWS.get(key)
    if bucket is None:
        if len(_WINDOWS) >= _MAX_TRACKED_MEMBERS:
            _prune_members(current)
        bucket = deque(maxlen=_MAX_WINDOW_MESSAGES)
        _WINDOWS[key] = bucket
    sample = _features(text)
    sample.at = current
    bucket.append(sample)


def _window_samples(
    group_id: int,
    user_id: int,
    *,
    window_seconds: float,
    now: float,
) -> list[FloodSample]:
    bucket = _WINDOWS.get((int(group_id), int(user_id)))
    if not bucket:
        return []
    floor = now - window_seconds
    return [sample for sample in bucket if sample.at >= floor]


def _near_duplicate(a: FloodSample, b: FloodSample, threshold: float) -> bool:
    """Two messages are near-duplicates when their shingles overlap enough.

    Short CJK adverts produce few character shingles, so a plain Jaccard floor
    is too brittle once a word is swapped; also accept a high *containment*
    (most of the smaller message appears in the larger) and a close SimHash.
    """
    jaccard = _jaccard(a.shingles, b.shingles)
    if jaccard >= threshold:
        return True
    if a.shingles and b.shingles:
        smaller, larger = sorted((a.shingles, b.shingles), key=len)
        containment = len(smaller & larger) / len(smaller)
        # Reworded adverts keep most of the body; a swapped word only nudges
        # containment. Require a high floor so unrelated replies never match.
        if containment >= max(0.75, threshold):
            return True
    # Identical link payload alone (same advert channel/invite) is strong.
    if a.link_key and a.link_key == b.link_key:
        return True
    return _hamming(a.simhash, b.simhash) <= 4 and jaccard >= 0.5


def _largest_similar_cluster(
    samples: list[FloodSample], threshold: float
) -> int:
    """Largest group of mutually near-duplicate messages (union-find).

    Union-find keeps this O(N^2) only over the bounded window (<=24), and
    unioning transitive matches such as A~B and B~C correctly counts a
    reworded burst even when no single message matches all the others.
    """
    count = len(samples)
    if count < 2:
        return count
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(count):
        for j in range(i + 1, count):
            if _near_duplicate(samples[i], samples[j], threshold):
                union(i, j)

    sizes: dict[int, int] = {}
    for i in range(count):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return max(sizes.values(), default=0)


def score_window(
    group_id: int,
    user_id: int,
    *,
    window_seconds: float,
    min_messages: int,
    similarity_min_hits: int,
    similarity_threshold: float,
    strong_score: float,
    weak_score: float,
    now: float | None = None,
) -> FloodVerdict:
    """Score the member's recent window and classify it into a tier."""
    current = time.monotonic() if now is None else now
    samples = _window_samples(
        group_id, user_id, window_seconds=window_seconds, now=current
    )
    verdict = FloodVerdict(message_count=len(samples))
    if len(samples) < max(2, int(min_messages)):
        return verdict

    verdict.window_seconds = round(samples[-1].at - samples[0].at, 3)

    # --- burst rhythm: how tight are the gaps compared with the window? ---
    gaps = [samples[i].at - samples[i - 1].at for i in range(1, len(samples))]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    burst = 0.0
    if mean_gap > 0:
        burst = min(1.0, (len(samples) - 1) / max(1.0, window_seconds / max(mean_gap, 1e-6)))
    # Regular, low-variance spacing is characteristic of automated posting.
    regularity = 0.0
    if len(gaps) >= 3 and mean_gap > 0:
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        cv = (variance ** 0.5) / mean_gap
        regularity = max(0.0, min(1.0, 1.0 - cv))

    # --- self-similarity: size of the largest near-duplicate cluster ---
    similar = _largest_similar_cluster(samples, similarity_threshold)
    verdict.similar_count = similar

    # --- link aggregation: repeated identical/rotated links ---
    link_counts: dict[str, int] = {}
    for sample in samples:
        if sample.link_key:
            link_counts[sample.link_key] = link_counts.get(sample.link_key, 0) + 1
    top_link = max(link_counts.values(), default=0)
    total_links = sum(sample.link_count for sample in samples)
    verdict.link_count = total_links

    # --- weighted score ---
    freq_score = min(1.0, len(samples) / max(2.0, float(min_messages) * 1.5))
    sim_hits = max(0, similar)
    sim_score = (
        min(1.0, sim_hits / max(2.0, float(similarity_min_hits)))
        if sim_hits >= 2
        else 0.0
    )
    link_score = min(1.0, top_link / max(2.0, float(min_messages))) if top_link else 0.0

    score = (
        0.30 * freq_score
        + 0.40 * sim_score
        + 0.20 * link_score
        + 0.10 * regularity
    )
    # A repeated link that also clusters in a burst is far more decisive.
    if top_link >= 2 and sim_hits >= 2 and (freq_score >= 0.6 or burst >= 0.6):
        score = min(1.0, score + 0.15)

    verdict.score = round(score, 4)

    strong = sim_hits >= max(2, int(similarity_min_hits)) and top_link >= 2
    if score >= strong_score and strong:
        verdict.tier = _TIER_STRONG
        verdict.reason = (
            f"{len(samples)} 条消息中 {sim_hits} 条高度相似"
            + (f"且重复链接 {top_link} 次" if top_link else "")
        )
    elif score >= weak_score and sim_hits >= 2:
        verdict.tier = _TIER_WEAK
        verdict.reason = f"{len(samples)} 条消息中 {sim_hits} 条相似"
    elif score >= weak_score:
        verdict.tier = _TIER_WEAK
        verdict.reason = f"窗口内 {len(samples)} 条消息发送频率异常"
    return verdict


def forget_member(group_id: int, user_id: int) -> None:
    _WINDOWS.pop((int(group_id), int(user_id)), None)


def _tier_constants() -> dict[str, str]:
    return {"none": _TIER_NONE, "weak": _TIER_WEAK, "strong": _TIER_STRONG}


__all__ = [
    "FloodVerdict",
    "observe_message",
    "score_window",
    "forget_member",
    "reset_state",
]
