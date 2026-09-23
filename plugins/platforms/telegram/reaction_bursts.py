"""Per-chat inbound burst tracker for Telegram agent reactions.

Consecutive bubbles from the same chat whose gap is within ``window_s`` are one
burst. The burst stays open until ``window_s`` has elapsed since the last
bubble. A reaction aimed at a non-final bubble of an open burst must not fire;
it is held and applied to the final bubble once the burst closes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional


# Short enough to match a typing burst, long enough that two bubbles a second
# apart are still one thought. Overridable from config.yaml only.
DEFAULT_REACTION_BURST_WINDOW_S = 2.0


@dataclass
class _Burst:
    message_ids: list
    last_ts: float


@dataclass
class _Pending:
    emoji: Optional[str]
    final_id: str


class ReactionBurstTracker:
    """In-memory burst state for one adapter. Not durable; a restart forgets it."""

    def __init__(self, window_s: float = DEFAULT_REACTION_BURST_WINDOW_S,
                 clock: Optional[Callable[[], float]] = None):
        self.window_s = float(window_s)
        self._clock = clock or time.monotonic
        self._bursts: dict[str, _Burst] = {}
        self._pending: dict[str, _Pending] = {}
        # Reactions whose burst closed between notes, waiting for a flush.
        self._ready: list[tuple[str, str, Optional[str]]] = []

    def _now(self, now: Optional[float] = None) -> float:
        return float(self._clock() if now is None else now)

    def note(self, chat_id, message_id, now: Optional[float] = None) -> None:
        """Record an inbound bubble. A gap larger than the window starts a new burst."""
        now = self._now(now)
        chat_id = str(chat_id or "")
        message_id = str(message_id or "")
        if not chat_id or not message_id or message_id == "None":
            return
        burst = self._bursts.get(chat_id)
        if burst is None or now - burst.last_ts > self.window_s:
            pending = self._pending.pop(chat_id, None)
            if pending is not None:
                self._ready.append((chat_id, pending.final_id, pending.emoji))
            self._bursts[chat_id] = _Burst([message_id], now)
            return
        if message_id not in burst.message_ids:
            burst.message_ids.append(message_id)
        burst.last_ts = now
        pending = self._pending.get(chat_id)
        if pending is not None:
            pending.final_id = burst.message_ids[-1]

    def should_defer(self, chat_id, message_id, now: Optional[float] = None) -> bool:
        """True when ``message_id`` is a non-final bubble of this chat's open burst."""
        now = self._now(now)
        chat_id = str(chat_id or "")
        message_id = str(message_id or "")
        burst = self._bursts.get(chat_id)
        if burst is None or not burst.message_ids:
            return False
        if now - burst.last_ts > self.window_s:
            return False
        if message_id not in burst.message_ids:
            return False
        return message_id != burst.message_ids[-1]

    def defer(self, chat_id, emoji: Optional[str]) -> str:
        """Hold ``emoji`` (None = clear) for the burst's current final bubble."""
        chat_id = str(chat_id)
        final_id = self._bursts[chat_id].message_ids[-1]
        self._pending[chat_id] = _Pending(emoji=emoji, final_id=final_id)
        return final_id

    def drop_pending_if_firing(self, chat_id, message_id) -> None:
        """An immediate fire on the current final replaces a held reaction."""
        chat_id = str(chat_id or "")
        message_id = str(message_id or "")
        pending = self._pending.get(chat_id)
        if pending is None:
            return
        burst = self._bursts.get(chat_id)
        final = burst.message_ids[-1] if burst and burst.message_ids else pending.final_id
        if message_id == final or message_id == pending.final_id:
            self._pending.pop(chat_id, None)

    def seconds_until_close(self, chat_id, now: Optional[float] = None) -> float:
        now = self._now(now)
        burst = self._bursts.get(str(chat_id))
        if burst is None:
            return 0.0
        return max(0.0, burst.last_ts + self.window_s - now)

    def take_due(self, now: Optional[float] = None) -> list[tuple[str, str, Optional[str]]]:
        """Pop reactions whose burst has closed: ``(chat_id, final_id, emoji)``."""
        now = self._now(now)
        due = list(self._ready)
        self._ready.clear()
        for chat_id, pending in list(self._pending.items()):
            burst = self._bursts.get(chat_id)
            if burst is None or now - burst.last_ts > self.window_s:
                due.append((chat_id, pending.final_id, pending.emoji))
                del self._pending[chat_id]
        return due
