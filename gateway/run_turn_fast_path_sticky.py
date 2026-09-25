"""Per-chat sticky fast-path verdict (gateway.telegram.fast_lane.sticky_seconds)."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from gateway.run_turn_fast_path import _ACTION_VERB_RE, _URL_RE

_STICKY_TASK = "__task__"
_sticky_lock = threading.Lock()
_sticky_by_chat: dict[Any, tuple[str, float]] = {}


def load_sticky_seconds(user_config: Any = None) -> int:
    """``gateway.telegram.fast_lane.sticky_seconds``; 0 means off."""
    try:
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        tg = gw.get("telegram") if isinstance(gw, dict) else None
        fl = tg.get("fast_lane") if isinstance(tg, dict) else None
        if not isinstance(fl, dict) or "sticky_seconds" not in fl:
            return 0
        return max(0, int(fl.get("sticky_seconds") or 0))
    except (TypeError, ValueError):
        return 0


def sticky_hard_bypass(message: Any) -> bool:
    if not isinstance(message, str):
        return True
    stripped = message.strip()
    if not stripped:
        return True
    if _URL_RE.search(stripped) or _ACTION_VERB_RE.search(stripped):
        return True
    return False


def sticky_lookup(chat_id: Any, sticky_seconds: int) -> Optional[str]:
    """Return stored lane verdict, ``None`` for sticky task mode, or miss when absent/expired."""
    if sticky_seconds <= 0:
        return None
    now = time.monotonic()
    with _sticky_lock:
        row = _sticky_by_chat.get(chat_id)
        if row is None:
            return None
        verdict, expires = row
        if now >= expires:
            _sticky_by_chat.pop(chat_id, None)
            return None
        _sticky_by_chat[chat_id] = (verdict, now + float(sticky_seconds))
    if verdict == _STICKY_TASK:
        return None
    return verdict


def sticky_has_entry(chat_id: Any, sticky_seconds: int) -> bool:
    """True when an unexpired sticky row exists (lane or task)."""
    if sticky_seconds <= 0:
        return False
    now = time.monotonic()
    with _sticky_lock:
        row = _sticky_by_chat.get(chat_id)
        if row is None:
            return False
        _, expires = row
        return now < expires


def sticky_store(chat_id: Any, verdict: Optional[str], sticky_seconds: int) -> None:
    if sticky_seconds <= 0:
        return
    stored = _STICKY_TASK if verdict is None else verdict
    expires = time.monotonic() + float(sticky_seconds)
    with _sticky_lock:
        _sticky_by_chat[chat_id] = (stored, expires)


def clear_sticky_state() -> None:
    """Test seam: drop all in-memory sticky rows."""
    with _sticky_lock:
        _sticky_by_chat.clear()
