"""Consumer for a Jev completion HITL flag (LAB-61).

``apply_completion_verdict`` sets ``hitl_escalation`` and appends a notice.
Nothing in that module delivers an ask. This module reads the flag and ends
the turn in a real question on the existing clarification seam
(``tools.clarify_gateway``): a pending clarify plus an explicit question, and
``completed`` stays false so the turn is not auto-claimed done.

LAB-57's ``hitl_gate`` lives in ningning-proactive-agent and is not imported.
The local mirror of its ``ask`` outcome is: register a pending question and do
not act. Config stays on the verifier (``gateway.jev_completion``, default
off). When the flag is absent this consumer is a no-op.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("gateway.run_turn")

CLARIFY_SEAM = "tools.clarify_gateway"
HITL_CHOICES = ("Confirm done", "Not done")
ASK_PHRASE = "Did this turn finish what you asked?"
NOTICE_MARKER = "⚠️ Completion check needs a human look"
DEFAULT_TIMEOUT_SECONDS = 3600

HITL_ASK_TEMPLATE = (
    "I need you to check this before I mark the turn done "
    "(verdict={verdict}, confidence={confidence}).\n"
    f"{ASK_PHRASE}\n"
    "1. Confirm done\n"
    "2. Not done\n"
    "Reply with 1, 2, or a correction."
)
CONFIRM_ACK = (
    "You confirmed. I had not marked this turn done — this reply is the "
    "human check, not an automatic completion."
)
NOT_DONE_ACK = (
    "Noted — not done. I had not marked this turn done. "
    "Send the correction when you're ready."
)

_CONFIRM = {"1", "1.", "confirm", "confirm done", "yes"}
_NOT_DONE = {"2", "2.", "not done", "no"}

_pending: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()


@dataclass(frozen=True)
class HitlIntercept:
    """Reply to a pending completion ask.

    ``ack`` is a gateway reply (confirm / not-done) and must not claim the
    verifier marked the turn done. ``release`` drops the ask so the user's
    text continues as a follow-up.
    """

    action: str
    text: str = ""


def _normalize_reply(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _confidence_text(confidence: Any) -> str:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return "n/a"
    return f"{float(confidence):.2f}"


def _draft_excerpt(final: Any) -> str:
    text = final if isinstance(final, str) else ""
    if NOTICE_MARKER in text:
        text = text.split(NOTICE_MARKER, 1)[0]
    text = text.strip()
    if not text:
        return ""
    if len(text) > 400:
        text = text[:400].rstrip() + "…"
    return f"Draft reply (not a completion):\n{text}\n\n"


def build_hitl_question(agent_result: dict) -> str:
    """Explicit question. The draft, if any, is labeled as not a completion."""
    meta = agent_result.get("jev_completion")
    meta = meta if isinstance(meta, dict) else {}
    verdict = str(meta.get("verdict") or "unknown")
    ask = HITL_ASK_TEMPLATE.format(
        verdict=verdict,
        confidence=_confidence_text(meta.get("confidence")),
    )
    return f"{_draft_excerpt(agent_result.get('final_response'))}{ask}"


def get_pending_ask(session_key: str) -> Optional[dict]:
    """Copy of the pending completion ask for ``session_key``, or None."""
    if not session_key:
        return None
    with _lock:
        entry = _pending.get(session_key)
        return dict(entry) if entry else None


def has_pending_ask(session_key: str) -> bool:
    return get_pending_ask(session_key) is not None


def _release_clarify(session_key: str, clarify_id: Optional[str]) -> None:
    if not session_key or not clarify_id:
        return
    try:
        from tools import clarify_gateway

        current = clarify_gateway.get_pending_for_session(
            session_key, include_choice_prompts=True,
        )
        if current is None or current.clarify_id != clarify_id:
            return
        clarify_gateway.resolve_gateway_clarify(clarify_id, "")
        clarify_gateway.clear_session(session_key)
    except Exception:
        logger.debug("completion HITL clarify release failed", exc_info=True)


def clear_pending_ask(session_key: str) -> None:
    """Drop the pending completion ask and its clarify entry, if any."""
    if not session_key:
        return
    with _lock:
        entry = _pending.pop(session_key, None)
    if entry:
        _release_clarify(session_key, entry.get("clarify_id"))


def _is_stale(entry: dict, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    created = float(entry.get("created_at") or 0)
    return time.time() - created > timeout


def consume_hitl_escalation(agent_result: Any, *, session_key: str) -> Optional[str]:
    """Read ``hitl_escalation``. When set, register a clarify ask and return it.

    Returns None when the flag is absent (gate off, confident done, fail-open).
    Never sets ``completed`` true. A missing session key still rewrites the
    reply into the question so delivery is not a normal done claim; the
    pending clarify needs a key for the inbound intercept.
    """
    if not isinstance(agent_result, dict) or not agent_result.get("hitl_escalation"):
        return None

    session_key = str(session_key or "")
    existing = get_pending_ask(session_key) if session_key else None
    if agent_result.get("hitl_ask_pending") and existing and existing.get("question"):
        return str(existing["question"])

    question = build_hitl_question(agent_result)
    clarify_id = ""
    if session_key:
        if existing:
            clear_pending_ask(session_key)
        clarify_id = uuid.uuid4().hex[:12]
        from tools.clarify_gateway import register as register_clarify

        register_clarify(
            clarify_id=clarify_id,
            session_key=session_key,
            question=question,
            choices=list(HITL_CHOICES),
        )
        meta = agent_result.get("jev_completion")
        meta = meta if isinstance(meta, dict) else {}
        with _lock:
            _pending[session_key] = {
                "clarify_id": clarify_id,
                "question": question,
                "choices": list(HITL_CHOICES),
                "seam": CLARIFY_SEAM,
                "created_at": time.time(),
                "verdict": str(meta.get("verdict") or ""),
            }
        logger.info(
            "jev_completion_hitl ask registered verdict=%s confidence=%s seam=%s",
            meta.get("verdict") or "unknown",
            _confidence_text(meta.get("confidence")),
            CLARIFY_SEAM,
        )

    agent_result["completed"] = False
    agent_result["partial"] = True
    agent_result["approval_pending"] = True
    agent_result["hitl_ask_pending"] = True
    agent_result["hitl_escalation_consumed"] = True
    # A streamed body may already have left. Clearing this sends the question
    # as the post-turn ask instead of suppressing it as a duplicate final.
    agent_result["already_sent"] = False
    agent_result["final_response"] = question
    agent_result["hitl_ask"] = {
        "seam": CLARIFY_SEAM,
        "clarify_id": clarify_id,
        "choices": list(HITL_CHOICES),
        "question": question,
    }
    return question


def intercept_completion_reply(session_key: str, text: str) -> Optional[HitlIntercept]:
    """Resolve a reply to a pending completion ask.

    None means no pending ask, a stale ask (dropped), or a slash/empty reply
    that must leave the ask pending. Confirm and not-done return an ack.
    Any other text releases the ask so the message can continue as a follow-up.
    """
    entry = get_pending_ask(session_key)
    if entry is None:
        return None
    if _is_stale(entry):
        clear_pending_ask(session_key)
        return None
    raw = (text or "").strip()
    if not raw or raw.startswith("/"):
        return None
    normalized = _normalize_reply(raw)
    if normalized in _CONFIRM:
        clear_pending_ask(session_key)
        return HitlIntercept(action="ack", text=CONFIRM_ACK)
    if normalized in _NOT_DONE:
        clear_pending_ask(session_key)
        return HitlIntercept(action="ack", text=NOT_DONE_ACK)
    clear_pending_ask(session_key)
    return HitlIntercept(action="release")
