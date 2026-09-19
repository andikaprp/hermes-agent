"""Optional TypeSafe Jev quality check on a LAB-52 fast-lane draft.

Runs AFTER the compact lane produces a draft and BEFORE that draft is sent.
Disabled by default; any failure / timeout / missing key / over-budget latency
fail-opens (send the draft). A confident ``escalate`` returns control to the
existing one-hop conversation path (``try_fast_lane`` -> ``None``).

Never mutates the session cache or the cached prompt prefix — tools stay on the
wire for the fallback path; this is an independent System One request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    _post_systemone,
    resolve_typesafe_api_key,
)
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

QUALITY_GATE_MARKER = "fast_lane_quality_gate"
CONFIG_KEY = "gateway.telegram.fast_lane.quality_gate"
DEFAULT_THRESHOLD = 0.7
DEFAULT_TIMEOUT_SECONDS = 1.5
QUALITY_QUESTION_ID = "quality"
_STATE_REDACT_CHARS = 280

# Frozen prompt shape — keep stable (PR / tests document this wording).
CHOICE_INSTRUCTIONS = (
    "Is this short casual reply good enough to send as-is for a no-task social turn, "
    "or is it poor/irrelevant enough that the full assistant should handle it?"
)
CHOICE_CRITERIA = {
    "send": "good enough to send as-is for a no-task social turn",
    "escalate": (
        "poor or irrelevant enough that the full assistant should handle it"
    ),
}


@dataclass(frozen=True)
class FastLaneQualityGateConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def parse_quality_gate_config(raw: Any) -> FastLaneQualityGateConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return FastLaneQualityGateConfig()
    enabled = str(raw.get("enabled", False)).lower() in {"true", "1", "yes"}
    try:
        threshold = float(raw.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD
    threshold = max(0.0, min(1.0, threshold))
    model = str(raw.get("model") or DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
    try:
        timeout_seconds = float(
            raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        )
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = max(0.1, timeout_seconds)
    return FastLaneQualityGateConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def load_quality_gate_config(user_config: Any = None) -> FastLaneQualityGateConfig:
    """``gateway.telegram.fast_lane.quality_gate`` from user YAML; default OFF."""
    try:
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        tg = gw.get("telegram") if isinstance(gw, dict) else None
        lane = tg.get("fast_lane") if isinstance(tg, dict) else None
        raw = lane.get("quality_gate") if isinstance(lane, dict) else None
        return parse_quality_gate_config(raw)
    except Exception:
        return FastLaneQualityGateConfig()


def _redact_lane_text(text: str, *, limit: int = _STATE_REDACT_CHARS) -> str:
    """Truncate lane context so long content is not sent to Jev."""
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def build_quality_gate_request(
    user_message: str,
    draft_reply: str,
    *,
    model: str = DEFAULT_JEV_MODEL,
) -> dict:
    """System One choice request: redacted user + draft as state.

    Prompt shape is frozen (see ``CHOICE_INSTRUCTIONS`` / ``CHOICE_CRITERIA``).
    """
    return {
        "model": model,
        "state": [
            "LAB-52 fast-lane draft quality check. "
            "The inbound turn was classified as a no-task social/ack turn. "
            "Judge only whether this short draft is acceptable to send as-is.",
            f"User message:\n{_redact_lane_text(user_message)}",
            f"Draft reply:\n{_redact_lane_text(draft_reply)}",
        ],
        "questions": {
            QUALITY_QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            },
        },
    }


def parse_quality_answer(payload: Any) -> tuple[str, float]:
    """Extract ``(choice, confidence)`` from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(QUALITY_QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {QUALITY_QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip().lower()
    if choice not in CHOICE_CRITERIA:
        raise ValueError(f"unknown quality choice: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quality answer missing confidence") from exc
    return choice, confidence


def log_quality_gate(
    *,
    decision: str,
    model: str,
    choice: str = "",
    confidence: Optional[float] = None,
    threshold: float = DEFAULT_THRESHOLD,
    ready_ms: Optional[float] = None,
    chat_id: Any = None,
    reason: str = "",
) -> None:
    payload = {
        "marker": QUALITY_GATE_MARKER,
        "decision": decision,
        "model": model,
        "choice": choice or "",
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "ready_ms": None if ready_ms is None else round(float(ready_ms), 1),
        "reason": reason or "",
    }
    logger.info(
        "[latency] "
        + QUALITY_GATE_MARKER
        + " chat=%s decision=%s model=%s choice=%s confidence=%s threshold=%s "
        "ready_ms=%s reason=%s",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX),
        payload["decision"],
        payload["model"],
        payload["choice"] or "-",
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        payload["ready_ms"] if payload["ready_ms"] is not None else "none",
        payload["reason"] or "ok",
        extra={"fast_lane_quality_gate": payload},
    )


def evaluate_fast_lane_draft(
    user_message: str,
    draft_reply: str,
    *,
    user_config: Any = None,
    chat_id: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
    cfg: Optional[FastLaneQualityGateConfig] = None,
) -> str:
    """Return ``\"send\"`` or ``\"escalate\"`` for a lane draft.

    Fail-open: disabled / missing key / HTTP / parse / timeout / over-budget /
    below-threshold escalate all return ``\"send\"``. Only a confident
    ``escalate`` choice returns ``\"escalate\"``.
    """
    gate = cfg if cfg is not None else load_quality_gate_config(user_config)
    if not gate.enabled:
        return "send"

    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_quality_gate(
            decision="send",
            model=gate.model,
            threshold=gate.threshold,
            chat_id=chat_id,
            reason="missing_key",
        )
        return "send"

    if not (draft_reply or "").strip():
        log_quality_gate(
            decision="send",
            model=gate.model,
            threshold=gate.threshold,
            chat_id=chat_id,
            reason="empty_draft",
        )
        return "send"

    budget_ms = gate.timeout_seconds * 1000.0
    try:
        body = build_quality_gate_request(
            user_message, draft_reply, model=gate.model,
        )
        data, _ttft_ms, ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=gate.timeout_seconds,
            http_client=http_client,
        )
        if ready_ms > budget_ms:
            log_quality_gate(
                decision="send",
                model=gate.model,
                threshold=gate.threshold,
                ready_ms=ready_ms,
                chat_id=chat_id,
                reason="over_budget",
            )
            return "send"
        choice, confidence = parse_quality_answer(data)
    except Exception as exc:
        reason = type(exc).__name__
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = "timeout"
        elif "http " in msg.lower():
            reason = msg.replace(" ", "_")[:64] if msg else reason
        log_quality_gate(
            decision="send",
            model=gate.model,
            threshold=gate.threshold,
            chat_id=chat_id,
            reason=reason,
        )
        return "send"

    clear_escalate = (
        choice == "escalate" and confidence >= gate.threshold
    )
    decision = "escalate" if clear_escalate else "send"
    log_quality_gate(
        decision=decision,
        model=gate.model,
        choice=choice,
        confidence=confidence,
        threshold=gate.threshold,
        ready_ms=ready_ms,
        chat_id=chat_id,
        reason="" if clear_escalate or choice == "send" else "below_threshold",
    )
    return decision
