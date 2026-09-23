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
from agent.jev_payload_hygiene import content_hash, text_metadata
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

QUALITY_GATE_MARKER = "fast_lane_quality_gate"
CONFIG_KEY = "gateway.telegram.fast_lane.quality_gate"
DEFAULT_THRESHOLD = 0.7
DEFAULT_TIMEOUT_SECONDS = 1.5
QUALITY_QUESTION_ID = "quality"

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


def build_quality_gate_request(
    user_message: str,
    draft_reply: str,
    *,
    model: str = DEFAULT_JEV_MODEL,
) -> dict:
    """System One choice request: hash/metadata for the user message and draft.

    Prompt shape is frozen (see ``CHOICE_INSTRUCTIONS`` / ``CHOICE_CRITERIA``).
    Neither the user message nor the draft reply is copied into ``state``.
    Any failure still fail-opens (send the draft); only a confident escalate
    leaves the lane.
    """
    return {
        "model": model,
        "state": [
            "LAB-52 fast-lane draft quality check. "
            "The inbound turn was classified as a no-task social/ack turn. "
            "Judge only from the hashes and metadata below; the user message "
            "and draft reply are not included.",
            "User message:\n" + text_metadata(user_message, label="user_message"),
            "Draft reply:\n" + text_metadata(draft_reply, label="draft_reply"),
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
    content_hash_value: str = "",
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
        "content_hash": content_hash_value or "",
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
    try:
        from gateway.jev_observability import record_jev_decision

        record_jev_decision(
            kind="quality_gate",
            tier=decision or choice,
            model=model,
            confidence=confidence,
            latency_ms=ready_ms,
            reason=reason or "ok",
            content_hash_value=content_hash_value,
        )
    except Exception:
        pass


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

    msg_hash = content_hash(f"{user_message or ''}\n{draft_reply or ''}")
    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_quality_gate(
            decision="send",
            model=gate.model,
            threshold=gate.threshold,
            chat_id=chat_id,
            reason="missing_key",
            content_hash_value=msg_hash,
        )
        return "send"

    if not (draft_reply or "").strip():
        log_quality_gate(
            decision="send",
            model=gate.model,
            threshold=gate.threshold,
            chat_id=chat_id,
            reason="empty_draft",
            content_hash_value=msg_hash,
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
                content_hash_value=msg_hash,
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
            content_hash_value=msg_hash,
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
        content_hash_value=msg_hash,
    )
    return decision
