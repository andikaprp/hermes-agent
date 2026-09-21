"""Optional TypeSafe Jev second opinion for uncertain fast-path turns.

The deterministic classifier in ``run_turn_fast_path`` stays first. Only its
declared uncertain band (shape-ok, lexicon miss) may call Jev. Disabled by
default; any failure falls back to today's behavior (main path for uncertain).
Never mutates the session cache or the cached prompt prefix — this runs before
the agent call and only decides routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _post_systemone,
    resolve_typesafe_api_key,
)
from agent.jev_payload_hygiene import content_hash, mask_state_text
from gateway.run_turn_fast_path import _ACK_WORDS, _DIRECTIVE_WORDS
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

JEV_ROUTING_MARKER = "jev_routing"
CONFIG_KEY = "gateway.telegram.jev_routing"
DEFAULT_THRESHOLD = 0.85
ROUTE_QUESTION_ID = "route"

# Wording used in the PR description / tests — keep stable.
CHOICE_INSTRUCTIONS = (
    "Is this inbound message a fast-lane turn (simple greeting/ack/social, no task) "
    "or a full-task turn (needs tools, work, or substantive handling)?"
)
CHOICE_CRITERIA = {
    "lane": "fast lane (simple greeting/ack/social reply with no task)",
    "task": "full task (needs tools, work, or substantive handling)",
}


@dataclass(frozen=True)
class JevRoutingConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def parse_jev_routing_config(raw: Any) -> JevRoutingConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevRoutingConfig()
    enabled = str(raw.get("enabled", False)).lower() in {"true", "1", "yes"}
    try:
        threshold = float(raw.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD
    threshold = max(0.0, min(1.0, threshold))
    model = str(raw.get("model") or DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
    try:
        timeout_seconds = float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = max(1.0, timeout_seconds)
    return JevRoutingConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def load_jev_routing_config(user_config: Any = None) -> JevRoutingConfig:
    """``gateway.telegram.jev_routing`` from user YAML; default OFF when absent."""
    try:
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        tg = gw.get("telegram") if isinstance(gw, dict) else None
        raw = tg.get("jev_routing") if isinstance(tg, dict) else None
        return parse_jev_routing_config(raw)
    except Exception:
        return JevRoutingConfig()


def _lexicon_state_context() -> str:
    """Classic no-task lexicon so Jev is not blind to the frozen classifier."""
    acks = ", ".join(sorted(_ACK_WORDS))
    directives = ", ".join(sorted(_DIRECTIVE_WORDS))
    return (
        "Hermes no-task fast-path lexicon context. "
        f"Ack words (lane when alone and assistant proposed nothing): {acks}. "
        f"Directive words (always full task): {directives}. "
        "Action verbs (merge/run/fix/…) anywhere mean full task. "
        "Bare greetings/thanks are social no-task. "
        "Ack answering an assistant proposal is a go-ahead (full task)."
    )


def build_jev_routing_request(message: str, *, model: str = DEFAULT_JEV_MODEL) -> dict:
    """System One choice request: redacted/truncated message + lexicon context as state."""
    text = mask_state_text(message or "", limit=280)
    return {
        "model": model,
        "state": [
            _lexicon_state_context(),
            f"Inbound message:\n{text}",
        ],
        "questions": {
            ROUTE_QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            },
        },
    }


def parse_route_answer(payload: Any) -> tuple[str, float]:
    """Extract ``(choice, confidence)`` from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(ROUTE_QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {ROUTE_QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip().lower()
    if choice not in CHOICE_CRITERIA:
        raise ValueError(f"unknown route choice: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("route answer missing confidence") from exc
    return choice, confidence


def log_jev_routing(
    *,
    state: str,
    model: str,
    choice: str,
    confidence: Optional[float],
    threshold: float,
    took_jev: bool,
    chat_id: Any = None,
    fallback: bool = False,
    reason: str = "",
    latency_ms: Optional[float] = None,
    content_hash_value: str = "",
) -> None:
    payload = {
        "marker": JEV_ROUTING_MARKER,
        "state": state,
        "model": model,
        "choice": choice or "",
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "took_jev": bool(took_jev),
        "fallback": bool(fallback),
        "reason": reason or "",
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
        "content_hash": content_hash_value or "",
    }
    logger.info(
        "[latency] "
        + JEV_ROUTING_MARKER
        + " chat=%s state=%s model=%s choice=%s confidence=%s threshold=%s "
        "took_jev=%s fallback=%s reason=%s latency_ms=%s",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX),
        payload["state"],
        payload["model"],
        payload["choice"] or "-",
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        "true" if took_jev else "false",
        "true" if fallback else "false",
        payload["reason"] or "ok",
        payload["latency_ms"] if payload["latency_ms"] is not None else "none",
        extra={"jev_routing": payload},
    )
    try:
        from gateway.jev_observability import record_jev_decision

        record_jev_decision(
            kind="routing",
            tier=choice or state,
            model=model,
            confidence=confidence,
            latency_ms=latency_ms,
            reason=reason or ("ok" if took_jev else "fallback"),
            content_hash_value=content_hash_value,
            took_jev=took_jev,
            fallback=fallback,
        )
    except Exception:
        pass


def maybe_jev_route_uncertain(
    message: Any,
    *,
    history: Optional[Sequence[Any]] = None,
    user_config: Any = None,
    chat_id: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """For an uncertain-band message: ask Jev, or return ``None`` (deterministic default).

    Returns ``"social"`` only when Jev chooses ``lane`` with confidence >= threshold.
    ``history`` is accepted for call-site symmetry; routing state is the message text.
    """
    del history  # state is the inbound text + lexicon; history unused by design
    cfg = load_jev_routing_config(user_config)
    if not cfg.enabled:
        return None
    msg_hash = content_hash(message) if isinstance(message, str) else ""
    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_routing(
            state="uncertain",
            model=cfg.model,
            choice="",
            confidence=None,
            threshold=cfg.threshold,
            took_jev=False,
            chat_id=chat_id,
            fallback=True,
            reason="missing_key",
            content_hash_value=msg_hash,
        )
        return None
    if not isinstance(message, str) or not message.strip():
        log_jev_routing(
            state="uncertain",
            model=cfg.model,
            choice="",
            confidence=None,
            threshold=cfg.threshold,
            took_jev=False,
            chat_id=chat_id,
            fallback=True,
            reason="empty_message",
            content_hash_value=msg_hash,
        )
        return None
    ready_ms: Optional[float] = None
    try:
        body = build_jev_routing_request(message, model=cfg.model)
        data, _ttft_ms, ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=cfg.timeout_seconds,
            http_client=http_client,
        )
        choice, confidence = parse_route_answer(data)
    except Exception as exc:
        reason = type(exc).__name__
        # Prefer a short stable reason when the exception message is known.
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "http " in msg.lower() or "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = msg.replace(" ", "_")[:64] if msg else reason
        log_jev_routing(
            state="uncertain",
            model=cfg.model,
            choice="",
            confidence=None,
            threshold=cfg.threshold,
            took_jev=False,
            chat_id=chat_id,
            fallback=True,
            reason=reason,
            latency_ms=ready_ms,
            content_hash_value=msg_hash,
        )
        return None

    took = confidence >= cfg.threshold
    log_jev_routing(
        state="uncertain",
        model=cfg.model,
        choice=choice,
        confidence=confidence,
        threshold=cfg.threshold,
        took_jev=took,
        chat_id=chat_id,
        fallback=not took,
        reason="" if took else "below_threshold",
        latency_ms=ready_ms,
        content_hash_value=msg_hash,
    )
    if took and choice == "lane":
        return "social"
    return None
