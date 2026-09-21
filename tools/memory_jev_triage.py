"""Optional TypeSafe Jev noul triage before durable memory ``add`` writes.

When ``memory.jev_triage.enabled`` is true and ``TYPESAFE_API_KEY`` is set, a
proposed ``memory add`` entry is scored with a System One noul (yes/no
probability) asking whether the fact is durable enough for the permanent store.
Disabled by default. Below threshold skips the write; any failure writes as today.
A cheap length pre-filter skips the ~0.5–1s Jev call for short entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _post_systemone,
    resolve_typesafe_api_key,
)
from agent.jev_payload_hygiene import content_hash, mask_state_text

logger = logging.getLogger(__name__)

JEV_MEMORY_TRIAGE_MARKER = "jev_memory_triage"
CONFIG_KEY = "memory.jev_triage"
DEFAULT_THRESHOLD = 0.75
# Cheap pre-filter: skip Jev for short session-noise-sized strings.
DEFAULT_MIN_CHARS = 40
QUESTION_ID = "durable"

NOUL_INSTRUCTIONS = (
    "Is this proposed memory entry a durable fact worth the permanent store "
    "(vs transient session detail)?"
)
NOUL_CRITERIA = {
    "true": "Durable fact that should persist across sessions (identity, standing prefs, stable env)",
    "false": "Transient session detail (task progress, one-off context, ephemeral chatter)",
}


@dataclass(frozen=True)
class JevMemoryTriageConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    min_chars: int = DEFAULT_MIN_CHARS


def parse_jev_memory_triage_config(raw: Any) -> JevMemoryTriageConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevMemoryTriageConfig()
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
    try:
        min_chars = int(raw.get("min_chars", DEFAULT_MIN_CHARS))
    except (TypeError, ValueError):
        min_chars = DEFAULT_MIN_CHARS
    min_chars = max(0, min_chars)
    return JevMemoryTriageConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
        min_chars=min_chars,
    )


def load_jev_memory_triage_config(user_config: Any = None) -> JevMemoryTriageConfig:
    """``memory.jev_triage`` from config; default OFF when absent."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly

            user_config = load_config_readonly()
        mem = user_config.get("memory") if isinstance(user_config, dict) else None
        raw = mem.get("jev_triage") if isinstance(mem, dict) else None
        return parse_jev_memory_triage_config(raw)
    except Exception:
        return JevMemoryTriageConfig()


def build_jev_memory_triage_request(
    entry_text: str,
    *,
    target: str = "memory",
    model: str = DEFAULT_JEV_MODEL,
) -> dict:
    """System One noul request over a redacted/truncated proposed entry."""
    label = "user profile" if target == "user" else "memory"
    text = mask_state_text(entry_text or "", limit=480)
    return {
        "model": model,
        "state": [
            f"Proposed {label} entry to persist across sessions:\n{text}",
        ],
        "questions": {
            QUESTION_ID: {
                "type": "noul",
                "instructions": NOUL_INSTRUCTIONS,
                "criteria": dict(NOUL_CRITERIA),
            },
        },
    }


def parse_noul_answer(payload: Any, question_id: str = QUESTION_ID) -> float:
    """Extract noul probability (0..1) from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(question_id)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {question_id}")
    if "noul" in ans:
        return float(ans["noul"])
    # Some payloads expose the yes-probability as ``probability`` / ``score``.
    for key in ("probability", "score", "value"):
        if key in ans:
            return float(ans[key])
    raise ValueError(f"answer {question_id} has no noul")


def log_jev_memory_triage(
    *,
    skipped: bool,
    noul: Optional[float],
    threshold: float,
    reason: str = "",
    model: str = "",
    fallback: bool = False,
    content_hash_value: str = "",
    latency_ms: Optional[float] = None,
) -> None:
    payload = {
        "marker": JEV_MEMORY_TRIAGE_MARKER,
        "skipped": bool(skipped),
        "noul": None if noul is None else round(float(noul), 4),
        "threshold": float(threshold),
        "model": model or "",
        "fallback": bool(fallback),
        "reason": reason or "",
        "content_hash": content_hash_value or "",
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
    }
    logger.info(
        "[latency] "
        + JEV_MEMORY_TRIAGE_MARKER
        + " skipped=%s noul=%s threshold=%s model=%s fallback=%s reason=%s",
        "true" if payload["skipped"] else "false",
        payload["noul"] if payload["noul"] is not None else "none",
        payload["threshold"],
        payload["model"] or "-",
        "true" if payload["fallback"] else "false",
        payload["reason"] or "ok",
        extra={"jev_memory_triage": payload},
    )
    try:
        from gateway.jev_observability import record_jev_decision

        record_jev_decision(
            kind="memory_triage",
            tier="skip" if skipped else "keep",
            model=model,
            confidence=noul,
            latency_ms=latency_ms,
            reason=reason or "ok",
            content_hash_value=content_hash_value,
            fallback=fallback,
        )
    except Exception:
        pass


@dataclass(frozen=True)
class JevMemoryTriageResult:
    """Outcome of optional triage. ``allow_write`` False means skip the durable write."""

    allow_write: bool
    skipped: bool = False
    noul: Optional[float] = None
    reason: str = ""
    fallback: bool = False


def should_triage_entry(content: str, cfg: JevMemoryTriageConfig) -> bool:
    """Cheap pre-filter: only call Jev when enabled and entry clears min length."""
    if not cfg.enabled:
        return False
    text = (content or "").strip()
    return len(text) >= cfg.min_chars


def triage_memory_add(
    content: str,
    *,
    target: str = "memory",
    user_config: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
    cfg: Optional[JevMemoryTriageConfig] = None,
) -> JevMemoryTriageResult:
    """Ask Jev whether ``content`` is durable; fail-open to allow write.

    Returns ``allow_write=False`` only when Jev answered with noul below threshold.
    """
    resolved = cfg if cfg is not None else load_jev_memory_triage_config(user_config)
    if not resolved.enabled:
        return JevMemoryTriageResult(allow_write=True, reason="disabled")
    text = (content or "").strip()
    if not should_triage_entry(text, resolved):
        return JevMemoryTriageResult(allow_write=True, reason="prefilter_short")
    entry_hash = content_hash(text)
    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_memory_triage(
            skipped=False,
            noul=None,
            threshold=resolved.threshold,
            reason="missing_key",
            model=resolved.model,
            fallback=True,
            content_hash_value=entry_hash,
        )
        return JevMemoryTriageResult(allow_write=True, fallback=True, reason="missing_key")
    ready_ms = None
    try:
        body = build_jev_memory_triage_request(text, target=target, model=resolved.model)
        data, _ttft_ms, ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=resolved.timeout_seconds,
            http_client=http_client,
        )
        noul = parse_noul_answer(data)
    except Exception as exc:
        reason = type(exc).__name__
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "http " in msg.lower() or "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = msg.replace(" ", "_")[:64] if msg else reason
        log_jev_memory_triage(
            skipped=False,
            noul=None,
            threshold=resolved.threshold,
            reason=reason,
            model=resolved.model,
            fallback=True,
            content_hash_value=entry_hash,
            latency_ms=ready_ms,
        )
        return JevMemoryTriageResult(allow_write=True, fallback=True, reason=reason)

    if noul >= resolved.threshold:
        log_jev_memory_triage(
            skipped=False,
            noul=noul,
            threshold=resolved.threshold,
            reason="",
            model=resolved.model,
            fallback=False,
            content_hash_value=entry_hash,
            latency_ms=ready_ms,
        )
        return JevMemoryTriageResult(allow_write=True, noul=noul, reason="above_threshold")

    log_jev_memory_triage(
        skipped=True,
        noul=noul,
        threshold=resolved.threshold,
        reason="below_threshold",
        model=resolved.model,
        fallback=False,
        content_hash_value=entry_hash,
        latency_ms=ready_ms,
    )
    return JevMemoryTriageResult(
        allow_write=False,
        skipped=True,
        noul=noul,
        reason="below_threshold",
    )


def skipped_add_tool_result(target: str, content: str, triage: JevMemoryTriageResult) -> dict:
    """JSON-serializable tool result when a proposed add is skipped by triage."""
    return {
        "success": True,
        "skipped": True,
        "target": target,
        "noul": triage.noul,
        "reason": triage.reason or "below_threshold",
        "message": (
            "Jev memory triage skipped this write: entry looks like transient session "
            "detail rather than a durable fact. Not saved to the permanent store."
        ),
        "content_preview": (content or "")[:120],
    }
