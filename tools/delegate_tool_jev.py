"""Optional TypeSafe Jev System One gate before ``delegate_task`` spawns.

When ``delegation.jev_check.enabled`` is true and ``TYPESAFE_API_KEY`` is set,
asks Jev whether the parent should handle the work directly or genuinely needs
a separate subagent context. Disabled / missing key / timeout / any API failure
falls through to today's default (spawn). Never blocks beyond ``timeout_seconds``.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _post_systemone,
    resolve_typesafe_api_key,
)

logger = logging.getLogger(__name__)

JEV_DELEGATE_MARKER = "jev_delegate"
DEFAULT_THRESHOLD = 0.8
DEFAULT_BEHAVIOR = "delegate"  # current hermes default: always spawn

CHOICE_INSTRUCTIONS = (
    "Should the parent agent handle this task in its own context, or does it "
    "genuinely need a separate subagent/worker with isolated context?"
)
CHOICE_CRITERIA = {
    "delegate": (
        "Needs a separate context/worker: reasoning-heavy subtask, work that "
        "would flood the parent context with intermediate data, or an independent "
        "parallel workstream the parent cannot do cleanly inline."
    ),
    "do_directly": (
        "Parent can handle it itself: a single tool call, mechanical steps with "
        "no separate reasoning budget needed, or work that does not benefit from "
        "an isolated subagent conversation."
    ),
}


@dataclass(frozen=True)
class JevCheckConfig:
    """Parsed ``delegation.jev_check`` block."""

    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class JevDelegateDecision:
    """Outcome of the optional pre-spawn Jev gate."""

    spawn: bool
    choice: str
    confidence: float
    override: bool
    fallback: bool
    task_hash: str
    reason: str = ""


def parse_jev_check_config(raw: Any) -> JevCheckConfig:
    """Build ``JevCheckConfig`` from a config mapping; unknown/malformed -> defaults."""
    if not isinstance(raw, dict):
        return JevCheckConfig()
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
    return JevCheckConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def task_hash_for(task_list: Sequence[Dict[str, Any]], context: Optional[str] = None) -> str:
    """Stable short hash of the spawn payload (goals + optional shared context)."""
    parts: List[str] = []
    if context:
        parts.append(str(context))
    for task in task_list:
        if not isinstance(task, dict):
            continue
        parts.append(str(task.get("goal") or ""))
        if task.get("context"):
            parts.append(str(task["context"]))
    blob = "\n".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_task_state(task_list: Sequence[Dict[str, Any]], context: Optional[str] = None) -> str:
    """Flatten the pending delegation into one state string for System One."""
    lines: List[str] = []
    if context:
        lines.append(f"Shared context: {context}")
    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            continue
        goal = str(task.get("goal") or "").strip()
        tctx = str(task.get("context") or "").strip()
        block = f"Task {i + 1}: {goal}" if goal else f"Task {i + 1}:"
        if tctx:
            block = f"{block}\nTask context: {tctx}"
        lines.append(block)
    return "\n\n".join(lines).strip() or "(empty task)"


def build_delegate_choice_request(
    task_list: Sequence[Dict[str, Any]],
    *,
    context: Optional[str] = None,
    model: str = DEFAULT_JEV_MODEL,
) -> Dict[str, Any]:
    """One System One choice question: delegate vs do_directly."""
    return {
        "model": model,
        "state": build_task_state(task_list, context),
        "questions": {
            "route": {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            }
        },
    }


def parse_choice_answer(payload: Any) -> tuple[str, float]:
    """Extract ``(choice, confidence)`` from a System One choice response."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get("route")
    if not isinstance(ans, dict):
        raise ValueError("missing answer for route")
    choice = str(ans.get("choice") or "").strip().lower()
    if choice not in CHOICE_CRITERIA:
        raise ValueError(f"unknown choice {choice!r}")
    confidence = float(ans.get("confidence", 0.0))
    return choice, confidence


def log_jev_delegate(
    *,
    task_hash: str,
    choice: str,
    confidence: float,
    override: bool,
    fallback: bool,
    reason: str = "",
) -> None:
    """Single measurable line for the pre-spawn gate."""
    payload = {
        "marker": JEV_DELEGATE_MARKER,
        "task_hash": task_hash,
        "choice": choice,
        "confidence": round(float(confidence), 4),
        "override": bool(override),
        "fallback": bool(fallback),
        "reason": reason or "",
    }
    logger.info(
        "[latency] "
        + JEV_DELEGATE_MARKER
        + " task_hash=%s choice=%s confidence=%s override=%s fallback=%s reason=%s",
        payload["task_hash"],
        payload["choice"],
        payload["confidence"],
        "true" if override else "false",
        "true" if fallback else "false",
        payload["reason"] or "ok",
        extra={"jev_delegate": payload},
    )


def _call_systemone(
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
    http_client: Any = None,
) -> Any:
    data, _ttft, _ready = _post_systemone(
        body, api_key=api_key, timeout_seconds=timeout_seconds, http_client=http_client,
    )
    return data


def evaluate_jev_delegate_gate(
    task_list: Sequence[Dict[str, Any]],
    *,
    context: Optional[str] = None,
    cfg: Optional[JevCheckConfig] = None,
    raw_config: Any = None,
    api_key: Optional[str] = None,
    http_client: Any = None,
) -> JevDelegateDecision:
    """Decide whether to spawn. Never raises; default is always spawn.

    Override only when confidence >= threshold AND choice differs from the
    current default (``delegate`` / spawn). Below threshold or any failure -> spawn.
    """
    parsed = cfg if cfg is not None else parse_jev_check_config(raw_config)
    th = task_hash_for(task_list, context)
    if not parsed.enabled:
        # Disabled: no API call, no log noise on the hot path.
        return JevDelegateDecision(
            spawn=True,
            choice=DEFAULT_BEHAVIOR,
            confidence=0.0,
            override=False,
            fallback=False,
            task_hash=th,
            reason="disabled",
        )

    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_delegate(
            task_hash=th, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, reason="missing_key",
        )
        return JevDelegateDecision(
            spawn=True, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, task_hash=th, reason="missing_key",
        )

    body = build_delegate_choice_request(task_list, context=context, model=parsed.model)
    try:
        # Hard wall-clock bound: httpx timeout alone can leave a hung DNS/connect
        # past the configured budget; never block spawn beyond timeout_seconds.
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                _call_systemone,
                body,
                api_key=key,
                timeout_seconds=parsed.timeout_seconds,
                http_client=http_client,
            )
            data = fut.result(timeout=parsed.timeout_seconds)
        choice, confidence = parse_choice_answer(data)
    except FuturesTimeoutError:
        log_jev_delegate(
            task_hash=th, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, reason="timeout",
        )
        return JevDelegateDecision(
            spawn=True, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, task_hash=th, reason="timeout",
        )
    except Exception as exc:
        reason = type(exc).__name__
        log_jev_delegate(
            task_hash=th, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, reason=reason,
        )
        return JevDelegateDecision(
            spawn=True, choice=DEFAULT_BEHAVIOR, confidence=0.0,
            override=False, fallback=True, task_hash=th, reason=reason,
        )

    if confidence < parsed.threshold:
        log_jev_delegate(
            task_hash=th, choice=choice, confidence=confidence,
            override=False, fallback=True, reason="below_threshold",
        )
        return JevDelegateDecision(
            spawn=True, choice=choice, confidence=confidence,
            override=False, fallback=True, task_hash=th, reason="below_threshold",
        )

    # High confidence: override only when choice != default spawn behavior.
    if choice == "do_directly":
        log_jev_delegate(
            task_hash=th, choice=choice, confidence=confidence,
            override=True, fallback=False, reason="ok",
        )
        return JevDelegateDecision(
            spawn=False, choice=choice, confidence=confidence,
            override=True, fallback=False, task_hash=th, reason="ok",
        )

    log_jev_delegate(
        task_hash=th, choice=choice, confidence=confidence,
        override=False, fallback=False, reason="ok",
    )
    return JevDelegateDecision(
        spawn=True, choice=choice, confidence=confidence,
        override=False, fallback=False, task_hash=th, reason="ok",
    )


def skipped_spawn_payload(decision: JevDelegateDecision, task_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """JSON body returned to the parent when Jev overrides spawn."""
    goals = [str(t.get("goal") or "") for t in task_list if isinstance(t, dict)]
    return {
        "status": "skipped",
        "reason": "jev_do_directly",
        "choice": decision.choice,
        "confidence": decision.confidence,
        "task_hash": decision.task_hash,
        "tasks": goals,
        "message": (
            "Jev recommends handling this in the parent context without spawning "
            "a subagent. Do the work directly with your existing tools."
        ),
    }
