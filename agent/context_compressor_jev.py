"""Config-gated TypeSafe Jev keep-priority scoring for context compression.

When ``compression.jev_scorer.enabled`` is true and ``TYPESAFE_API_KEY`` is set,
the compressor scores the compressible middle window with Jev (System One) before
the existing summarizer runs. Disabled / missing key / any API failure falls back
to the byte-identical existing path. Scoring never mutates the cached prefix
outside the compression swap.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

JEV_SCORER_MARKER = "jev_scorer"
TYPESAFE_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_BATCH_SIZE = 40
DEFAULT_KEEP_THRESHOLD = 1.2
DEFAULT_TIMEOUT_SECONDS = 30.0
# TypeSafe budget: state + longest question ~32k tokens (~150k English chars).
STATE_PLUS_QUESTION_BUDGET_CHARS = 150_000
TRUNCATE_INDICATOR = "\n...[truncated for Jev scoring]...\n"

SCORE_INSTRUCTIONS = (
    "How important is this message for correctly answering the user's most recent request?"
)
SCORE_CRITERIA = [
    "Safe to drop: noise, redundant tool dumps, or content not needed for the latest user request",
    "Background: useful context that helps but is not decisive for the latest request",
    "Must keep: directly required to correctly answer the user's most recent request",
]

# Synthetic / goal scaffolding that must survive scoring.
_PIN_PREFIXES = (
    "[Your active task list",
    "[Planning state preserved",
    "[PRIOR CONTEXT",
    "[CONTEXT COMPRESSED",
    "[IMPORTANT: Background",
)


@dataclass(frozen=True)
class JevScorerConfig:
    """Parsed ``compression.jev_scorer`` block."""

    enabled: bool = False
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass
class JevThinResult:
    """Outcome of an optional Jev thin pass over the compressible window."""

    messages: List[Dict[str, Any]]
    fallback: bool
    n_scored: int = 0
    n_kept: int = 0
    ttft_ms: Optional[float] = None
    ready_ms: Optional[float] = None
    reason: str = ""


def parse_jev_scorer_config(raw: Any) -> JevScorerConfig:
    """Build ``JevScorerConfig`` from a config mapping; unknown/malformed -> defaults."""
    if not isinstance(raw, dict):
        return JevScorerConfig()
    enabled = str(raw.get("enabled", False)).lower() in {"true", "1", "yes"}
    try:
        keep_threshold = float(raw.get("keep_threshold", DEFAULT_KEEP_THRESHOLD))
    except (TypeError, ValueError):
        keep_threshold = DEFAULT_KEEP_THRESHOLD
    try:
        batch_size = int(raw.get("batch_size", DEFAULT_BATCH_SIZE))
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    batch_size = max(1, min(batch_size, 64))
    model = str(raw.get("model") or DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
    try:
        timeout_seconds = float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = max(1.0, timeout_seconds)
    return JevScorerConfig(
        enabled=enabled,
        keep_threshold=keep_threshold,
        batch_size=batch_size,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def resolve_typesafe_api_key() -> str:
    """Read ``TYPESAFE_API_KEY`` via profile secret scope, then process env."""
    from agent.auxiliary_client import _scoped_key_env

    return _scoped_key_env(TYPESAFE_API_KEY_ENV)


def log_jev_scorer(
    *,
    n_scored: int,
    n_kept: int,
    ttft_ms: Optional[float],
    ready_ms: Optional[float],
    fallback: bool,
    reason: str = "",
) -> None:
    """Single measurable line; mirrors ``gateway/run_turn_fast_lane.log_fast_lane`` style."""
    payload = {
        "marker": JEV_SCORER_MARKER,
        "n_scored": int(n_scored),
        "n_kept": int(n_kept),
        "ttft_ms": None if ttft_ms is None else round(float(ttft_ms), 1),
        "ready_ms": None if ready_ms is None else round(float(ready_ms), 1),
        "fallback": bool(fallback),
        "reason": reason or "",
    }
    logger.info(
        "[latency] "
        + JEV_SCORER_MARKER
        + " n_scored=%s n_kept=%s ttft_ms=%s ready_ms=%s fallback=%s reason=%s",
        payload["n_scored"],
        payload["n_kept"],
        payload["ttft_ms"] if payload["ttft_ms"] is not None else "none",
        payload["ready_ms"] if payload["ready_ms"] is not None else "none",
        "true" if fallback else "false",
        payload["reason"] or "ok",
        extra={"jev_scorer": payload},
    )


def _content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def message_to_state_text(msg: Dict[str, Any]) -> str:
    """Flatten one OpenAI-shaped message into a single text state entry for Jev."""
    role = str(msg.get("role") or "unknown")
    text = _content_as_text(msg.get("content")).strip()
    if role == "assistant" and msg.get("tool_calls"):
        calls = []
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or "?"
            args = fn.get("arguments") or ""
            if isinstance(args, (dict, list)):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = str(args)
            args_s = str(args)
            if len(args_s) > 400:
                args_s = args_s[:400] + "..."
            calls.append(f"{name}({args_s})")
        call_block = "; ".join(calls)
        text = f"{text}\n[tool_calls: {call_block}]".strip() if text else f"[tool_calls: {call_block}]"
    if role == "tool":
        tid = msg.get("tool_call_id") or ""
        return f"[tool {tid}] {text}".strip()
    return f"[{role}] {text}".strip()


def _question_for_index(index: int) -> Dict[str, Any]:
    return {
        "type": "score",
        "instructions": f"For state[{index}] only: {SCORE_INSTRUCTIONS}",
        "criteria": list(SCORE_CRITERIA),
    }


def _estimate_payload_chars(state: Sequence[str], questions: Dict[str, Any]) -> int:
    longest_q = max((len(json.dumps(q, ensure_ascii=False)) for q in questions.values()), default=0)
    state_chars = sum(len(s) for s in state)
    return state_chars + longest_q


def truncate_state_texts_to_budget(
    texts: List[str],
    *,
    budget_chars: int = STATE_PLUS_QUESTION_BUDGET_CHARS,
    question_template: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Shrink long tool dumps so state + longest question fit the TypeSafe budget.

    Truncation inserts ``TRUNCATE_INDICATOR`` between head and tail. Never drops entries.
    """
    if not texts:
        return []
    n = len(texts)
    questions = {
        f"m{i}": (question_template or _question_for_index(i)) for i in range(n)
    }
    out = list(texts)
    # First pass: equalize oversized entries while over budget.
    while _estimate_payload_chars(out, questions) > budget_chars:
        lengths = [len(t) for t in out]
        max_len = max(lengths)
        if max_len <= 64:
            # Pathological: still over budget with tiny strings; hard-cap each.
            out = [t[:32] + TRUNCATE_INDICATOR if len(t) > 32 else t for t in out]
            break
        # Truncate the longest entry by half (keep head+tail with indicator).
        idx = lengths.index(max_len)
        text = out[idx]
        keep = max(32, max_len // 2)
        head = keep // 2
        tail = keep - head
        out[idx] = text[:head] + TRUNCATE_INDICATOR + text[-tail:]
    return out


def build_fanout_request(
    messages: Sequence[Dict[str, Any]],
    *,
    model: str = DEFAULT_JEV_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    budget_chars: int = STATE_PLUS_QUESTION_BUDGET_CHARS,
) -> Dict[str, Any]:
    """Build one System One request: state array + one score question per index.

    Caller should slice ``messages`` to at most ``batch_size`` entries.
    """
    if len(messages) > batch_size:
        raise ValueError(f"batch exceeds batch_size={batch_size}: got {len(messages)}")
    raw_state = [message_to_state_text(m) for m in messages]
    state = truncate_state_texts_to_budget(raw_state, budget_chars=budget_chars)
    questions = {f"m{i}": _question_for_index(i) for i in range(len(state))}
    return {
        "model": model,
        "state": state,
        "questions": questions,
    }


def parse_score_answers(payload: Any, n: int) -> List[float]:
    """Extract per-index float scores from a System One response body."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    scores: List[float] = []
    for i in range(n):
        ans = answers.get(f"m{i}")
        if not isinstance(ans, dict):
            raise ValueError(f"missing answer for m{i}")
        # Prefer probability-weighted score; fall back to level index.
        if "score" in ans:
            scores.append(float(ans["score"]))
        elif "level" in ans:
            scores.append(float(ans["level"]))
        else:
            raise ValueError(f"answer m{i} has no score/level")
    return scores


def pinned_active_task_indices(messages: Sequence[Dict[str, Any]]) -> set[int]:
    """Indices that must never drop: goal scaffolding + active-task block.

    The active-task block is the last real user message in the window through the
    end of the window (assistant/tool context answering that request).
    """
    pinned: set[int] = set()
    last_user = -1
    for i, msg in enumerate(messages):
        role = msg.get("role")
        text = _content_as_text(msg.get("content")).lstrip()
        if role == "user" and any(text.startswith(p) for p in _PIN_PREFIXES):
            pinned.add(i)
        if role == "user" and text and not any(text.startswith(p) for p in _PIN_PREFIXES):
            # Skip blank / synthetic echoes for the active-task anchor.
            if not text.startswith(("[System:", "Cronjob Response:", "[ASYNC", "[OUT-OF-BAND")):
                last_user = i
    if last_user >= 0:
        for i in range(last_user, len(messages)):
            pinned.add(i)
    return pinned


def apply_keep_policy(
    messages: Sequence[Dict[str, Any]],
    scores: Sequence[float],
    *,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    pinned: Optional[set[int]] = None,
    span_summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    """Keep score >= threshold (and pinned); replace dropped spans with one-line stubs.

    ``span_summarizer`` should return a one-line replacement for a dropped span
    (existing compression provider). When absent or raising, a deterministic
    stub is used so spans are never silently deleted.
    """
    if len(scores) != len(messages):
        raise ValueError(f"score count {len(scores)} != message count {len(messages)}")
    pin = pinned if pinned is not None else pinned_active_task_indices(messages)
    keep_flags = [
        (i in pin) or (float(scores[i]) >= float(keep_threshold)) for i in range(len(messages))
    ]
    out: List[Dict[str, Any]] = []
    n_kept = 0
    i = 0
    n = len(messages)
    while i < n:
        if keep_flags[i]:
            out.append(dict(messages[i]))
            n_kept += 1
            i += 1
            continue
        # Collect contiguous drop span.
        j = i
        while j < n and not keep_flags[j]:
            j += 1
        span = [dict(m) for m in messages[i:j]]
        stub_text = _one_line_span_stub(span, span_summarizer)
        out.append(
            {
                "role": "user",
                "content": stub_text,
                "_jev_drop_stub": True,
            }
        )
        i = j
    return out, n_kept


def _one_line_span_stub(
    span: List[Dict[str, Any]],
    span_summarizer: Optional[Callable[[List[Dict[str, Any]]], str]],
) -> str:
    if span_summarizer is not None:
        try:
            text = (span_summarizer(span) or "").strip()
            if text:
                # Force a single line for the summarizer input.
                return " ".join(text.split())
        except Exception as exc:
            logger.debug("jev span summarizer failed: %s", type(exc).__name__)
    roles = [str(m.get("role") or "?") for m in span]
    toolish = sum(1 for r in roles if r == "tool")
    return (
        f"[Jev drop stub: {len(span)} message(s) demoted "
        f"(roles={','.join(roles[:8])}{'...' if len(roles) > 8 else ''}; "
        f"tool_results={toolish}). Detail preserved in session history.]"
    )


def _post_systemone(
    body: Dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
    http_client: Any = None,
) -> tuple[Any, float, float]:
    """POST one System One request. Returns (json, ttft_ms, ready_ms)."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    client = http_client
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout_seconds)
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    try:
        resp = client.post(TYPESAFE_SYSTEMONE_URL, headers=headers, json=body)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 429:
            raise RuntimeError("jev rate limited (429)")
        if resp.status_code >= 400:
            raise RuntimeError(f"jev http {resp.status_code}")
        data = resp.json()
        ready_ms = (time.perf_counter() - t0) * 1000.0
        return data, float(ttft_ms), float(ready_ms)
    finally:
        if owns_client:
            client.close()


def score_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    api_key: str,
    cfg: JevScorerConfig,
    http_client: Any = None,
) -> tuple[List[float], float, float]:
    """Score every message via fan-out batches. Raises on any failure."""
    if not api_key:
        raise RuntimeError("missing TYPESAFE_API_KEY")
    if not messages:
        return [], 0.0, 0.0
    all_scores: List[float] = []
    first_ttft: Optional[float] = None
    total_ready = 0.0
    batch = max(1, cfg.batch_size)
    for start in range(0, len(messages), batch):
        chunk = list(messages[start : start + batch])
        body = build_fanout_request(
            chunk, model=cfg.model, batch_size=batch,
        )
        data, ttft_ms, ready_ms = _post_systemone(
            body, api_key=api_key, timeout_seconds=cfg.timeout_seconds, http_client=http_client,
        )
        if first_ttft is None:
            first_ttft = ttft_ms
        total_ready += ready_ms
        all_scores.extend(parse_score_answers(data, len(chunk)))
    return all_scores, float(first_ttft or 0.0), float(total_ready)


def thin_compressible_window(
    messages: Sequence[Dict[str, Any]],
    *,
    cfg: JevScorerConfig,
    api_key: Optional[str] = None,
    http_client: Any = None,
    span_summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> JevThinResult:
    """Score + thin the compressible window, or fall back unchanged.

    Never raises: any error returns ``fallback=True`` with the original messages.
    """
    original = [dict(m) for m in messages]
    if not cfg.enabled:
        return JevThinResult(messages=original, fallback=True, reason="disabled")
    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_scorer(
            n_scored=0, n_kept=0, ttft_ms=None, ready_ms=None, fallback=True, reason="missing_key",
        )
        return JevThinResult(messages=original, fallback=True, reason="missing_key")
    try:
        scores, ttft_ms, ready_ms = score_messages(
            original, api_key=key, cfg=cfg, http_client=http_client,
        )
        pinned = pinned_active_task_indices(original)
        thinned, n_kept = apply_keep_policy(
            original,
            scores,
            keep_threshold=cfg.keep_threshold,
            pinned=pinned,
            span_summarizer=span_summarizer,
        )
        log_jev_scorer(
            n_scored=len(scores),
            n_kept=n_kept,
            ttft_ms=ttft_ms,
            ready_ms=ready_ms,
            fallback=False,
        )
        return JevThinResult(
            messages=thinned,
            fallback=False,
            n_scored=len(scores),
            n_kept=n_kept,
            ttft_ms=ttft_ms,
            ready_ms=ready_ms,
        )
    except Exception as exc:
        reason = type(exc).__name__
        log_jev_scorer(
            n_scored=0, n_kept=0, ttft_ms=None, ready_ms=None, fallback=True, reason=reason,
        )
        logger.info("jev_scorer fallback to existing compression path: %s", reason)
        return JevThinResult(messages=original, fallback=True, reason=reason)
