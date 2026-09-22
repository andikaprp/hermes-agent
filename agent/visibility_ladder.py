"""Ningning-only, never Rancana

Query-aware four-level visibility ladder for the main context-compaction path.

Levels are ``hidden``, ``short``, ``long``, and ``full``. An item is never
deleted: it moves between levels and stays recoverable from the ledger.
``hidden`` means not sent, not dropped from the ledger.

Scoring runs off the fast-lane hot path. ``slow_path_only`` (the default)
scores during main compaction. ``async_precompute`` runs the same decision-model
call on a worker and is joined only by the slow path. Neither mode is invoked
from ``gateway.run_turn_fast_lane``. A fast-lane stack frame refuses the call
before any System One post.

Hard fallback: any Jev error, HTTP 429, timeout, or missing ``TYPESAFE_API_KEY``
returns the original window unchanged. ``fallback=true reason=below_threshold``
is a successful decision (Jev was not confident enough; the unchanged path is
kept), not a failure.

``compression.cache_reuse_decision`` is not implemented here. It stays false.
LAB-52 says never mutate the cached prefix; AGENTS.md says prompt caching is
sacred; measured steady-state prompt-cache hit rate is about 99 percent.

Reload boundary: init-consumed. See ``agent.semantic_pins.RELOAD_BOUNDARY``.
"""

from __future__ import annotations

import copy
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    JevScorerConfig,
    VISIBILITY_LEVELS,
    log_jev_scorer,
    message_to_state_text,
    truncate_state_texts_to_budget,
)
from agent.semantic_pins import (
    PIN_CLASSES,
    PIN_CLASS_SET,
    SemanticPinsConfig,
    explicit_pin_class,
    redact_message_for_compaction,
    split_explicit_pins,
)

logger = logging.getLogger(__name__)

VISIBILITY_LEVEL_SET = frozenset(VISIBILITY_LEVELS)
SCORING_MODES: Tuple[str, ...] = ("slow_path_only", "async_precompute")

_SHORT_CAP = 160
_LONG_CAP = 800
_CONFIDENCE_FLOOR = 0.5

# Fast-lane modules must not gain a decision-model call from this ladder.
_FAST_LANE_MODULE_PREFIXES = (
    "gateway.run_turn_fast_lane",
)


@dataclass(frozen=True)
class VisibilityLadderConfig:
    """Parsed ``compression.visibility_ladder`` block. Default OFF."""

    enabled: bool = False
    scoring: str = "slow_path_only"


@dataclass
class Lab52ApplyResult:
    """Outcome of preparing the compressible window.

    ``fallback`` means the caller must keep the original window and must not
    splice pins. ``turns is original`` on that path.
    """

    turns: List[Dict[str, Any]]
    pinned_verbatim: List[Dict[str, Any]] = field(default_factory=list)
    fallback: bool = False
    reason: str = ""
    applied: bool = False
    skip_binary_thin: bool = False
    n_scored: int = 0
    n_kept: int = 0
    n_pinned: int = 0
    level_chars: Dict[str, int] = field(default_factory=dict)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "", "none"}:
        return False
    return default


def parse_visibility_ladder_config(raw: Any) -> VisibilityLadderConfig:
    """Parse ``compression.visibility_ladder``. Missing or malformed means disabled."""
    if not isinstance(raw, Mapping):
        return VisibilityLadderConfig(enabled=False, scoring="slow_path_only")
    scoring = str(raw.get("scoring") or "slow_path_only").strip()
    if scoring not in SCORING_MODES:
        scoring = "slow_path_only"
    return VisibilityLadderConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        scoring=scoring,
    )


def item_id(message: Mapping[str, Any], index: int = 0) -> str:
    """Stable id for ledger recovery. Explicit ids win over the window index."""
    for key in ("visibility_id", "tool_call_id"):
        raw = message.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return f"idx:{index}:{message.get('role') or ''}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def render_level(text: str, level: str) -> Optional[str]:
    """Sent text at *level*. ``hidden`` returns None (not sent).

    Sent character counts are monotonic: hidden <= short <= long <= full.
    Short is one line. Long is a paragraph. Full is verbatim. Each lower
    level is capped by the next so a stub can never outgrow the verbatim text.
    """
    if level not in VISIBILITY_LEVEL_SET:
        raise ValueError(f"unknown visibility level: {level!r}")
    full = text or ""
    if level == "hidden":
        return None
    if level == "full":
        return full
    one_line = " ".join(full.split())
    short = one_line[:_SHORT_CAP]
    if len(one_line) > _SHORT_CAP:
        short = short.rstrip() + "…"
    long = one_line[:_LONG_CAP]
    if len(one_line) > _LONG_CAP:
        long = long.rstrip() + "…"
    if len(long) > len(full):
        long = full
    if len(short) > len(long):
        short = long
    if level == "short":
        return short
    return long


def sent_char_count(text: str, level: str) -> int:
    rendered = render_level(text, level)
    return 0 if rendered is None else len(rendered)


def render_message(message: Mapping[str, Any], level: str) -> Optional[Dict[str, Any]]:
    """Sent copy of *message* at *level*, or None when hidden.

    Content is redacted at the compaction boundary before the level cap, so a
    ``full`` pin cannot reintroduce a secret span.
    """
    redacted = redact_message_for_compaction(message)
    if level == "hidden":
        return None
    rendered = render_level(_content_text(redacted.get("content")), level)
    if rendered is None:
        return None
    out = copy.deepcopy(redacted)
    out["content"] = rendered
    out["visibility_level"] = level
    return out


class VisibilityLedger:
    """Items move between levels. Nothing is deleted.

    A 2400-line grep can be ``short`` for one question and ``hidden`` for the
    next, and ``recover`` still returns the original message.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def observe(self, messages: Sequence[Mapping[str, Any]]) -> None:
        for index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            iid = item_id(msg, index)
            if iid in self._items:
                continue
            self._items[iid] = {
                "id": iid,
                "message": copy.deepcopy(msg),
                "level": "full",
            }

    def set_level(self, iid: str, level: str) -> None:
        if level not in VISIBILITY_LEVEL_SET:
            raise ValueError(f"unknown visibility level: {level!r}")
        item = self._items.get(iid)
        if item is None:
            raise KeyError(iid)
        item["level"] = level

    def level_of(self, iid: str) -> str:
        return self._items[iid]["level"]

    def recover(self, iid: str) -> Dict[str, Any]:
        return copy.deepcopy(self._items[iid]["message"])

    def count(self) -> int:
        return len(self._items)

    def ids(self) -> List[str]:
        return list(self._items)

    def project(self) -> List[Dict[str, Any]]:
        sent: List[Dict[str, Any]] = []
        for item in self._items.values():
            rendered = render_message(item["message"], item["level"])
            if rendered is not None:
                sent.append(rendered)
        return sent


def current_question(messages: Sequence[Any], focus_topic: Optional[str] = None) -> str:
    """The question visibility is scored against. Focus topic wins, else latest user text."""
    if focus_topic and str(focus_topic).strip():
        return str(focus_topic).strip()
    for msg in reversed(list(messages or [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _content_text(msg.get("content")).strip()
            if text:
                return text
    return ""


def _on_fast_lane_hot_path() -> bool:
    """True when a fast-lane frame is on the stack. No System One post in that case."""
    for frame in inspect.stack():
        name = frame.frame.f_globals.get("__name__", "") or ""
        if name.startswith(_FAST_LANE_MODULE_PREFIXES):
            return True
    return False


def _failure_reason(exc: BaseException) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "below_threshold" in text:
        return "below_threshold"
    if "429" in text or "rate limit" in text:
        return "429"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "missing" in text and "key" in text:
        return "missing_key"
    if "fast_lane_hot_path" in text:
        return "fast_lane_hot_path"
    return "error"


def _zero_levels() -> Dict[str, int]:
    return {level: 0 for level in VISIBILITY_LEVELS}


def _safe_question(question: str) -> str:
    from agent.jev_payload_hygiene import redact_secrets

    return redact_secrets((question or "").strip())[:500]


def _visibility_question(index: int, question: str) -> Dict[str, Any]:
    q = _safe_question(question)
    return {
        "type": "choice",
        "instructions": (
            f"For state[{index}] only, given the current question {q!r}, "
            "which visibility level should this item be sent at? "
            "hidden is not sent. short is one line. long is a paragraph. "
            "full is verbatim. Do not invent a level."
        ),
        "criteria": {
            "hidden": "Not needed to answer the current question; do not send",
            "short": "One line is enough for the current question",
            "long": "A paragraph is needed for the current question",
            "full": "The verbatim text is required for the current question",
        },
    }


def _pin_question(index: int) -> Dict[str, Any]:
    return {
        "type": "choice",
        "instructions": (
            f"For state[{index}] only, which closed pin class applies, or none? "
            "active_task is the current goal, current file, or current error or failing test. "
            "goal_scaffold is acceptance criteria, a named commitment, or a not-done list in force. "
            "standing_instruction is a rule the user stated that is still in force."
        ),
        "criteria": {
            "active_task": "Current goal, current file, or current error or failing test",
            "goal_scaffold": "Acceptance criteria, a named commitment, or a not-done list in force",
            "standing_instruction": "A rule the user stated that is still in force",
            "none": "Not a pin; may be shrunk",
        },
    }


def _confidence(answer: Mapping[str, Any]) -> Optional[float]:
    raw = answer.get("confidence") if "confidence" in answer else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _post_visibility(
    messages: Sequence[Mapping[str, Any]],
    *,
    question: str,
    want_pins: bool,
    model: str,
    timeout_seconds: float,
    http_client: Any,
    api_key: str,
) -> Dict[str, Any]:
    from agent.context_compressor_jev import _post_systemone

    state = truncate_state_texts_to_budget([message_to_state_text(dict(m)) for m in messages])
    questions: Dict[str, Any] = {}
    for index in range(len(state)):
        questions[f"v{index}"] = _visibility_question(index, question)
        if want_pins:
            questions[f"p{index}"] = _pin_question(index)
    body = {"model": model, "state": state, "questions": questions}
    data, _ttft_ms, _ready_ms = _post_systemone(
        body,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    return data


def score_visibility(
    messages: Sequence[Mapping[str, Any]],
    *,
    question: str,
    ladder: VisibilityLadderConfig,
    want_pins: bool,
    jev_cfg: Optional[JevScorerConfig] = None,
    http_client: Any = None,
) -> Tuple[List[str], List[Optional[str]], str]:
    """Score *messages* against *question*. Returns ``(levels, pin_classes, reason)``.

    ``reason`` is empty on success. On failure it is ``error``, ``429``,
    ``timeout``, ``missing_key``, ``below_threshold``, or ``fast_lane_hot_path``.
    Does not post when the fast-lane hot path is on the stack.
    """
    if _on_fast_lane_hot_path():
        raise RuntimeError("fast_lane_hot_path")

    from agent import context_compressor_jev as jev

    api_key = jev.resolve_typesafe_api_key()
    if not api_key:
        raise RuntimeError("missing key")

    cfg = jev_cfg if isinstance(jev_cfg, JevScorerConfig) else JevScorerConfig()
    model = cfg.model or DEFAULT_JEV_MODEL
    timeout_seconds = float(cfg.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    batch_size = max(1, int(cfg.batch_size or 40))

    def _run() -> Tuple[List[str], List[Optional[str]]]:
        levels: List[str] = []
        pins: List[Optional[str]] = []
        rows = list(messages)
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            data = _post_visibility(
                batch,
                question=question,
                want_pins=want_pins,
                model=model,
                timeout_seconds=timeout_seconds,
                http_client=http_client,
                api_key=api_key,
            )
            answers = data.get("answers") if isinstance(data, dict) else None
            if not isinstance(answers, dict):
                raise RuntimeError("jev response missing answers")
            low_confidence = False
            for index in range(len(batch)):
                vis = answers.get(f"v{index}")
                if not isinstance(vis, dict):
                    raise RuntimeError(f"jev response missing v{index}")
                choice = str(vis.get("choice") or "").strip()
                if choice not in VISIBILITY_LEVEL_SET:
                    raise RuntimeError(f"unknown visibility choice: {choice!r}")
                confidence = _confidence(vis)
                if confidence is not None and confidence < _CONFIDENCE_FLOOR:
                    low_confidence = True
                levels.append(choice)
                pin_class: Optional[str] = None
                if want_pins:
                    pin_ans = answers.get(f"p{index}")
                    if isinstance(pin_ans, dict):
                        raw_pin = str(pin_ans.get("choice") or "").strip()
                        if raw_pin in PIN_CLASS_SET:
                            pin_class = raw_pin
                        elif raw_pin not in {"none", ""}:
                            raise RuntimeError(f"unknown pin choice: {raw_pin!r}")
                        pin_conf = _confidence(pin_ans)
                        if pin_conf is not None and pin_conf < _CONFIDENCE_FLOOR:
                            low_confidence = True
                pins.append(pin_class)
            if low_confidence:
                raise RuntimeError("below_threshold")
        return levels, pins

    scoring = ladder.scoring if ladder.scoring in SCORING_MODES else "slow_path_only"
    try:
        if scoring == "async_precompute":
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab52-vis") as pool:
                future = pool.submit(_run)
                levels, pins = future.result(timeout=timeout_seconds)
        else:
            levels, pins = _run()
    except FuturesTimeoutError as exc:
        raise TimeoutError("visibility scoring timeout") from exc
    return levels, pins, ""


def _level_chars_for(messages: Sequence[Mapping[str, Any]], levels: Sequence[str]) -> Dict[str, int]:
    counts = _zero_levels()
    for msg, level in zip(messages, levels):
        if level not in VISIBILITY_LEVEL_SET:
            continue
        text = _content_text(redact_message_for_compaction(msg).get("content"))
        counts[level] = counts.get(level, 0) + sent_char_count(text, level)
    return counts


def _log_outcome(
    *,
    n_scored: int,
    n_kept: int,
    n_pinned: int,
    level_chars: Dict[str, int],
    fallback: bool,
    reason: str,
    model: str = "",
) -> None:
    log_jev_scorer(
        n_scored=n_scored,
        n_kept=n_kept,
        n_pinned=n_pinned,
        level_chars=level_chars,
        ttft_ms=None,
        ready_ms=None,
        fallback=fallback,
        reason=reason,
        model=model,
    )


def prepare_compressible_window(
    turns: List[Dict[str, Any]],
    *,
    pins: SemanticPinsConfig,
    ladder: VisibilityLadderConfig,
    question: str,
    ledger: Optional[VisibilityLedger] = None,
    jev_cfg: Optional[JevScorerConfig] = None,
    http_client: Any = None,
) -> Lab52ApplyResult:
    """Apply pins and/or the visibility ladder to the compressible middle.

    When both features are off this is not called. When the ladder is on, a
    Jev failure returns *turns* unchanged (same list) and an empty splice so
    the caller stays on the original compression path. Explicit pins with the
    ladder off do not call Jev; they are extracted and redacted locally, then
    the existing Jev keep-score thin may still run on the remainder. A pin is
    not in that remainder, so a low keep score cannot stub it.
    """
    pins = pins if isinstance(pins, SemanticPinsConfig) else SemanticPinsConfig()
    ladder = ladder if isinstance(ladder, VisibilityLadderConfig) else VisibilityLadderConfig()
    if not pins.enabled and not ladder.enabled:
        return Lab52ApplyResult(turns=turns, applied=False, reason="disabled")

    if ledger is not None:
        ledger.observe(turns)

    if not ladder.enabled:
        remainder, pinned = split_explicit_pins(turns, pins)
        level_chars = _zero_levels()
        for msg in pinned:
            level_chars["full"] += sent_char_count(_content_text(msg.get("content")), "full")
        if pinned and ledger is not None:
            for msg in pinned:
                iid = item_id(msg)
                if iid in ledger.ids():
                    ledger.set_level(iid, "full")
        _log_outcome(
            n_scored=0,
            n_kept=len(remainder) + len(pinned),
            n_pinned=len(pinned),
            level_chars=level_chars,
            fallback=False,
            reason="pins_only",
        )
        projected = remainder
        if not projected:
            projected = [{"role": "assistant", "content": "(pinned spans retained verbatim outside this summary)"}]
        return Lab52ApplyResult(
            turns=projected,
            pinned_verbatim=pinned,
            fallback=False,
            reason="pins_only",
            applied=True,
            skip_binary_thin=False,
            n_scored=0,
            n_kept=len(remainder) + len(pinned),
            n_pinned=len(pinned),
            level_chars=level_chars,
        )

    # Ladder is on: Jev-dependent. Any failure restores the original list.
    try:
        levels, classified, _reason = score_visibility(
            turns,
            question=question,
            ladder=ladder,
            want_pins=pins.enabled,
            jev_cfg=jev_cfg,
            http_client=http_client,
        )
    except Exception as exc:
        reason = _failure_reason(exc)
        if "fast_lane_hot_path" in str(exc):
            reason = "fast_lane_hot_path"
        _log_outcome(
            n_scored=0,
            n_kept=0,
            n_pinned=0,
            level_chars=_zero_levels(),
            fallback=True,
            reason=reason,
        )
        return Lab52ApplyResult(
            turns=turns,
            pinned_verbatim=[],
            fallback=True,
            reason=reason,
            applied=False,
            skip_binary_thin=False,
        )

    if len(levels) != len(turns):
        _log_outcome(
            n_scored=len(turns),
            n_kept=0,
            n_pinned=0,
            level_chars=_zero_levels(),
            fallback=True,
            reason="error",
        )
        return Lab52ApplyResult(turns=turns, fallback=True, reason="error")

    pinned_verbatim: List[Dict[str, Any]] = []
    projected: List[Dict[str, Any]] = []
    effective_levels: List[str] = []
    allowed_pins = pins.classes if pins.enabled else ()
    for index, msg in enumerate(turns):
        if not isinstance(msg, dict):
            continue
        level = levels[index]
        pin_class = explicit_pin_class(msg, allowed_pins) if pins.enabled else None
        if pin_class is None and pins.enabled:
            classified_name = classified[index] if index < len(classified) else None
            if classified_name in allowed_pins and classified_name in PIN_CLASSES:
                pin_class = classified_name
        # A pin survives verbatim independent of the visibility choice and the
        # keep score. It is spliced, not summarized.
        if pin_class:
            level = "full"
            redacted = redact_message_for_compaction(msg)
            pinned_verbatim.append(redacted)
            effective_levels.append("full")
            if ledger is not None:
                iid = item_id(msg, index)
                if iid in ledger.ids():
                    ledger.set_level(iid, "full")
            continue
        effective_levels.append(level)
        if ledger is not None:
            iid = item_id(msg, index)
            if iid in ledger.ids():
                ledger.set_level(iid, level)
        if level == "hidden":
            continue
        if level == "full":
            pinned_verbatim.append(redact_message_for_compaction(msg))
            continue
        rendered = render_message(msg, level)
        if rendered is not None:
            projected.append(rendered)

    level_chars = _level_chars_for(turns, effective_levels)
    n_pinned = len(pinned_verbatim)
    n_kept = sum(1 for level in effective_levels if level != "hidden")
    _log_outcome(
        n_scored=len(turns),
        n_kept=n_kept,
        n_pinned=n_pinned,
        level_chars=level_chars,
        fallback=False,
        reason="applied",
    )
    if not projected:
        projected = [{"role": "assistant", "content": "(no spans sent at the current visibility level)"}]
    return Lab52ApplyResult(
        turns=projected,
        pinned_verbatim=pinned_verbatim,
        fallback=False,
        reason="applied",
        applied=True,
        skip_binary_thin=True,
        n_scored=len(turns),
        n_kept=n_kept,
        n_pinned=n_pinned,
        level_chars=level_chars,
    )
