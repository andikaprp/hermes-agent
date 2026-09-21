"""Optional TypeSafe Jev action gate for computer/browser use (LAB-60).

Withheld-context consent: Jev picks the next action from a table of actions
already judged safe. Only the goal + short element labels + action descriptions
are sent — never screenshots, page text, or field values. A goal or label that
looks sensitive is refused before the API call.

Jev can only ever return an action id from that table (which must include
``reobserve`` and ``none``). Errors, timeouts, missing keys, and low confidence
fail open to ``reobserve`` so a Jev outage never blocks computer/browser use.

Complements the LAB-55 tool-call risk-gate pattern: choosing an id here does
not bypass existing consequential-action approval / risk gates on the
computer_use and browser tool paths — those still run when the chosen action
is executed.

Config (OFF by default), mirrored under both surfaces:

* ``computer_use.jev_action_gate``
* ``browser.jev_action_gate``
"""

from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    _post_systemone,
    resolve_typesafe_api_key,
)

logger = logging.getLogger(__name__)

JEV_ACTION_GATE_MARKER = "jev_action_gate"
CONFIG_KEYS = ("computer_use.jev_action_gate", "browser.jev_action_gate")
DEFAULT_THRESHOLD = 0.65
DEFAULT_TIMEOUT_SECONDS = 3.0
QUESTION_ID = "next_action"
REOBSERVE_ID = "reobserve"
NONE_ID = "none"
REQUIRED_ACTION_IDS = frozenset({REOBSERVE_ID, NONE_ID})

MAX_CANDIDATES = 32
MAX_REGIONS = 100
MAX_HISTORY = 16
MAX_GOAL_CHARS = 2000
MAX_LABEL_CHARS = 300
MAX_DESCRIPTION_CHARS = 600
MAX_HISTORY_CHARS = 160

_ACTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

CHOICE_INSTRUCTIONS = (
    "Which single pre-approved action should be taken next toward the goal? "
    "Pick only from the provided criteria ids. Prefer reobserve when the right "
    "move is unclear; prefer none when no safe action advances the goal."
)

_SENSITIVE_WORDS = re.compile(
    r"(?i)\b("
    r"password|passwd|passphrase|api[_ -]?key|access[_ -]?token|client[_ -]?secret|"
    r"authorization|bearer|session[_ -]?cookie|credit[_ -]?card|card[_ -]?number|"
    r"\bcvv\b|\bssn\b|social[_ -]?security|private[_ -]?key|secret[_ -]?key|"
    r"\botp\b|one[_ -]?time[_ -]?code|\b2fa\b|\bmfa\b|\bpin\b|"
    r"screenshot|page[_ -]?text|field[_ -]?value|clipboard"
    r")\b"
)
_TOKEN_SHAPES = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"
    r")\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "screenshot", "screenshots", "image", "images", "image_url", "image_b64",
    "png", "jpeg", "page_text", "page_content", "html", "dom", "inner_text",
    "field_value", "field_values", "typed_text", "password",
    "clipboard", "pixels", "frame", "frames",
})


@dataclass(frozen=True)
class JevActionGateConfig:
    """Parsed ``*.jev_action_gate`` block."""

    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class JevActionDecision:
    """Outcome of the optional action gate."""

    action_id: str
    confidence: float
    fallback: bool
    reason: str = ""
    took_jev: bool = False


def parse_jev_action_gate_config(raw: Any) -> JevActionGateConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevActionGateConfig()
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
    return JevActionGateConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def _dig(config: Any, dotted: str) -> Any:
    cur = config
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load_jev_action_gate_config(
    user_config: Any = None,
    *,
    surface: Optional[str] = None,
) -> JevActionGateConfig:
    """Load gate config. ``surface`` is ``computer_use`` / ``browser`` / None (either)."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly

            user_config = load_config_readonly()
        if not isinstance(user_config, dict):
            return JevActionGateConfig()
        if surface in ("computer_use", "browser"):
            keys: Tuple[str, ...] = (f"{surface}.jev_action_gate",)
        else:
            keys = CONFIG_KEYS
        parsed_blocks: List[JevActionGateConfig] = []
        for key in keys:
            raw = _dig(user_config, key)
            if raw is None:
                continue
            parsed = parse_jev_action_gate_config(raw)
            if parsed.enabled:
                return parsed
            parsed_blocks.append(parsed)
        return parsed_blocks[0] if parsed_blocks else JevActionGateConfig()
    except Exception:
        return JevActionGateConfig()


def normalize_text(text: str) -> str:
    """Fold look-alikes / invisible chars so the sensitive gate cannot be dodged."""
    folded = unicodedata.normalize("NFKC", text or "")
    return "".join(
        c for c in folded
        if unicodedata.category(c) not in {"Cf", "Cc"} or c in "\n\t"
    )


def looks_sensitive(text: Any) -> bool:
    """True when *text* must not be sent to Jev (credentials / PII shapes / withheld words)."""
    if not isinstance(text, str) or not text.strip():
        return False
    probe = normalize_text(text)
    return bool(
        _SENSITIVE_WORDS.search(probe)
        or _TOKEN_SHAPES.search(probe)
        or _EMAIL.search(probe)
    )


def assert_no_withheld_payload(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if the request carries screenshots / page text / field values."""
    bad = sorted(k for k in payload if str(k).lower() in _FORBIDDEN_PAYLOAD_KEYS)
    if bad:
        raise ValueError(f"withheld-context violation: forbidden keys {bad}")


def _safe_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    if looks_sensitive(text):
        raise ValueError(f"{name} looks sensitive and will not be sent")
    return text


def _normalize_candidates(candidates: Sequence[Any]) -> Dict[str, str]:
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"candidates must hold at most {MAX_CANDIDATES} actions")
    table: Dict[str, str] = {}
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("each candidate must be a mapping with id and description")
        action_id = item.get("id")
        if not isinstance(action_id, str) or not _ACTION_ID_RE.fullmatch(action_id):
            raise ValueError(
                "candidate ids must match [A-Za-z0-9][A-Za-z0-9._:-]{0,63}"
            )
        if action_id in table:
            raise ValueError(f"duplicate candidate id: {action_id}")
        table[action_id] = _safe_text(
            item.get("description"),
            f"candidate {action_id} description",
            MAX_DESCRIPTION_CHARS,
        )
    missing = REQUIRED_ACTION_IDS - set(table)
    if missing:
        raise ValueError(
            f"candidates must include required ids {sorted(REQUIRED_ACTION_IDS)}; "
            f"missing {sorted(missing)}"
        )
    if len(table) < 2:
        raise ValueError("candidates must hold at least 2 actions")
    return table


def _normalize_regions(regions: Any) -> List[Dict[str, Any]]:
    if regions is None:
        return []
    if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
        raise ValueError(f"regions must be a list of at most {MAX_REGIONS}")
    out: List[Dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("each region must be a mapping")
        allowed = {"id", "role", "label", "interactive"}
        if set(region) - allowed:
            raise ValueError("a region may contain only id, role, label, interactive")
        out.append({
            "id": _safe_text(region.get("id"), "region id", 128),
            "role": _safe_text(region.get("role", "element"), "region role", 64),
            "label": _safe_text(region.get("label"), "region label", MAX_LABEL_CHARS),
            "interactive": bool(region.get("interactive", False)),
        })
    return out


def _normalize_history(history: Any) -> List[Dict[str, str]]:
    if history is None:
        return []
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        raise ValueError(f"history must be a list of at most {MAX_HISTORY}")
    out: List[Dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, Mapping):
            raise ValueError("each history item must be a mapping")
        allowed = {"selected_id", "outcome"}
        if set(entry) - allowed:
            raise ValueError("a history item may contain only selected_id and outcome")
        cleaned: Dict[str, str] = {}
        for key in ("selected_id", "outcome"):
            if key in entry and entry.get(key) is not None:
                cleaned[key] = _safe_text(
                    entry.get(key), f"history {key}", MAX_HISTORY_CHARS,
                )
        out.append(cleaned)
    return out


def build_action_choice_request(
    *,
    goal: str,
    candidates: Sequence[Mapping[str, Any]],
    regions: Optional[Sequence[Mapping[str, Any]]] = None,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    observation_id: str = "",
    model: str = DEFAULT_JEV_MODEL,
) -> Dict[str, Any]:
    """Build a System One choice request under the withheld-context contract.

    Raises ``ValueError`` on sensitive text or a bad action table.
    """
    table = _normalize_candidates(candidates)
    clean_regions = _normalize_regions(regions)
    clean_history = _normalize_history(history)
    goal_text = _safe_text(goal, "goal", MAX_GOAL_CHARS)
    state: List[str] = [
        "Hermes computer/browser action gate. Choose only from the provided action ids. "
        "You never receive screen captures, page text, or field values.",
        f"Goal:\n{goal_text}",
    ]
    if clean_regions:
        lines = [
            f"- [{r['id']}] {r['role']}: {r['label']}"
            + (" (interactive)" if r["interactive"] else "")
            for r in clean_regions
        ]
        state.append("On-screen element labels (short only):\n" + "\n".join(lines))
    if clean_history:
        lines = [
            f"- {e.get('selected_id', '?')}: {e.get('outcome', '')}".rstrip(": ")
            for e in clean_history
        ]
        state.append("Recent choices:\n" + "\n".join(lines))
    if observation_id:
        state.append(f"observation_id={str(observation_id)[:256]}")
    return {
        "model": model,
        "state": state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(table),
            },
        },
    }


def parse_action_answer(payload: Any, allowed_ids: Mapping[str, Any]) -> Tuple[str, float]:
    """Extract ``(action_id, confidence)`` constrained to *allowed_ids*. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip()
    if choice not in allowed_ids:
        raise ValueError(f"action id not in approved table: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("action answer missing confidence") from exc
    return choice, confidence


def log_jev_action_gate(
    *,
    action_id: str,
    confidence: Optional[float],
    threshold: float,
    took_jev: bool,
    fallback: bool,
    reason: str = "",
    model: str = "",
    surface: str = "",
) -> None:
    payload = {
        "marker": JEV_ACTION_GATE_MARKER,
        "action_id": action_id or "",
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "took_jev": bool(took_jev),
        "fallback": bool(fallback),
        "reason": reason or "",
        "model": model or "",
        "surface": surface or "",
    }
    logger.info(
        "[latency] "
        + JEV_ACTION_GATE_MARKER
        + " surface=%s action_id=%s confidence=%s threshold=%s took_jev=%s "
        "fallback=%s reason=%s model=%s",
        payload["surface"] or "-",
        payload["action_id"] or "-",
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        "true" if took_jev else "false",
        "true" if fallback else "false",
        payload["reason"] or "ok",
        payload["model"] or "-",
        extra={"jev_action_gate": payload},
    )


def _reobserve(
    *,
    reason: str,
    confidence: float = 0.0,
    fallback: bool = True,
    took_jev: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = "",
    surface: str = "",
) -> JevActionDecision:
    log_jev_action_gate(
        action_id=REOBSERVE_ID,
        confidence=confidence if took_jev else None,
        threshold=threshold,
        took_jev=took_jev,
        fallback=fallback,
        reason=reason,
        model=model,
        surface=surface,
    )
    return JevActionDecision(
        action_id=REOBSERVE_ID,
        confidence=confidence,
        fallback=fallback,
        reason=reason,
        took_jev=took_jev,
    )


def choose_next_action(
    *,
    goal: str,
    candidates: Sequence[Mapping[str, Any]],
    regions: Optional[Sequence[Mapping[str, Any]]] = None,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    observation_id: str = "",
    cfg: Optional[JevActionGateConfig] = None,
    user_config: Any = None,
    surface: Optional[str] = None,
    api_key: Optional[str] = None,
    http_client: Any = None,
    request_payload: Optional[Mapping[str, Any]] = None,
) -> JevActionDecision:
    """Ask Jev for the next action id, or return ``reobserve`` (never raises).

    When disabled, returns ``reobserve`` with reason ``disabled`` without an API call.
    Callers that should act freely when the gate is off should check ``cfg.enabled`` first.
    """
    parsed = cfg if cfg is not None else load_jev_action_gate_config(
        user_config, surface=surface,
    )
    surface_name = surface or ""
    if not parsed.enabled:
        return _reobserve(
            reason="disabled",
            fallback=False,
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )

    if request_payload is not None:
        try:
            assert_no_withheld_payload(request_payload)
        except ValueError as exc:
            return _reobserve(
                reason=str(exc)[:64] or "withheld_payload",
                threshold=parsed.threshold,
                model=parsed.model,
                surface=surface_name,
            )

    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        return _reobserve(
            reason="missing_key",
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )

    try:
        body = build_action_choice_request(
            goal=goal,
            candidates=candidates,
            regions=regions,
            history=history,
            observation_id=observation_id,
            model=parsed.model,
        )
        table = body["questions"][QUESTION_ID]["criteria"]
    except ValueError as exc:
        return _reobserve(
            reason=str(exc)[:80] or "invalid_request",
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(
                _post_systemone,
                body,
                api_key=key,
                timeout_seconds=parsed.timeout_seconds,
                http_client=http_client,
            )
            data, _ttft, _ready = fut.result(timeout=parsed.timeout_seconds)
        choice, confidence = parse_action_answer(data, table)
    except FuturesTimeoutError:
        return _reobserve(
            reason="timeout",
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )
    except Exception as exc:
        reason = type(exc).__name__
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = "timeout"
        return _reobserve(
            reason=reason,
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )

    if confidence < parsed.threshold:
        return _reobserve(
            reason="below_threshold",
            confidence=confidence,
            took_jev=True,
            threshold=parsed.threshold,
            model=parsed.model,
            surface=surface_name,
        )

    log_jev_action_gate(
        action_id=choice,
        confidence=confidence,
        threshold=parsed.threshold,
        took_jev=True,
        fallback=False,
        reason="ok",
        model=parsed.model,
        surface=surface_name,
    )
    return JevActionDecision(
        action_id=choice,
        confidence=confidence,
        fallback=False,
        reason="ok",
        took_jev=True,
    )


def decision_payload(decision: JevActionDecision) -> Dict[str, Any]:
    """JSON-serializable body for tool results / logs."""
    return {
        "action_id": decision.action_id,
        "confidence": decision.confidence,
        "fallback": decision.fallback,
        "reason": decision.reason,
        "took_jev": decision.took_jev,
        "reobserve": decision.action_id == REOBSERVE_ID,
        "none": decision.action_id == NONE_ID,
    }
