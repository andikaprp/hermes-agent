"""Optional TypeSafe Jev skill selection at session setup.

When ``agent.skill_routing.enabled`` is true and ``TYPESAFE_API_KEY`` is set,
a System One choice call picks which skill (if any) to fully load for the
incoming task *before* the session skill set is committed into the system
prompt. Disabled / missing key / empty task / any failure keeps today's
``skills.auto_load`` behavior. Never runs mid-conversation — the auto-load
path resolves once per agent lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    _post_systemone,
    resolve_typesafe_api_key,
)
from agent.jev_payload_hygiene import content_hash, text_metadata

logger = logging.getLogger(__name__)

JEV_SKILL_ROUTE_MARKER = "jev_skill_route"
CONFIG_KEY = "agent.skill_routing"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_CANDIDATES = 40
SKILL_QUESTION_ID = "skill"
NONE_CHOICE = "none"

CHOICE_INSTRUCTIONS = (
    "Which single skill should be fully loaded for this incoming task? "
    "Pick none if no listed skill clearly applies."
)


@dataclass(frozen=True)
class JevSkillRoutingConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SkillCandidate:
    """One Choice option: id is the skill name (or ``none``)."""

    skill_id: str
    description: str
    relevance: str

    def criteria_text(self) -> str:
        desc = (self.description or "").strip() or self.skill_id
        rel = (self.relevance or "").strip()
        if rel:
            return f"{desc} Relevant when: {rel}"
        return desc


def parse_jev_skill_routing_config(raw: Any) -> JevSkillRoutingConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevSkillRoutingConfig()
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
    return JevSkillRoutingConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def load_jev_skill_routing_config(user_config: Any = None) -> JevSkillRoutingConfig:
    """``agent.skill_routing`` from user YAML; default OFF when absent."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly
            user_config = load_config_readonly()
        agent = user_config.get("agent") if isinstance(user_config, dict) else None
        raw = agent.get("skill_routing") if isinstance(agent, dict) else None
        return parse_jev_skill_routing_config(raw)
    except Exception:
        return JevSkillRoutingConfig()


def skill_routing_enabled(user_config: Any = None) -> bool:
    return load_jev_skill_routing_config(user_config).enabled


def _usage_rank(skill_name: str) -> tuple[int, int, str]:
    """Higher view/use counts first; name as stable tiebreaker."""
    try:
        from tools.skill_usage import get_record
        rec = get_record(skill_name) or {}
        views = int(rec.get("view_count") or 0)
        uses = int(rec.get("use_count") or 0)
    except Exception:
        views, uses = 0, 0
    return (-(views + uses), -views, skill_name)


def _relevance_from_frontmatter(frontmatter: Dict[str, Any]) -> str:
    """Prefer ``when_to_use`` list/string; fall back to empty."""
    raw = frontmatter.get("when_to_use")
    if isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
        return "; ".join(parts)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return ""


def collect_top_skill_candidates(
    *,
    home_override: Optional[Path] = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    disabled_names: Optional[set[str]] = None,
) -> List[SkillCandidate]:
    """Top skill candidates (name + one-line description + relevance criteria).

    Ranked by usage when available; capped so the Choice set stays within
    System One cardinality/budget. Always safe to call — empty on any error.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(str(home_override)) if home_override is not None else None
    try:
        from agent.skill_utils import (
            extract_skill_description,
            get_all_skills_dirs,
            get_disabled_skill_names,
            iter_skill_index_files,
            parse_frontmatter,
            skill_matches_platform,
        )

        disabled = disabled_names if disabled_names is not None else get_disabled_skill_names()
        seen: set[str] = set()
        collected: List[SkillCandidate] = []
        for skills_dir in get_all_skills_dirs():
            if not skills_dir.is_dir():
                continue
            for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
                try:
                    frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not skill_matches_platform(frontmatter):
                    continue
                name = str(frontmatter.get("name") or skill_file.parent.name).strip()
                if not name or name in seen or name in disabled or name == NONE_CHOICE:
                    continue
                seen.add(name)
                collected.append(
                    SkillCandidate(
                        skill_id=name,
                        description=extract_skill_description(frontmatter),
                        relevance=_relevance_from_frontmatter(frontmatter),
                    )
                )
        collected.sort(key=lambda c: _usage_rank(c.skill_id))
        return collected[: max(1, int(max_candidates))]
    except Exception:
        logger.debug("skill_routing candidate scan failed", exc_info=True)
        return []
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)


def build_jev_skill_routing_request(
    task_text: str,
    candidates: Sequence[SkillCandidate],
    *,
    model: str = DEFAULT_JEV_MODEL,
) -> dict:
    """System One choice request: hash/metadata for the task, skill ids as criteria."""
    criteria: Dict[str, str] = {
        c.skill_id: c.criteria_text() for c in candidates
    }
    criteria[NONE_CHOICE] = "No listed skill clearly applies to this task"
    return {
        "model": model,
        "state": [
            "Hermes session-setup skill selection. Pick the one skill whose "
            "instructions should be fully loaded into the system prompt for "
            "this new session, or none. The inbound task is hash/metadata only.",
            text_metadata(task_text or "", label="Inbound task"),
        ],
        "questions": {
            SKILL_QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": criteria,
            },
        },
    }


def parse_skill_answer(payload: Any, *, valid_ids: set[str]) -> tuple[str, float]:
    """Extract ``(choice, confidence)`` from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(SKILL_QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {SKILL_QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip()
    if choice not in valid_ids:
        raise ValueError(f"unknown skill choice: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("skill answer missing confidence") from exc
    return choice, confidence


def log_jev_skill_route(
    *,
    chosen: Sequence[str],
    confidence: Optional[float],
    threshold: float,
    fallback: bool,
    reason: str = "",
    model: str = "",
    content_hash_value: str = "",
    latency_ms: Optional[float] = None,
) -> None:
    """Single measurable line: chosen skill ids, confidence, fallback reason."""
    chosen_ids = [str(c) for c in chosen]
    payload = {
        "marker": JEV_SKILL_ROUTE_MARKER,
        "chosen": chosen_ids,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "fallback": bool(fallback),
        "reason": reason or "",
        "model": model or "",
        "content_hash": content_hash_value or "",
        "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
    }
    logger.info(
        "[latency] "
        + JEV_SKILL_ROUTE_MARKER
        + " chosen=%s confidence=%s threshold=%s fallback=%s reason=%s model=%s",
        ",".join(chosen_ids) if chosen_ids else "-",
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        "true" if fallback else "false",
        payload["reason"] or "ok",
        payload["model"] or "-",
        extra={"jev_skill_route": payload},
    )
    try:
        from gateway.jev_observability import record_jev_decision

        record_jev_decision(
            kind="skill_route",
            tier=",".join(chosen_ids) if chosen_ids else "none",
            model=model,
            confidence=confidence,
            latency_ms=latency_ms,
            reason=reason or "ok",
            content_hash_value=content_hash_value,
            fallback=fallback,
        )
    except Exception:
        pass


def _fallback_reason_from_exc(exc: BaseException) -> str:
    reason = type(exc).__name__
    msg = str(exc).strip()
    if "429" in msg:
        return "rate_limited"
    if "timeout" in msg.lower() or "timed out" in msg.lower():
        return "timeout"
    if "http " in msg.lower():
        return msg.replace(" ", "_")[:64] if msg else reason
    return reason


def maybe_override_auto_load_skills(
    task_text: Any,
    *,
    default_names: Sequence[str],
    user_config: Any = None,
    home_override: Optional[Path] = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
    candidates: Optional[Sequence[SkillCandidate]] = None,
) -> Optional[List[str]]:
    """Ask Jev which skill to load, or return ``None`` (keep ``default_names``).

    Returns a replacement name list only when confidence >= threshold AND the
    pick differs from the default loader set. Never raises.
    """
    cfg = load_jev_skill_routing_config(user_config)
    default_list = [str(n).strip() for n in default_names if str(n).strip()]
    if not cfg.enabled:
        return None
    task_hash = content_hash(task_text) if isinstance(task_text, str) else ""
    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_skill_route(
            chosen=default_list,
            confidence=None,
            threshold=cfg.threshold,
            fallback=True,
            reason="missing_key",
            model=cfg.model,
            content_hash_value=task_hash,
        )
        return None
    if not isinstance(task_text, str) or not task_text.strip():
        log_jev_skill_route(
            chosen=default_list,
            confidence=None,
            threshold=cfg.threshold,
            fallback=True,
            reason="empty_task",
            model=cfg.model,
            content_hash_value=task_hash,
        )
        return None

    ready_ms = None
    try:
        cand_list = list(candidates) if candidates is not None else collect_top_skill_candidates(
            home_override=home_override,
        )
        if not cand_list:
            log_jev_skill_route(
                chosen=default_list,
                confidence=None,
                threshold=cfg.threshold,
                fallback=True,
                reason="no_candidates",
                model=cfg.model,
                content_hash_value=task_hash,
            )
            return None
        body = build_jev_skill_routing_request(task_text, cand_list, model=cfg.model)
        valid_ids = {c.skill_id for c in cand_list} | {NONE_CHOICE}
        data, _ttft_ms, ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=cfg.timeout_seconds,
            http_client=http_client,
        )
        choice, confidence = parse_skill_answer(data, valid_ids=valid_ids)
    except Exception as exc:
        reason = _fallback_reason_from_exc(exc)
        log_jev_skill_route(
            chosen=default_list,
            confidence=None,
            threshold=cfg.threshold,
            fallback=True,
            reason=reason,
            model=cfg.model,
            content_hash_value=task_hash,
            latency_ms=ready_ms,
        )
        return None

    if confidence < cfg.threshold:
        log_jev_skill_route(
            chosen=default_list,
            confidence=confidence,
            threshold=cfg.threshold,
            fallback=True,
            reason="below_threshold",
            model=cfg.model,
            content_hash_value=task_hash,
            latency_ms=ready_ms,
        )
        return None

    jev_names: List[str] = [] if choice == NONE_CHOICE else [choice]
    if set(jev_names) == set(default_list):
        log_jev_skill_route(
            chosen=jev_names,
            confidence=confidence,
            threshold=cfg.threshold,
            fallback=False,
            reason="matches_default",
            model=cfg.model,
            content_hash_value=task_hash,
            latency_ms=ready_ms,
        )
        return None

    log_jev_skill_route(
        chosen=jev_names,
        confidence=confidence,
        threshold=cfg.threshold,
        fallback=False,
        reason="",
        model=cfg.model,
        content_hash_value=task_hash,
        latency_ms=ready_ms,
    )
    return jev_names
