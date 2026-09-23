"""Optional TypeSafe Jev completion verification (Canny-style) before delivery.

Runs AFTER the agent returns a turn result and BEFORE the gateway claims the
turn completed/delivered. Scores ``done`` / ``partial`` / ``failed`` with
confidence from result metadata + existing verification evidence hashes — never
raw prompts.

Disabled by default. Missing key / HTTP / timeout / parse errors fail open
(current behavior). Low confidence, evidence contradiction, or a confident
non-done verdict escalates to HITL (never auto-claims done). Successful
verdicts are recorded in ``verification_evidence.db``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _post_systemone,
    resolve_typesafe_api_key,
)
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

JEV_COMPLETION_MARKER = "jev_completion"
CONFIG_KEY = "gateway.jev_completion"
DEFAULT_THRESHOLD = 0.75
COMPLETION_QUESTION_ID = "completion"
HITL_NOTICE = (
    "⚠️ Completion check needs a human look before this turn is marked done "
    "(verdict={verdict}, confidence={confidence}). Reply to confirm or correct."
)

CHOICE_INSTRUCTIONS = (
    "Did this agent turn fully complete the user's request, only partially "
    "complete it, or fail to complete it?"
)
CHOICE_CRITERIA = {
    "done": "The turn fully completed the requested work with supporting evidence",
    "partial": "Some work landed but the request is not fully satisfied",
    "failed": "The turn did not complete the requested work",
}


@dataclass(frozen=True)
class JevCompletionConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CompletionVerdict:
    """Outcome of optional completion scoring."""

    claim_done: bool
    escalate: bool
    verdict: str
    confidence: Optional[float]
    fallback: bool
    reason: str = ""
    result_hash: str = ""
    evidence_status: str = ""
    contradicted: bool = False


def parse_jev_completion_config(raw: Any) -> JevCompletionConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevCompletionConfig()
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
    return JevCompletionConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def load_jev_completion_config(user_config: Any = None) -> JevCompletionConfig:
    """``gateway.jev_completion`` from user YAML; default OFF when absent."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly

            user_config = load_config_readonly()
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        raw = gw.get("jev_completion") if isinstance(gw, dict) else None
        return parse_jev_completion_config(raw)
    except Exception:
        return JevCompletionConfig()


def result_payload_hash(agent_result: Any) -> str:
    """Stable short hash of completion-relevant result fields (never raw text)."""
    if not isinstance(agent_result, dict):
        return hashlib.sha256(b"").hexdigest()[:16]
    final = agent_result.get("final_response")
    final_len = len(final) if isinstance(final, str) else 0
    blob = json.dumps(
        {
            "completed": agent_result.get("completed"),
            "failed": bool(agent_result.get("failed")),
            "partial": bool(agent_result.get("partial")),
            "interrupted": bool(agent_result.get("interrupted")),
            "api_calls": int(agent_result.get("api_calls") or 0),
            "response_chars": final_len,
            "has_error": bool(agent_result.get("error")),
            "failure_reason": str(agent_result.get("failure_reason") or "")[:64],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:16]


def _evidence_snapshot(
    *,
    session_id: Optional[str],
    cwd: Any = None,
) -> dict[str, Any]:
    """Read existing verification ledger metadata (no raw command dumps)."""
    try:
        from agent.verification_evidence import verification_status

        status = verification_status(session_id=session_id, cwd=cwd)
    except Exception:
        return {"status": "unavailable"}
    if not isinstance(status, dict):
        return {"status": "unavailable"}
    evidence = status.get("evidence") if isinstance(status.get("evidence"), dict) else None
    snap: dict[str, Any] = {
        "status": str(status.get("status") or "unknown"),
        "root_hash": "",
        "kind": "",
        "scope": "",
        "exit_code": None,
    }
    root = status.get("root")
    if root:
        snap["root_hash"] = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    if evidence:
        snap["kind"] = str(evidence.get("kind") or "")[:32]
        snap["scope"] = str(evidence.get("scope") or "")[:16]
        try:
            snap["exit_code"] = int(evidence.get("exit_code"))
        except (TypeError, ValueError):
            snap["exit_code"] = None
        # Hash of ledger row identity — never the raw command/output.
        row_blob = json.dumps(
            {
                "id": evidence.get("id"),
                "kind": evidence.get("kind"),
                "status": evidence.get("status"),
                "exit_code": evidence.get("exit_code"),
                "canonical_command": evidence.get("canonical_command"),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8", errors="replace")
        snap["evidence_hash"] = hashlib.sha256(row_blob).hexdigest()[:16]
    return snap


def evidence_contradicts_done(evidence_status: str) -> bool:
    """True when claiming ``done`` would fight the ledger."""
    return evidence_status in {"failed", "stale"}


def build_completion_request(
    *,
    result_hash: str,
    agent_result: dict,
    evidence: dict[str, Any],
    model: str = DEFAULT_JEV_MODEL,
) -> dict:
    """System One choice over hashes/metadata only (no raw prompts/replies)."""
    state = [
        "LAB-61 Canny-style completion verification. "
        "Judge only from the hashed result metadata and verification ledger snapshot below. "
        "Never assume unseen prompt text.",
        (
            f"result_hash={result_hash} "
            f"completed={agent_result.get('completed')!r} "
            f"failed={bool(agent_result.get('failed'))} "
            f"partial={bool(agent_result.get('partial'))} "
            f"interrupted={bool(agent_result.get('interrupted'))} "
            f"api_calls={int(agent_result.get('api_calls') or 0)} "
            f"response_chars={len(agent_result.get('final_response') or '') if isinstance(agent_result.get('final_response'), str) else 0} "
            f"has_error={bool(agent_result.get('error'))}"
        ),
        (
            f"evidence_status={evidence.get('status')} "
            f"evidence_kind={evidence.get('kind') or '-'} "
            f"evidence_scope={evidence.get('scope') or '-'} "
            f"evidence_exit={evidence.get('exit_code')!r} "
            f"evidence_hash={evidence.get('evidence_hash') or '-'} "
            f"root_hash={evidence.get('root_hash') or '-'}"
        ),
    ]
    return {
        "model": model,
        "state": state,
        "questions": {
            COMPLETION_QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            },
        },
    }


def parse_completion_answer(payload: Any) -> tuple[str, float]:
    """Extract ``(choice, confidence)`` from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(COMPLETION_QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {COMPLETION_QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip().lower()
    if choice not in CHOICE_CRITERIA:
        raise ValueError(f"unknown completion choice: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("completion answer missing confidence") from exc
    return choice, confidence


def log_jev_completion(
    *,
    decision: str,
    model: str,
    choice: str = "",
    confidence: Optional[float] = None,
    threshold: float = DEFAULT_THRESHOLD,
    chat_id: Any = None,
    reason: str = "",
    result_hash: str = "",
    evidence_status: str = "",
    contradicted: bool = False,
) -> None:
    payload = {
        "marker": JEV_COMPLETION_MARKER,
        "decision": decision,
        "model": model,
        "choice": choice or "",
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "reason": reason or "",
        "result_hash": result_hash or "",
        "evidence_status": evidence_status or "",
        "contradicted": bool(contradicted),
    }
    logger.info(
        "[latency] "
        + JEV_COMPLETION_MARKER
        + " chat=%s decision=%s model=%s choice=%s confidence=%s threshold=%s "
        "result_hash=%s evidence=%s contradicted=%s reason=%s",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX),
        payload["decision"],
        payload["model"],
        payload["choice"] or "-",
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        payload["result_hash"] or "-",
        payload["evidence_status"] or "-",
        "true" if contradicted else "false",
        payload["reason"] or "ok",
        extra={"jev_completion": payload},
    )


def _record_verdict(
    *,
    session_id: Optional[str],
    cwd: Any,
    verdict: str,
    confidence: Optional[float],
    result_hash: str,
    evidence_status: str,
    decision: str,
    reason: str,
) -> None:
    try:
        from agent.verification_evidence import record_jev_completion_verdict

        record_jev_completion_verdict(
            session_id=session_id,
            cwd=cwd,
            verdict=verdict,
            confidence=confidence,
            result_hash=result_hash,
            evidence_status=evidence_status,
            decision=decision,
            reason=reason,
        )
    except Exception as exc:
        logger.debug("jev_completion evidence write failed: %s", exc)


def evaluate_turn_completion(
    agent_result: Any,
    *,
    user_config: Any = None,
    chat_id: Any = None,
    session_id: Optional[str] = None,
    cwd: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
    cfg: Optional[JevCompletionConfig] = None,
) -> CompletionVerdict:
    """Score turn completion; fail-open returns ``claim_done=True`` without escalate.

    Escalation (HITL): low confidence, evidence contradiction vs ``done``, or a
    confident ``partial``/``failed`` verdict. Errors/timeouts/disabled do not escalate.
    """
    gate = cfg if cfg is not None else load_jev_completion_config(user_config)
    if not gate.enabled:
        return CompletionVerdict(
            claim_done=True, escalate=False, verdict="", confidence=None,
            fallback=True, reason="disabled",
        )

    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_completion(
            decision="passthrough", model=gate.model, threshold=gate.threshold,
            chat_id=chat_id, reason="missing_key",
        )
        return CompletionVerdict(
            claim_done=True, escalate=False, verdict="", confidence=None,
            fallback=True, reason="missing_key",
        )

    if not isinstance(agent_result, dict):
        log_jev_completion(
            decision="passthrough", model=gate.model, threshold=gate.threshold,
            chat_id=chat_id, reason="bad_result",
        )
        return CompletionVerdict(
            claim_done=True, escalate=False, verdict="", confidence=None,
            fallback=True, reason="bad_result",
        )

    # Already failed / interrupted turns keep current delivery shaping.
    if agent_result.get("failed") or agent_result.get("interrupted"):
        log_jev_completion(
            decision="passthrough", model=gate.model, threshold=gate.threshold,
            chat_id=chat_id, reason="already_terminal",
            result_hash=result_payload_hash(agent_result),
        )
        return CompletionVerdict(
            claim_done=True, escalate=False, verdict="", confidence=None,
            fallback=True, reason="already_terminal",
            result_hash=result_payload_hash(agent_result),
        )

    rhash = result_payload_hash(agent_result)
    evidence = _evidence_snapshot(session_id=session_id, cwd=cwd)
    ev_status = str(evidence.get("status") or "unknown")

    try:
        body = build_completion_request(
            result_hash=rhash, agent_result=agent_result, evidence=evidence, model=gate.model,
        )
        data, _ttft_ms, _ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=gate.timeout_seconds,
            http_client=http_client,
        )
        choice, confidence = parse_completion_answer(data)
    except Exception as exc:
        reason = type(exc).__name__
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = "timeout"
        elif "http " in msg.lower():
            reason = msg.replace(" ", "_")[:64] if msg else reason
        log_jev_completion(
            decision="passthrough", model=gate.model, threshold=gate.threshold,
            chat_id=chat_id, reason=reason, result_hash=rhash, evidence_status=ev_status,
        )
        return CompletionVerdict(
            claim_done=True, escalate=False, verdict="", confidence=None,
            fallback=True, reason=reason, result_hash=rhash, evidence_status=ev_status,
        )

    below = confidence < gate.threshold
    contradicted = choice == "done" and evidence_contradicts_done(ev_status)
    escalate = below or contradicted or choice in {"partial", "failed"}
    claim_done = (not escalate) and choice == "done"
    if claim_done:
        decision = "claim_done"
        reason = ""
    elif below:
        decision = "escalate"
        reason = "below_threshold"
    elif contradicted:
        decision = "escalate"
        reason = "evidence_contradiction"
    else:
        decision = "escalate"
        reason = f"verdict_{choice}"

    log_jev_completion(
        decision=decision,
        model=gate.model,
        choice=choice,
        confidence=confidence,
        threshold=gate.threshold,
        chat_id=chat_id,
        reason=reason,
        result_hash=rhash,
        evidence_status=ev_status,
        contradicted=contradicted,
    )
    _record_verdict(
        session_id=session_id,
        cwd=cwd,
        verdict=choice,
        confidence=confidence,
        result_hash=rhash,
        evidence_status=ev_status,
        decision=decision,
        reason=reason,
    )
    return CompletionVerdict(
        claim_done=claim_done,
        escalate=escalate,
        verdict=choice,
        confidence=confidence,
        fallback=False,
        reason=reason,
        result_hash=rhash,
        evidence_status=ev_status,
        contradicted=contradicted,
    )


def apply_completion_verdict(agent_result: dict, verdict: CompletionVerdict) -> dict:
    """Mutate a turn result so HITL escalation never auto-claims done.

    Returns the same dict. Fail-open / claim_done leave flags alone.
    """
    if not isinstance(agent_result, dict) or verdict.fallback or not verdict.escalate:
        if isinstance(agent_result, dict) and verdict.claim_done and not verdict.fallback:
            agent_result["jev_completion"] = {
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "claim_done": True,
                "escalate": False,
                "result_hash": verdict.result_hash,
                "evidence_status": verdict.evidence_status,
            }
        return agent_result

    agent_result["completed"] = False
    agent_result["partial"] = True
    # Read by gateway.jev_completion_hitl (tools.clarify_gateway ask). This
    # module only sets the flag; it does not deliver the question.
    agent_result["hitl_escalation"] = True
    agent_result["jev_completion"] = {
        "verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "claim_done": False,
        "escalate": True,
        "reason": verdict.reason,
        "result_hash": verdict.result_hash,
        "evidence_status": verdict.evidence_status,
        "contradicted": verdict.contradicted,
    }
    conf = (
        f"{verdict.confidence:.2f}"
        if isinstance(verdict.confidence, (int, float))
        else "n/a"
    )
    notice = HITL_NOTICE.format(verdict=verdict.verdict or "unknown", confidence=conf)
    final = agent_result.get("final_response")
    if isinstance(final, str) and final.strip():
        if notice not in final:
            agent_result["final_response"] = f"{final.rstrip()}\n\n{notice}"
    else:
        agent_result["final_response"] = notice
    return agent_result


def maybe_verify_turn_completion(
    agent_result: Any,
    *,
    user_config: Any = None,
    chat_id: Any = None,
    session_id: Optional[str] = None,
    cwd: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
) -> Any:
    """Config-gated entry: score + apply HITL flags; identity on disabled/fail-open."""
    if not isinstance(agent_result, dict):
        return agent_result
    cfg = load_jev_completion_config(user_config)
    if not cfg.enabled:
        return agent_result
    verdict = evaluate_turn_completion(
        agent_result,
        user_config=user_config,
        chat_id=chat_id,
        session_id=session_id,
        cwd=cwd,
        http_client=http_client,
        api_key=api_key,
        cfg=cfg,
    )
    return apply_completion_verdict(agent_result, verdict)
