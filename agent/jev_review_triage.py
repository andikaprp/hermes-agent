"""Optional TypeSafe Jev code-review triage (jev-review style).

When ``code_review.jev_triage.enabled`` is true and ``TYPESAFE_API_KEY`` is set,
grades a review comment set plus PR diff *metadata* into a verdict:
``approve`` / ``request-changes`` / ``needs-human``, with confidence.

Safety contracts (hard):
- confidence below threshold ALWAYS becomes ``needs-human`` (never auto-approve)
- missing key / API failure / empty payload -> ``needs-human`` (safe escape)
- this layer never submits a GitHub ``APPROVE`` review event; it posts at most
  one commentary comment per PR (marker-deduped) so the human review gate stays
- disabled by default; callers skip posting when ``should_post`` is False

State shape follows the hashes/metadata pattern: file paths, patch hashes,
add/del counts, comment text, and a bounded diff excerpt — not raw full-file dumps.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.context_compressor_jev import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    TRUNCATE_INDICATOR,
    _post_systemone,
    resolve_typesafe_api_key,
)

logger = logging.getLogger(__name__)

JEV_REVIEW_TRIAGE_MARKER = "jev_review_triage"
VERDICT_COMMENT_MARKER = "<!-- hermes-jev-review-triage -->"
CONFIG_KEY = "code_review.jev_triage"
DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_DIFF_CHARS = 12_000
VERDICT_QUESTION_ID = "verdict"

VERDICTS = ("approve", "request-changes", "needs-human")

CHOICE_INSTRUCTIONS = (
    "Given this PR's diff metadata and review comment set, what should the "
    "review outcome be? Prefer needs-human when evidence is thin, conflicting, "
    "or high-stakes."
)
CHOICE_CRITERIA = {
    "approve": (
        "Safe to approve: comments are non-blocking or empty, diff metadata "
        "shows a focused change, no unresolved critical/security findings."
    ),
    "request-changes": (
        "Blocking defects remain: critical/warning findings in comments, "
        "incorrect or incomplete change relative to the stated PR intent."
    ),
    "needs-human": (
        "Defer to a human reviewer: low confidence, conflicting signals, "
        "security-sensitive surface, or insufficient evidence in metadata/comments."
    ),
}


@dataclass(frozen=True)
class JevReviewTriageConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_JEV_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS


@dataclass(frozen=True)
class FileDiffMeta:
    """One changed file — hashes + counts, not the full file body."""

    path: str
    patch_sha256: str = ""
    additions: int = 0
    deletions: int = 0
    status: str = "modified"


@dataclass(frozen=True)
class ReviewComment:
    """One review finding (inline or summary)."""

    body: str
    path: str = ""
    line: Optional[int] = None
    severity: str = ""


@dataclass(frozen=True)
class JevReviewTriageResult:
    """Outcome of optional review triage."""

    verdict: str  # approve | request-changes | needs-human
    confidence: Optional[float] = None
    reason: str = ""
    fallback: bool = False
    should_post: bool = True
    model: str = ""
    raw_choice: str = ""

    @property
    def is_approve(self) -> bool:
        return self.verdict == "approve"


@dataclass
class ReviewTriageInput:
    """Structured PR review payload for Jev (metadata + comments + bounded diff)."""

    pr_number: int = 0
    title: str = ""
    head_sha: str = ""
    base_ref: str = ""
    files: List[FileDiffMeta] = field(default_factory=list)
    comments: List[ReviewComment] = field(default_factory=list)
    diff_excerpt: str = ""


def parse_jev_review_triage_config(raw: Any) -> JevReviewTriageConfig:
    """Build config from a mapping; unknown/malformed -> defaults (disabled)."""
    if not isinstance(raw, dict):
        return JevReviewTriageConfig()
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
        max_diff_chars = int(raw.get("max_diff_chars", DEFAULT_MAX_DIFF_CHARS))
    except (TypeError, ValueError):
        max_diff_chars = DEFAULT_MAX_DIFF_CHARS
    max_diff_chars = max(1_000, max_diff_chars)
    return JevReviewTriageConfig(
        enabled=enabled,
        threshold=threshold,
        model=model,
        timeout_seconds=timeout_seconds,
        max_diff_chars=max_diff_chars,
    )


def load_jev_review_triage_config(user_config: Any = None) -> JevReviewTriageConfig:
    """``code_review.jev_triage`` from config; default OFF when absent."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly

            user_config = load_config_readonly()
        block = user_config.get("code_review") if isinstance(user_config, dict) else None
        raw = block.get("jev_triage") if isinstance(block, dict) else None
        return parse_jev_review_triage_config(raw)
    except Exception:
        return JevReviewTriageConfig()


def patch_sha256(patch_text: str) -> str:
    """Stable short hash of a unified-diff patch (empty patch -> empty hash)."""
    text = (patch_text or "").encode("utf-8", errors="replace")
    if not text:
        return ""
    return hashlib.sha256(text).hexdigest()[:16]


def file_meta_from_patch(
    path: str,
    patch_text: str,
    *,
    additions: int = 0,
    deletions: int = 0,
    status: str = "modified",
) -> FileDiffMeta:
    """Build FileDiffMeta from a patch string (hash the patch, not the whole file)."""
    return FileDiffMeta(
        path=path,
        patch_sha256=patch_sha256(patch_text),
        additions=max(0, int(additions)),
        deletions=max(0, int(deletions)),
        status=status or "modified",
    )


def _truncate_diff(diff_text: str, max_chars: int) -> str:
    text = (diff_text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + TRUNCATE_INDICATOR + text[-tail:]


def build_review_state(
    payload: ReviewTriageInput,
    *,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> List[str]:
    """Flatten PR metadata + comments + bounded diff into System One state entries."""
    header_lines = [
        f"PR #{payload.pr_number}: {payload.title or '(untitled)'}".strip(),
        f"head_sha={payload.head_sha or 'unknown'} base_ref={payload.base_ref or 'unknown'}",
    ]
    file_lines: List[str] = ["Changed files (path + patch hash + churn):"]
    if payload.files:
        for f in payload.files:
            file_lines.append(
                f"- {f.path} status={f.status} "
                f"+{f.additions}/-{f.deletions} patch_sha256={f.patch_sha256 or 'none'}"
            )
    else:
        file_lines.append("- (none listed)")

    comment_lines: List[str] = ["Review comments:"]
    if payload.comments:
        for i, c in enumerate(payload.comments, start=1):
            loc = c.path or "(summary)"
            if c.line is not None:
                loc = f"{loc}:{c.line}"
            sev = f" [{c.severity}]" if c.severity else ""
            body = (c.body or "").strip().replace("\n", " ")
            if len(body) > 500:
                body = body[:500] + "…"
            comment_lines.append(f"{i}. {loc}{sev}: {body}")
    else:
        comment_lines.append("- (no comments)")

    state = [
        "\n".join(header_lines),
        "\n".join(file_lines),
        "\n".join(comment_lines),
    ]
    diff = _truncate_diff(payload.diff_excerpt, max_diff_chars)
    if diff:
        state.append(f"Diff excerpt (bounded; not full files):\n{diff}")
    return state


def build_jev_review_triage_request(
    payload: ReviewTriageInput,
    *,
    model: str = DEFAULT_JEV_MODEL,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict:
    """System One choice request over review metadata + comments."""
    return {
        "model": model,
        "state": build_review_state(payload, max_diff_chars=max_diff_chars),
        "questions": {
            VERDICT_QUESTION_ID: {
                "type": "choice",
                "instructions": CHOICE_INSTRUCTIONS,
                "criteria": dict(CHOICE_CRITERIA),
            },
        },
    }


def parse_verdict_answer(payload: Any) -> tuple[str, float]:
    """Extract ``(verdict, confidence)`` from a System One response. Raises on bad shape."""
    if not isinstance(payload, dict):
        raise ValueError("jev response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("jev response missing answers")
    ans = answers.get(VERDICT_QUESTION_ID)
    if not isinstance(ans, dict):
        raise ValueError(f"missing answer for {VERDICT_QUESTION_ID}")
    choice = str(ans.get("choice") or "").strip().lower().replace("_", "-")
    # Accept common aliases from models.
    aliases = {
        "request_changes": "request-changes",
        "requestchanges": "request-changes",
        "changes-requested": "request-changes",
        "needs_human": "needs-human",
        "needshuman": "needs-human",
        "human": "needs-human",
        "lgtm": "approve",
    }
    choice = aliases.get(choice, choice)
    if choice not in VERDICTS:
        raise ValueError(f"unknown verdict choice: {choice!r}")
    try:
        confidence = float(ans["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("verdict answer missing confidence") from exc
    return choice, confidence


def apply_confidence_gate(
    choice: str,
    confidence: float,
    threshold: float,
) -> tuple[str, str]:
    """Force ``needs-human`` when confidence is below threshold (never keep approve).

    Returns ``(verdict, reason)``.
    """
    if confidence < threshold:
        return "needs-human", "below_threshold"
    return choice, "ok"


def log_jev_review_triage(
    *,
    verdict: str,
    confidence: Optional[float],
    threshold: float,
    model: str = "",
    fallback: bool = False,
    reason: str = "",
    should_post: bool = True,
) -> None:
    payload = {
        "marker": JEV_REVIEW_TRIAGE_MARKER,
        "verdict": verdict,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "threshold": float(threshold),
        "model": model or "",
        "fallback": bool(fallback),
        "reason": reason or "",
        "should_post": bool(should_post),
    }
    logger.info(
        "[latency] "
        + JEV_REVIEW_TRIAGE_MARKER
        + " verdict=%s confidence=%s threshold=%s model=%s fallback=%s "
        "should_post=%s reason=%s",
        payload["verdict"],
        payload["confidence"] if payload["confidence"] is not None else "none",
        payload["threshold"],
        payload["model"] or "-",
        "true" if fallback else "false",
        "true" if should_post else "false",
        payload["reason"] or "ok",
        extra={"jev_review_triage": payload},
    )


def triage_pr_review(
    payload: ReviewTriageInput,
    *,
    user_config: Any = None,
    http_client: Any = None,
    api_key: Optional[str] = None,
    cfg: Optional[JevReviewTriageConfig] = None,
) -> JevReviewTriageResult:
    """Ask Jev for a PR review verdict; low confidence / errors -> needs-human.

    Never returns ``approve`` when confidence is below threshold. Disabled config
    returns ``should_post=False`` so callers leave the existing review flow alone.
    """
    resolved = cfg if cfg is not None else load_jev_review_triage_config(user_config)
    if not resolved.enabled:
        return JevReviewTriageResult(
            verdict="needs-human",
            reason="disabled",
            should_post=False,
            model=resolved.model,
        )

    key = (api_key if api_key is not None else resolve_typesafe_api_key()).strip()
    if not key:
        log_jev_review_triage(
            verdict="needs-human",
            confidence=None,
            threshold=resolved.threshold,
            model=resolved.model,
            fallback=True,
            reason="missing_key",
        )
        return JevReviewTriageResult(
            verdict="needs-human",
            fallback=True,
            reason="missing_key",
            model=resolved.model,
        )

    if not payload.files and not payload.comments and not (payload.diff_excerpt or "").strip():
        log_jev_review_triage(
            verdict="needs-human",
            confidence=None,
            threshold=resolved.threshold,
            model=resolved.model,
            fallback=True,
            reason="empty_payload",
        )
        return JevReviewTriageResult(
            verdict="needs-human",
            fallback=True,
            reason="empty_payload",
            model=resolved.model,
        )

    try:
        body = build_jev_review_triage_request(
            payload,
            model=resolved.model,
            max_diff_chars=resolved.max_diff_chars,
        )
        data, _ttft_ms, _ready_ms = _post_systemone(
            body,
            api_key=key,
            timeout_seconds=resolved.timeout_seconds,
            http_client=http_client,
        )
        choice, confidence = parse_verdict_answer(data)
    except Exception as exc:
        reason = type(exc).__name__
        msg = str(exc).strip()
        if "429" in msg:
            reason = "rate_limited"
        elif "http " in msg.lower() or "timeout" in msg.lower() or "timed out" in msg.lower():
            reason = msg.replace(" ", "_")[:64] if msg else reason
        log_jev_review_triage(
            verdict="needs-human",
            confidence=None,
            threshold=resolved.threshold,
            model=resolved.model,
            fallback=True,
            reason=reason,
        )
        return JevReviewTriageResult(
            verdict="needs-human",
            fallback=True,
            reason=reason,
            model=resolved.model,
        )

    verdict, gate_reason = apply_confidence_gate(choice, confidence, resolved.threshold)
    # Belt: never leave approve through a below-threshold path.
    if confidence < resolved.threshold and verdict == "approve":
        verdict, gate_reason = "needs-human", "below_threshold"

    log_jev_review_triage(
        verdict=verdict,
        confidence=confidence,
        threshold=resolved.threshold,
        model=resolved.model,
        fallback=False,
        reason=gate_reason,
    )
    return JevReviewTriageResult(
        verdict=verdict,
        confidence=confidence,
        reason=gate_reason,
        model=resolved.model,
        raw_choice=choice,
    )


def format_verdict_comment(result: JevReviewTriageResult, *, pr_number: int = 0) -> str:
    """Markdown body for the single PR triage comment (includes dedupe marker)."""
    conf = (
        f"{result.confidence:.2f}"
        if result.confidence is not None
        else "n/a"
    )
    label = {
        "approve": "Approve (recommendation only — human gate still required)",
        "request-changes": "Request changes",
        "needs-human": "Needs human review",
    }.get(result.verdict, result.verdict)
    pr_line = f"**PR:** #{pr_number}\n" if pr_number else ""
    return (
        f"## Jev review triage\n\n"
        f"{pr_line}"
        f"**Verdict:** {label}\n"
        f"**Confidence:** {conf}\n"
        f"**Reason:** {result.reason or 'ok'}\n\n"
        f"_Hermes Jev triage does not replace the human review gate; "
        f"it never auto-approves on low confidence._\n"
        f"{VERDICT_COMMENT_MARKER}\n"
    )


def github_review_event_for(verdict: str) -> str:
    """Map a triage verdict to a GitHub review event.

    Always ``COMMENT`` — this seam must not submit ``APPROVE`` (preserves the
    human review gate). Formal approve/request-changes remain a human or
    explicit skill action outside this triage layer.
    """
    del verdict  # event is commentary regardless of verdict
    return "COMMENT"


def comment_already_posted(existing_bodies: Sequence[str]) -> bool:
    """True when a prior triage comment (marker) is already on the PR."""
    return any(VERDICT_COMMENT_MARKER in (body or "") for body in existing_bodies)


def post_verdict_comment_once(
    *,
    existing_bodies: Sequence[str],
    body: str,
    poster: Callable[[str], Any],
) -> Dict[str, Any]:
    """Post at most one triage comment; skip when the marker is already present.

    ``poster(body)`` performs the side effect (e.g. ``gh pr comment``). Returns
    a small status dict for callers/tests.
    """
    if comment_already_posted(existing_bodies):
        return {"posted": False, "reason": "already_posted"}
    if VERDICT_COMMENT_MARKER not in body:
        body = body.rstrip() + f"\n{VERDICT_COMMENT_MARKER}\n"
    poster(body)
    return {"posted": True, "reason": "ok"}
