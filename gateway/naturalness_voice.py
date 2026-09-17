"""Voice-standard audit for Ningy's 1:1 chat replies.

Scope split: this module judges *voice and register*. It deliberately does not score length,
list-count, repetition or affection caps, which ``gateway/naturalness_eval.py`` (LAB-3
naturalness branch) already covers. The two compose: length layer + this voice layer.

The standard, as approved by Andika 2026-09-17:

* voice: a sharp, casual, helpful friend. Short, direct, specific, warm without gushing;
* answer first, then caveat or next step;
* match his language and energy, including casual Indonesian;
* ask only when a wrong guess would be expensive;
* never narrate machinery or internal status;
* social/ack turn: one short natural message; no bullets, no headers, no quote/reply_to
  unless answering an older message;
* work answer: direct answer first; bullets only for options, steps or comparisons;
* plain words and contractions are fine;
* banned: em dash, canned assistant phrases, fake enthusiasm, over-apology, moralizing,
  "let me know", process narration;
* preserve uncertainty plainly; emoji and reactions only when they genuinely fit;
* surface only details that change his decision or next action;
* no filler questions when the key fact can be stated directly;
* one idea per bubble; split beats instead of using sections or tables in ordinary chat;
* technical internals only when they are the next action or he asked for evidence;
* "verified" requires live evidence.

Every finding carries one of `REASON_CODES`. Detectors are conservative on purpose: a
false accusation on a good reply is worse than a miss, because it teaches the wrong voice.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

try:  # redaction helpers live with the delivery receipts; failing that, log no id at all
    from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token
except Exception:  # pragma: no cover - never let a logging helper break the module
    CHAT_DIGEST_PREFIX = "chat"

    def redacted_token(value: Any, *, prefix: str) -> str:
        return f"{prefix}=redacted"

REGISTER_CHAT = "chat"
REGISTER_SOCIAL = "social"
REASON_CODES = (
    "em_dash",
    "canned_phrase",
    "social_structure",
    "process_narration",
    "buried_answer",
    "filler_question",
    "unneeded_internal_detail",
    "unsupported_verified_claim",
)

_SOCIAL_REGISTERS = frozenset({REGISTER_CHAT, REGISTER_SOCIAL})

# ── detectors ────────────────────────────────────────────────────────────────

_EM_DASH = re.compile(r"[—–]")

_CANNED = re.compile(
    r"\b(?:acknowledged|as an ai|i'?d be happy to|happy to help|great question|"
    r"feel free to|i hope this helps|hope (?:that|this) helps|sure thing|absolutely!|"
    r"let me know|i'?ll let you know|please confirm|is there anything else|"
    r"let me know if|happy to clarify|just let me know)\b",
    re.I,
)

_STRUCTURE = (
    ("header", re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)),
    ("bold_label", re.compile(r"^\s*\*\*[^*\n]{1,60}\*\*\s*:?\s*$", re.M)),
    ("bullet", re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+\S", re.M)),
    ("table", re.compile(r"^\s*\|.*\|\s*$", re.M)),
    ("fence", re.compile(r"^\s*```", re.M)),
)

_PROCESS = re.compile(
    r"(?:^|[.!?]\s+)(?:let me\b|now (?:the|i|let|testing|checking|running|adding|updating|"
    r"posting|verifying|fixing)\b|first,? i\b|next,? i\b|i'?ll (?:check|look|run|start|now)\b|"
    r"i'?m going to\b|starting (?:now|the)\b|working on it\b|verifying\b|committing\b|"
    r"deploying\b|re-?running\b|reading the\b|checking the\b|building the\b|"
    # Narration also arrives as a bare gerund opening: "Smoke-testing with the production
    # interpreter before restarting." Same failure, different part of speech.
    r"(?:[A-Z][a-z]*-(?:testing|checking|verifying|running|reading|writing)|"
    r"Testing|Checking|Verifying|Running|Adding|Updating|Posting|Fixing|Deploying|Building|"
    r"Reading|Writing|Restarting|Rerunning|Smoke-?testing|Implementing|Committing|Merging|"
    r"Auditing|Screening|Calibrating|Reverting|Patching|Inspecting|Extracting)\b)",
    re.I | re.M,
)

_META_LEAD = re.compile(
    r"^(?:from here|here'?s|here is|current|board updated|status|quick update|update:|"
    r"done\.|so far|to summarise|to summarize|summary|recap|the headline)",
    re.I,
)

# Offering to do something he did not ask for. Anchored at a sentence start and phrased as
# an offer, so "What do you want me to verify first?" (an approved positive) is not a match.
_FILLER_QUESTION = re.compile(
    r"(?:^|[.!?]\s+)(?:want me to|should i|shall i|do you want me to|would you like me to|"
    r"want me to go ahead)\b",
    re.I,
)
_EXPENSIVE = re.compile(
    r"\b(?:deploy|delete|drop|credential|secret|token|money|pay|billing|production|prod|"
    r"destroy|revoke|rotate|migrate)\b",
    re.I,
)

# Internals that only belong in a 1:1 reply when they are the next action or he asked.
_INTERNAL = (
    ("ticket_id", re.compile(r"\b[A-Z]{2,5}-\d{1,5}\b")),
    ("check_code", re.compile(r"\bB\d{2}\b(?:\s*[,/=]|$)")),
    ("file_path", re.compile(r"(?:/workspace/|/home/|~/\.[a-z]|\b[\w/-]+\.py\b)")),
    ("backticked", re.compile(r"`[^`\n]+`")),
    ("pid", re.compile(r"\bPID\b|\bpid\s*\d+")),
    ("env_flag", re.compile(r"\b[A-Z][A-Z0-9_]{3,}\s*=")),
    ("code_fence_id", re.compile(r"^\s*```(?!$)", re.M)),
)

_VERIFIED_CLAIM = re.compile(
    r"\b(?:verified|verifies|confirmed|proven|proves|tested|validated)\b", re.I
)
_LIVE_EVIDENCE = re.compile(
    r"\b(?:live|log|receipt|measured|timestamp|observed|on real|PID|exit code|"
    r"from the gateway|in the gateway|sent|delivered|at \d{2}:\d{2})\b",
    re.I,
)

# The one condition that licenses technical detail in a 1:1 reply: he asked for it. Matched on
# HIS message only, never on the reply, so a reply cannot license itself by using the words.
_EVIDENCE_REQUEST = re.compile(
    r"\b(?:check|status|receipts?|evidence|prove|proven|verify|verification|confirm|"
    r"reports?|logs?|details?|explain|summar(?:y|ise|ize)|update me|show me|"
    r"walk me through|how do you know|what changed|did it work|numbers|breakdown)\b",
    re.I,
)

# A line that renders as structure rather than prose. What hides an answer is a header,
# fence, table row, bullet or bold label sitting ahead of the first prose line, not how long
# the answer's first sentence happens to be.
_STRUCTURE_LINE = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|~~~|\*\*[^*]+\*\*\s*:?\s*$)"
)


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.code}: {self.detail}"


def _structure_hits(text: str) -> list[str]:
    return [name for name, pattern in _STRUCTURE if pattern.search(text)]


def _internal_hits(text: str) -> list[str]:
    return [name for name, pattern in _INTERNAL if pattern.search(text)]


def inbound_asks_for_evidence(inbound: Any) -> bool:
    """Whether his message licensed technical detail: a check, a status, receipts, evidence.

    Named apart from ``audit_voice``'s ``asks_for_evidence`` parameter so the parameter cannot
    shadow the function inside the body.
    """
    return bool(_EVIDENCE_REQUEST.search(inbound)) if isinstance(inbound, str) else False


def audit_voice(
    text: str,
    *,
    register: str = REGISTER_CHAT,
    inbound: str = "",
    asks_for_evidence: Optional[bool] = None,
) -> list[Finding]:
    """Return every voice-standard finding for one reply, in reason-code order.

    ``register`` is the turn type: a social/ack turn or an ordinary chat turn. Pass ``inbound``
    and licensing is decided for you: a message that asked to check, for status, receipts or
    evidence licenses technical detail, which is the one condition that allows it in a 1:1
    reply. ``asks_for_evidence`` overrides that derivation when a caller already knows.
    """
    findings: list[Finding] = []
    stripped = text.strip()
    if not stripped:
        return findings

    if asks_for_evidence is None:
        asks_for_evidence = inbound_asks_for_evidence(inbound)

    for match in _EM_DASH.finditer(stripped):
        findings.append(Finding("em_dash", f"em dash at char {match.start()}"))
        break

    canned = _CANNED.search(stripped)
    if canned:
        findings.append(Finding("canned_phrase", repr(canned.group(0))))

    if register in _SOCIAL_REGISTERS:
        hits = _structure_hits(stripped)
        if hits:
            findings.append(Finding("social_structure", ", ".join(hits)))

    narration = _PROCESS.search(stripped)
    if narration:
        findings.append(Finding("process_narration", repr(narration.group(0).strip())))

    first_line = next((ln for ln in stripped.splitlines() if ln.strip()), "")
    # A buried answer is hidden behind structure or a meta opener, not merely long. Sentence
    # length is deliberately NOT a trigger: a direct answer written as one long sentence is
    # the standard working, and flagging it was a false accusation on a good reply.
    leading_structure = []
    first_prose = ""
    for line in stripped.splitlines():
        if not line.strip():
            continue
        if _STRUCTURE_LINE.match(line):
            leading_structure.append(line.strip())
            continue
        first_prose = line.strip()
        break
    if leading_structure:
        findings.append(Finding("buried_answer",
                                "structure before the answer: " + repr(leading_structure[0][:40])))
    elif _META_LEAD.match(first_prose or first_line.strip()):
        findings.append(Finding("buried_answer", "meta opener before the answer"))

    filler = _FILLER_QUESTION.search(stripped)
    if filler and not _EXPENSIVE.search(stripped):
        findings.append(Finding("filler_question", repr(filler.group(0).strip())))

    internals = _internal_hits(stripped)
    if internals and not asks_for_evidence and register in _SOCIAL_REGISTERS:
        # Next-action commands are licensed; inventory detail is not.
        if not _is_next_action(stripped):
            findings.append(Finding("unneeded_internal_detail", ", ".join(internals)))

    claim = _VERIFIED_CLAIM.search(stripped)
    if claim and not _LIVE_EVIDENCE.search(stripped):
        findings.append(Finding("unsupported_verified_claim", repr(claim.group(0))))

    order = {code: i for i, code in enumerate(REASON_CODES)}
    return sorted(findings, key=lambda f: order[f.code])


def _is_next_action(text: str) -> bool:
    """True when the only internals present are the command he should run next."""
    return bool(re.search(r"^\s*(?:run|try|paste|execute|open)\b", text, re.I))


def audit_corpus(entries: Iterable[dict]) -> dict:
    """Score a corpus of ``{text, register, expect, codes}`` entries. Test helper."""
    rows = list(entries)
    passed, failed = 0, []
    for entry in rows:
        found = [f.code for f in audit_voice(entry["text"], register=entry.get("register", REGISTER_CHAT))]
        if entry["expect"] == "pass":
            if not found:
                passed += 1
            else:
                failed.append({"text": entry["text"][:80], "expect": "pass", "found": found})
            continue
        missing = [c for c in entry.get("codes", []) if c not in found]
        if found and not missing:
            passed += 1
        else:
            failed.append({"text": entry["text"][:80], "expect": "reject",
                           "must_include": entry.get("codes", []), "found": found,
                           "missing": missing})
    return {"passed": passed, "total": len(rows), "failed": failed}


def codes_for(text: str, *, register: str = REGISTER_CHAT, inbound: str = "",
              asks_for_evidence: Optional[bool] = None) -> Sequence[str]:
    """Convenience: the reason codes for one reply."""
    return [f.code for f in audit_voice(text, register=register, inbound=inbound,
                                        asks_for_evidence=asks_for_evidence)]


# ---------------------------------------------------------------------------
# Shadow scoring
# ---------------------------------------------------------------------------
# Enforcement is deliberately NOT here. Shadow mode exists so fresh replies can be measured
# while the generator habits behind em_dash / process_narration / unneeded_internal_detail
# are still being fixed. Nothing in this module may block, rewrite or delay a send; the
# reply-time guard is separate follow-up work gated on fresh violations dropping under 10%,
# zero false failures on the approved corpus, and explicit approval.
VOICE_SHADOW_MARKER = "voice_shadow"


def is_voice_shadow_enabled(user_config: Any = None) -> bool:
    """Whether outgoing replies are shadow-scored. Absent key means ON.

    The gateway loads user YAML without merging DEFAULT_CONFIG, so absence must still mean
    enabled or the shadow data silently stops accumulating on configs that predate the key.
    """
    if not isinstance(user_config, dict):
        return True
    node: Any = user_config
    for key in ("gateway", "telegram", "voice_shadow"):
        if not isinstance(node, dict):
            return True
        node = node.get(key, None)
        if node is None:
            return True
    if isinstance(node, bool):
        return node
    if isinstance(node, str):
        return node.strip().lower() not in {"false", "off", "no", "0", ""}
    return bool(node)


def log_voice_shadow(text: Any, *, register: str = REGISTER_CHAT,
                     chat_id: Any = None, inbound: str = "") -> Sequence[str]:
    """Score an outgoing reply and log the reason codes. Never blocks and never rewrites.

    ``inbound`` is his message for this turn: technical detail is licensed when he asked to
    check, for a status, or for evidence, and without it every receipt reads as a violation.
    Returns the codes so a test can assert on them without reading the log. Deliberately
    total: any failure is swallowed, because a measurement must never fail a send.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return []
        codes = codes_for(text, register=register, inbound=inbound if isinstance(inbound, str) else "")
        # Records whether technical detail was licensed, so a licensed-clean reply cannot be
        # mistaken for a habit that actually improved.
        licensed = inbound_asks_for_evidence(inbound)
        logger.info(
            "[latency] " + VOICE_SHADOW_MARKER + " chat=%s register=%s asked=%s codes=%s words=%d",
            redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX), register,
            "yes" if licensed else "no", ",".join(codes) or "clean", len(text.split()),
            extra={"voice_shadow": {"register": register, "codes": list(codes),
                                    "words": len(text.split()), "licensed": licensed}},
        )
        return codes
    except Exception:  # pragma: no cover - a scorer must not break a turn
        logger.debug("voice shadow scoring failed", exc_info=True)
        return []
