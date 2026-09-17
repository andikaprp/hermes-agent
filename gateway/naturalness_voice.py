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

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

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

_MAX_FIRST_SENTENCE_WORDS = 28


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.code}: {self.detail}"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _structure_hits(text: str) -> list[str]:
    return [name for name, pattern in _STRUCTURE if pattern.search(text)]


def _internal_hits(text: str) -> list[str]:
    return [name for name, pattern in _INTERNAL if pattern.search(text)]


def audit_voice(
    text: str,
    *,
    register: str = REGISTER_CHAT,
    inbound: str = "",
    asks_for_evidence: bool = False,
) -> list[Finding]:
    """Return every voice-standard finding for one reply, in reason-code order.

    ``register`` is the turn type: a social/ack turn or an ordinary chat turn. Pass
    ``asks_for_evidence=True`` when he explicitly asked for the internals, which is the
    one condition that licenses technical detail in a 1:1 reply.
    """
    findings: list[Finding] = []
    stripped = text.strip()
    if not stripped:
        return findings

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
    sentences = _sentences(stripped)
    first = sentences[0] if sentences else ""
    if _META_LEAD.match(first_line.strip()) or _structure_hits(first_line):
        findings.append(Finding("buried_answer", "leads with meta or structure"))
    elif register == REGISTER_CHAT and len(first.split()) > _MAX_FIRST_SENTENCE_WORDS:
        findings.append(Finding("buried_answer", f"first sentence is {len(first.split())} words"))

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
              asks_for_evidence: bool = False) -> Sequence[str]:
    """Convenience: the reason codes for one reply."""
    return [f.code for f in audit_voice(text, register=register, inbound=inbound,
                                        asks_for_evidence=asks_for_evidence)]
