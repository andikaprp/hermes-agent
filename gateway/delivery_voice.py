"""Last-mile voice guard for conversational gateway replies.

This is intentionally a narrow delivery check: it removes only stock assistant
scaffolding and replaces em dashes, leaving names, affection, language mix,
humour, safety text, and bubble boundaries untouched.
"""
import re

_INTERNAL_DELIVERY_MARKERS = frozenset({
    "[response interrupted]",
    "operation interrupted.",
    "operation cancelled.",
    "operation canceled.",
})

# Optional trailing punctuation so "Understood!" / "Got it." still match.
_AI_SHAPED_DELIVERY_PHRASES = (
    re.compile(r"\bgot it\b[!,.:;]?\s*", re.IGNORECASE),
    re.compile(r"\bunderstood\b[!,.:;]?\s*", re.IGNORECASE),
    re.compile(r"\bhere(?:'s| is) what i found\s*:?\s*", re.IGNORECASE),
    re.compile(r"\bwhat would you like to do next\??\s*", re.IGNORECASE),
    re.compile(r"\blet me know if you need anything else\.?\s*", re.IGNORECASE),
)

_LEADING_LETTER = re.compile(r"^(\W*)(\w)", re.UNICODE)


def _is_internal_marker(text: str) -> bool:
    return text.strip().casefold() in _INTERNAL_DELIVERY_MARKERS


def _strip_internal_markers(text: str) -> str:
    if _is_internal_marker(text):
        return ""
    kept = [line for line in text.splitlines() if not _is_internal_marker(line)]
    return "\n".join(kept).strip()


def _recapitalize_start(text: str) -> str:
    match = _LEADING_LETTER.match(text)
    if not match:
        return text
    prefix, first = match.group(1), match.group(2)
    return prefix + first.upper() + text[match.end():]


def final_delivery_voice_check(text: str) -> str:
    """Return text safe for final conversational delivery without changing its meaning."""
    checked = _strip_internal_markers(str(text or "").strip())
    if not checked:
        return ""
    # Em dash and en dash are AI-shaped pauses; keep the clause, not the glyph.
    checked = checked.replace("—", ", ").replace("–", ", ")
    stripped_scaffolding = False
    for phrase in _AI_SHAPED_DELIVERY_PHRASES:
        updated = phrase.sub("", checked)
        if updated != checked:
            stripped_scaffolding = True
            checked = updated
    checked = re.sub(r"\s+,", ",", checked)
    checked = re.sub(r",\s*,+", ",", checked)
    checked = re.sub(r"[ \t]{2,}", " ", checked)
    checked = checked.strip(" \t,:")
    if stripped_scaffolding and checked:
        checked = _recapitalize_start(checked)
    return checked
