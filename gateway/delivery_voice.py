"""Last-mile voice guard for conversational gateway replies.

This is intentionally a narrow delivery check: it removes only stock assistant
scaffolding and normalizes em dash / en dash punctuation, leaving names,
affection, language mix, humour, safety text, and bubble boundaries untouched.

Dashes are handled mechanically here rather than by prompt text, because a rule
that lives only in the prompt is context a drifting reply can ignore: a spaced
dash reads as a parenthetical aside and becomes ", ", while a dash glued to its
neighbours is a range or a hyphenation and becomes "-". Text the user asked to
see verbatim (fenced code blocks and inline code spans) is copied through
byte-for-byte; rewriting punctuation inside a snippet would corrupt real
content.

Two seams call into here: the assembled final text
(``final_delivery_voice_check``) and the *streamed* frames that reach the user
while the answer is still being generated, when the guarded final send is
suppressed (``normalize_stream_dashes``). The stream variant is narrower on
purpose: it removes nothing, so a growing preview never snaps backwards.
"""
import logging
import re

logger = logging.getLogger(__name__)

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
    re.compile(r"\bwhat'?s next\??\s*", re.IGNORECASE),
    re.compile(r"\blet me know how (?:it|things) go(?:es)?\.?\s*", re.IGNORECASE),
    re.compile(r"\bfeel free to reach out\.?\s*", re.IGNORECASE),
    re.compile(r"\bany questions\??\s*", re.IGNORECASE),
)

_LEADING_LETTER = re.compile(r"^(\W*)(\w)", re.UNICODE)

# An aside: a dash with a space on its LEFT and whitespace (or the end of the
# chunk) on its right. A dash at the start of a line keeps its space on the
# right only, so a dash bullet stays a hyphen instead of turning into a comma.
_SPACED_DASH_RE = re.compile(r"(?<=[ \t])[\u2013\u2014]+(?=[ \t]|$)", re.MULTILINE)
# Anything left over was glued to its neighbours: a range/hyphenation, not an aside.
_BARE_DASH_RE = re.compile(r"[\u2013\u2014]+")
# A fence delimiter line (up to three leading spaces, CommonMark-style). The
# delimiter lines themselves stay inside the protected region, so the block is
# copied through including its fences.
_FENCE_DELIMITER_RE = re.compile(r"[ \t]{0,3}(`{3,}|~{3,})")

# A URL is data, not prose: rewriting a dash inside one would point the link at a
# different address, so it is protected exactly like a code span.
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>`]+", re.IGNORECASE)

# Punctuation can end up doubled once a dash (or a stripped phrase) leaves a
# comma behind; these collapse it so ",," and " ," can never ship.
_STRAY_SPACE_BEFORE_COMMA_RE = re.compile(r"\s+,")
_DOUBLED_COMMA_RE = re.compile(r",\s*,+")
_RUN_OF_SPACES_RE = re.compile(r"[ \t]{2,}")

# Stream-frame variants of the two rules above. A streamed frame is a *prefix* of
# the frame that follows, so these never match a newline: the transform may only
# repair the comma shapes its own dash rewrite creates.
_STREAM_STRAY_SPACE_BEFORE_COMMA_RE = re.compile(r"[ \t]+,")
_STREAM_RUN_OF_SPACES_AFTER_COMMA_RE = re.compile(r"(?<=,)[ \t]{2,}")


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


def _iter_fenced_lines(text: str):
    """Yield ``(line, in_fence)`` per line, so fenced regions are never rewritten.

    An unclosed fence protects the rest of the message: that text is what the
    model meant to hand over verbatim, and guessing otherwise is worse than
    leaving a stray backtick alone.
    """
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE_DELIMITER_RE.match(line):
            yield line, True
            in_fence = not in_fence
            continue
        yield line, in_fence


def _iter_inline_spans(line: str):
    """Yield ``(chunk, is_code)`` for one non-fenced line, protecting ``code`` spans.

    A backtick run is closed only by a run of the same length, and an unbalanced
    backtick is left as ordinary prose rather than swallowing the line.
    """
    prose_start = 0
    index = 0
    length = len(line)
    while index < length:
        if line[index] != "`":
            index += 1
            continue
        run_end = index
        while run_end < length and line[run_end] == "`":
            run_end += 1
        ticks = line[index:run_end]
        close = line.find(ticks, run_end)
        while close != -1 and line.startswith(ticks + "`", close):
            close = line.find(ticks, close + 1)
        if close == -1:
            index = run_end
            continue
        if index > prose_start:
            yield line[prose_start:index], False
        span_end = close + len(ticks)
        yield line[index:span_end], True
        prose_start = index = span_end
    if prose_start < length:
        yield line[prose_start:], False


def _iter_code_aware(text: str):
    """Split ``text`` into ``(chunk, is_code)`` pieces; verbatim text is marked code."""
    for line, in_fence in _iter_fenced_lines(text):
        if in_fence:
            yield line, True
        else:
            yield from _iter_inline_spans(line)


def _iter_url_spans(chunk: str):
    """Split one prose chunk into ``(piece, is_url)`` so a link is copied verbatim."""
    cursor = 0
    for match in _URL_RE.finditer(chunk):
        if match.start() > cursor:
            yield chunk[cursor:match.start()], False
        yield match.group(0), True
        cursor = match.end()
    if cursor < len(chunk):
        yield chunk[cursor:], False


def _normalize_dashes(chunk: str) -> str:
    """Rewrite em/en dashes in prose: a spaced aside becomes ", ", a glued dash "-"."""
    return _BARE_DASH_RE.sub("-", _SPACED_DASH_RE.sub(", ", chunk))


def _tidy_dash_punctuation(chunk: str) -> str:
    """Collapse punctuation the rewrite can double up, so ",," and " ," cannot ship."""
    chunk = _STRAY_SPACE_BEFORE_COMMA_RE.sub(",", chunk)
    chunk = _DOUBLED_COMMA_RE.sub(",", chunk)
    return _RUN_OF_SPACES_RE.sub(" ", chunk)


def _strip_delivery_scaffolding(chunk: str):
    """Remove stock assistant phrases from prose; returns ``(text, changed)``."""
    changed = False
    for phrase in _AI_SHAPED_DELIVERY_PHRASES:
        updated = phrase.sub("", chunk)
        if updated != chunk:
            changed = True
            chunk = updated
    return chunk, changed


def final_delivery_voice_check(text: str) -> str:
    """Return text safe for final conversational delivery without changing its meaning.

    Idempotent: once a message has been through here it carries no U+2013/U+2014
    in prose and no doubled punctuation, so running it again is a no-op.
    """
    checked = _strip_internal_markers(str(text or "").strip())
    if not checked:
        return ""
    pieces = []
    stripped_scaffolding = False
    for chunk, is_code in _iter_code_aware(checked):
        if is_code:
            pieces.append(chunk)
            continue
        chunk = _normalize_dashes(chunk)
        chunk, changed = _strip_delivery_scaffolding(chunk)
        stripped_scaffolding = stripped_scaffolding or changed
        pieces.append(_tidy_dash_punctuation(chunk))
    checked = "".join(pieces).strip(" \t,:")
    if stripped_scaffolding and checked:
        checked = _recapitalize_start(checked)
    return checked


def _tidy_stream_dash_punctuation(chunk: str) -> str:
    """Collapse punctuation a dash rewrite can double up (``",,"``, ``" ,"``, ``",  "``).

    Narrower than :func:`_tidy_dash_punctuation` on purpose: a streamed frame is a
    *prefix* of the next one, so the transform must not touch whitespace runs it
    did not create — collapsing a nested list's indentation would make the preview
    jump. The two rules below repair exactly the shapes ``", "`` leaves behind
    (a space before the comma, and the extra space next to the one it inserts),
    and neither ever eats a newline.
    """
    chunk = _STREAM_STRAY_SPACE_BEFORE_COMMA_RE.sub(",", chunk)
    chunk = _DOUBLED_COMMA_RE.sub(",", chunk)
    return _STREAM_RUN_OF_SPACES_AFTER_COMMA_RE.sub(" ", chunk)


def normalize_stream_dashes(text: str) -> str:
    """Dash-only normalization for a frame of *streamed* (not yet final) text.

    This is the seam that bypasses :func:`final_delivery_voice_check`: when the
    model streams its answer, the guarded final send is suppressed and the raw
    frames are what the user reads. Only the dash rewrite is applied here —
    scaffolding phrases and internal markers are deliberately NOT stripped,
    because removing content mid-stream makes a growing preview snap backwards.

    Properties a streaming transform must have, and this one does:
      * idempotent, so re-running it on an already-clean frame is a no-op;
      * code-aware: fenced blocks and inline code spans are copied byte-for-byte;
      * URL-safe: a dash inside a bare link is never rewritten;
      * never doubles punctuation;
      * total: it swallows its own failures, because a formatting helper must
        never delay, reorder or drop a frame (mirrors ``naturalness_voice``).
    """
    try:
        if not isinstance(text, str) or not text:
            return text
        pieces = []
        for chunk, is_code in _iter_code_aware(text):
            if is_code:
                pieces.append(chunk)
                continue
            for piece, is_url in _iter_url_spans(chunk):
                pieces.append(piece if is_url
                              else _tidy_stream_dash_punctuation(_normalize_dashes(piece)))
        return "".join(pieces)
    except Exception:  # pragma: no cover - defensive: a frame must always ship
        logger.debug("stream dash normalization failed; sending the frame unchanged",
                     exc_info=True)
        return text

