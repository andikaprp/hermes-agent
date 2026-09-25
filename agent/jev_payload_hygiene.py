"""Outbound Jev payload hygiene: one choke point for System One state.

Decision logs never store raw text. System One request builders must not
send raw prompt text or tool results either — ``text_metadata`` replaces
them with a content hash plus length, counts, and flags. ``tool_result_metadata``
is that same choke point for tool outputs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

# Mirror of tools/delegate_tool_jev.task_hash_for — short content fingerprint.
def content_hash(text: Any, *, n: int = 16) -> str:
    if text is None:
        blob = b""
    elif isinstance(text, bytes):
        blob = text
    else:
        blob = str(text).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:n]


_SECRET_WORDS = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|authorization\s*:|bearer\s+[a-z0-9._-]{8,}|"
    r"password|passwd|client[_ -]?secret|private[_ -]?key|BEGIN [A-Z ]*PRIVATE KEY)"
)
_TOKEN_SHAPES = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_LABEL_CHARS = re.compile(r"[^A-Za-z0-9_\[\]:.\- ]+")
_IDENT = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def redact_secrets(text: str) -> str:
    """Mask credential-shaped spans. Not used on the Jev request path."""
    out = text or ""
    out = _TOKEN_SHAPES.sub("[secret]", out)
    out = _SECRET_WORDS.sub("[secret]", out)
    out = _EMAIL.sub("[email]", out)
    return out


def mask_state_text(text: str, *, limit: int = 280) -> str:
    """Redact secrets and truncate. Prefer ``text_metadata`` for Jev state."""
    cleaned = redact_secrets((text or "").strip().replace("\n", " "))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def safe_ident(value: Any, *, limit: int = 64) -> str:
    """Identifier token, or empty if the value is free text."""
    text = str(value or "").strip()
    if not text or len(text) > limit:
        return ""
    if _IDENT.fullmatch(text):
        return text
    return ""


def _as_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


def _safe_label(label: str) -> str:
    cleaned = _LABEL_CHARS.sub("", label or "").strip()
    return cleaned[:48]


def _flag_token(flag: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "", str(flag or ""))[:32]
    return token


def _has_secret(text: str) -> bool:
    return bool(
        _SECRET_WORDS.search(text)
        or _TOKEN_SHAPES.search(text)
        or _EMAIL.search(text)
    )


def text_metadata(
    text: Any,
    *,
    label: str = "",
    extra_flags: Sequence[str] | None = None,
    extra_counts: Mapping[str, int] | None = None,
) -> str:
    """Replace raw text with hash + length/counts/flags. Never echoes ``text``.

    This is the single choke point for outbound Jev state. Callers pass a
    static ``label`` (role, field name). Counts and flags are derived; the
    source characters are not copied into the result.
    """
    raw = _as_text(text)
    digest = content_hash(raw)
    words = len(raw.split()) if raw.strip() else 0
    lines = 0 if not raw else raw.count("\n") + 1
    flags: list[str] = []
    if not raw.strip():
        flags.append("empty")
    if _has_secret(raw):
        flags.append("has_secret")
    if "\n" in raw:
        flags.append("multiline")
    for flag in extra_flags or ():
        token = _flag_token(flag)
        if token and token not in flags:
            flags.append(token)
    parts: list[str] = []
    safe_label = _safe_label(label)
    if safe_label and safe_label not in raw:
        parts.append(safe_label)
    parts.append(f"hash={digest}")
    parts.append(f"chars={len(raw)}")
    parts.append(f"words={words}")
    parts.append(f"lines={lines}")
    for key, value in (extra_counts or {}).items():
        token = _flag_token(str(key))
        if not token:
            continue
        try:
            parts.append(f"{token}={int(value)}")
        except (TypeError, ValueError):
            continue
    if flags:
        parts.append("flags=" + ",".join(flags))
    return " ".join(parts)


def tool_result_metadata(text: str, *, tool_call_id: str = "") -> str:
    """Replace raw tool output with hash + length metadata only."""
    tid = safe_ident(tool_call_id, limit=32)
    label = f"[tool {tid}]" if tid else "[tool]"
    return text_metadata(text, label=label)
