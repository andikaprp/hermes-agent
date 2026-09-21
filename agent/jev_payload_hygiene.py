"""Outbound Jev payload hygiene: redact secrets, truncate, hash tool dumps.

Decision logs / observability never store raw text. System One still needs
enough state to decide — this module strips secrets and replaces bulky tool
results with hash metadata before those builders send.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

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


def redact_secrets(text: str) -> str:
    """Mask credential-shaped spans; keep surrounding context for Jev."""
    out = text or ""
    out = _TOKEN_SHAPES.sub("[secret]", out)
    out = _SECRET_WORDS.sub("[secret]", out)
    out = _EMAIL.sub("[email]", out)
    return out


def mask_state_text(text: str, *, limit: int = 280) -> str:
    """Redact secrets and truncate for System One state strings."""
    cleaned = redact_secrets((text or "").strip().replace("\n", " "))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def tool_result_metadata(text: str, *, tool_call_id: str = "") -> str:
    """Replace raw tool output with hash + length metadata only."""
    raw = text or ""
    digest = content_hash(raw)
    tid = (tool_call_id or "").strip()
    prefix = f"[tool {tid}] " if tid else "[tool] "
    return f"{prefix}hash={digest} chars={len(raw)}"
