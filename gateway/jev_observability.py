"""Config-gated local Jev decision observability (LAB-59).

``<hermes_home>/logs/jev-decisions.jsonl`` is THE decision store (plus an
in-process ring of the same rows). ``log_jev_*`` lines are diagnostic logs,
not a second store — this module does not tail gateway.log. The dashboard
GET routes are a read-only local view of this store. Nothing is sent
off-box.

The jsonl uses stdlib ``RotatingFileHandler`` semantics: when the next line
would exceed ``DECISIONS_MAX_BYTES``, the current file rolls to
``jev-decisions.jsonl.1`` (backupCount=1). ``maxBytes == 0`` disables
rollover, matching the stdlib handler.

Modes (``gateway.jev_observability.mode``), default ``off``:

- ``off`` — do not record; API reports disabled
- ``shadow`` — record + export/API readable (behaviour of Jev features unchanged)
- ``on`` — same recording; dashboard UI is intended to surface the feed
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CONFIG_KEY = "gateway.jev_observability"
DECISIONS_FILENAME = "jev-decisions.jsonl"
# Stdlib RotatingFileHandler: 0 disables rollover. backupCount=1 -> ``.1`` only.
DECISIONS_MAX_BYTES = 1_048_576
DECISIONS_BACKUP_COUNT = 1
DEFAULT_LIMIT = 200
VALID_MODES = frozenset({"off", "shadow", "on"})

_LOCK = threading.RLock()
_RING: Deque[Dict[str, Any]] = deque(maxlen=DEFAULT_LIMIT)
_RING_LIMIT = DEFAULT_LIMIT


@dataclass(frozen=True)
class JevObservabilityConfig:
    mode: str = "off"
    limit: int = DEFAULT_LIMIT


def parse_jev_observability_config(raw: Any) -> JevObservabilityConfig:
    """Build config from a mapping; unknown/malformed -> defaults (off)."""
    if not isinstance(raw, dict):
        return JevObservabilityConfig()
    mode = str(raw.get("mode") or "off").strip().lower()
    if mode not in VALID_MODES:
        # Legacy bool-ish: enabled true → on, false → off.
        if str(raw.get("enabled", "")).lower() in {"true", "1", "yes"}:
            mode = "on"
        else:
            mode = "off"
    try:
        limit = int(raw.get("limit", DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    limit = max(1, min(5000, limit))
    return JevObservabilityConfig(mode=mode, limit=limit)


def load_jev_observability_config(user_config: Any = None) -> JevObservabilityConfig:
    """``gateway.jev_observability`` from YAML; default OFF when absent."""
    try:
        if user_config is None:
            from hermes_cli.config import load_config_readonly

            user_config = load_config_readonly()
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        raw = gw.get("jev_observability") if isinstance(gw, dict) else None
        return parse_jev_observability_config(raw)
    except Exception:
        return JevObservabilityConfig()


def content_hash(text: Any, *, n: int = 16) -> str:
    """Stable short hash of outbound state text (never the text itself)."""
    if text is None:
        blob = b""
    elif isinstance(text, bytes):
        blob = text
    else:
        blob = str(text).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:n]


def decisions_path() -> Any:
    """Profile-aware jsonl path; resolved at call time."""
    return Path(get_hermes_home()) / "logs" / DECISIONS_FILENAME


def _rollover_if_needed(path: Path, incoming: int, max_bytes: int) -> None:
    """Rename ``path`` to ``path.1`` using stdlib RotatingFileHandler (backupCount=1).

    Matches ``shouldRollover``: never roll an empty/missing file; roll when
    ``size + incoming >= maxBytes``. A single record may exceed the cap.
    """
    if max_bytes <= 0:
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= 0 or size + incoming < max_bytes:
        return
    handler = RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=DECISIONS_BACKUP_COUNT,
        delay=True,
    )
    try:
        handler.doRollover()
    finally:
        handler.close()


def _append_decision_line(path: Path, line: str, *, max_bytes: int | None = None) -> None:
    cap = DECISIONS_MAX_BYTES if max_bytes is None else max_bytes
    payload = line if line.endswith("\n") else line + "\n"
    data = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    _rollover_if_needed(path, len(data), cap)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _ensure_ring_capacity(limit: int) -> None:
    global _RING, _RING_LIMIT
    if limit == _RING_LIMIT:
        return
    items = list(_RING)
    _RING = deque(items[-limit:], maxlen=limit)
    _RING_LIMIT = limit


def recording_enabled(cfg: Optional[JevObservabilityConfig] = None) -> bool:
    resolved = cfg if cfg is not None else load_jev_observability_config()
    return resolved.mode in {"shadow", "on"}


def record_jev_decision(
    *,
    kind: str,
    tier: str = "",
    model: str = "",
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
    reason: str = "",
    content_hash_value: str = "",
    user_config: Any = None,
    **extra: Any,
) -> Optional[Dict[str, Any]]:
    """Append one decision record when mode is shadow/on. Best-effort; never raises."""
    try:
        cfg = load_jev_observability_config(user_config)
        if cfg.mode not in {"shadow", "on"}:
            return None
        _ensure_ring_capacity(cfg.limit)
        entry: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "kind": str(kind or ""),
            "tier": str(tier or ""),
            "model": str(model or ""),
            "confidence": None if confidence is None else round(float(confidence), 4),
            "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
            "reason": str(reason or "")[:128],
            "content_hash": str(content_hash_value or "")[:64],
            "mode": cfg.mode,
        }
        # Metadata-only extras (no free-form text blobs).
        for key, value in extra.items():
            if key in entry or value is None:
                continue
            if isinstance(value, (bool, int, float)):
                entry[key] = value
            elif isinstance(value, str) and len(value) <= 128:
                entry[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(x, (str, int, float, bool)) for x in value
            ):
                entry[key] = list(value)[:32]
        with _LOCK:
            _RING.append(entry)
            line = json.dumps(entry, separators=(",", ":"), default=str)
            _append_decision_line(decisions_path(), line)
        return entry
    except Exception as exc:  # pragma: no cover - observability must never break a turn
        logger.debug("jev observability record failed: %s", exc)
        return None


def _tail_jsonl(path: Any, *, limit: int) -> List[Dict[str, Any]]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max(16_384, limit * 512)))
            chunk = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = chunk.splitlines()
    if size > max(16_384, limit * 512) and lines:
        lines = lines[1:]  # first line may be truncated
    out: List[Dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out[-limit:]


def read_recent_decisions(
    *,
    limit: Optional[int] = None,
    user_config: Any = None,
) -> Dict[str, Any]:
    """Last N decisions for dashboard/export. Empty when mode is off."""
    cfg = load_jev_observability_config(user_config)
    n = limit if limit is not None else cfg.limit
    try:
        n = max(1, min(5000, int(n)))
    except (TypeError, ValueError):
        n = cfg.limit
    if cfg.mode == "off":
        return {
            "mode": "off",
            "enabled": False,
            "limit": n,
            "decisions": [],
            "path": str(decisions_path()),
        }
    with _LOCK:
        ring = list(_RING)[-n:]
    disk = _tail_jsonl(decisions_path(), limit=n)
    # Prefer the freshest merge of ring + disk by ts (ring may have mid-process rows).
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in disk + ring:
        key = f"{row.get('ts')}|{row.get('kind')}|{row.get('content_hash')}|{row.get('tier')}"
        by_key[key] = row
    merged = sorted(by_key.values(), key=lambda r: float(r.get("ts") or 0), reverse=True)[:n]
    return {
        "mode": cfg.mode,
        "enabled": True,
        "limit": n,
        "decisions": merged,
        "path": str(decisions_path()),
    }


def reset_observability_for_tests() -> None:
    """Clear the in-process ring (tests only)."""
    with _LOCK:
        _RING.clear()
