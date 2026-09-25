"""Ningning-only, never Rancana

Semantic do-not-shrink pins for the main context-compaction path.

A pinned span survives compaction verbatim regardless of ``protect_first_n``,
``protect_last_n``, and the Jev keep score. Pins are a closed class set.
Unknown class names are dropped, never added. The feature is config-gated and
default OFF: nothing in the compressor changes until ``compression.semantic_pins.enabled``
is true on the agent that was constructed with that snapshot.

Reload boundary (init-consumed): parsed in ``agent_init._parse_compression_config``
and stored on ``ContextCompressor`` at construction. Not listed in
``GatewayRunner._CACHE_BUSTING_CONFIG_KEYS`` and not applied by
``tui_gateway.session_compression._apply_live_compression_config``. A flip takes
effect when the agent is reconstructed, not on the next turn of a live cached
agent. Same boundary as ``compression.jev_scorer``. This does not resolve the
existing conflict for ``compression.threshold``, which the hermes-configuration
docs call init-consumed while ``gateway/run.py`` lists it in the cache-busting
table and ``tests/tui_gateway/test_compression_config_hot_reload.py`` tests TUI
hot reload.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PIN_CLASSES: Tuple[str, ...] = (
    "active_task",
    "goal_scaffold",
    "standing_instruction",
)
PIN_CLASS_SET = frozenset(PIN_CLASSES)

# Mark a message with one of these keys. The value must be a closed-set class.
PIN_MARK_KEYS = ("semantic_pin",)

CONFIG_CLASSIFICATION: Dict[str, str] = {
    "compression.semantic_pins.enabled": "init-consumed",
    "compression.semantic_pins.classes": "init-consumed",
    "compression.visibility_ladder.enabled": "init-consumed",
    "compression.visibility_ladder.scoring": "init-consumed",
    "compression.cache_reuse_decision": "init-consumed",
}

RELOAD_BOUNDARY = (
    "init-consumed: compression.semantic_pins, compression.visibility_ladder, and "
    "compression.cache_reuse_decision are parsed in agent_init._parse_compression_config "
    "and stored on ContextCompressor at construction. They are not in "
    "GatewayRunner._CACHE_BUSTING_CONFIG_KEYS and are not applied by "
    "tui_gateway.session_compression._apply_live_compression_config. A flip takes "
    "effect when the agent is reconstructed, not on the next turn of a live cached "
    "agent. Same boundary as compression.jev_scorer. This does not resolve the "
    "existing conflict for compression.threshold (hermes-configuration docs say it "
    "is consumed at agent init; gateway/run.py lists it in the cache-busting table; "
    "tests/tui_gateway/test_compression_config_hot_reload.py tests TUI hot reload)."
)


@dataclass(frozen=True)
class SemanticPinsConfig:
    """Parsed ``compression.semantic_pins`` block. Default OFF."""

    enabled: bool = False
    classes: Tuple[str, ...] = PIN_CLASSES


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "", "none"}:
        return False
    return default


def _closed_classes(raw: Any) -> Tuple[str, ...]:
    """Keep only names in the closed set, in closed-set order. Never invent a class."""
    if raw is None:
        return PIN_CLASSES
    if isinstance(raw, str):
        requested = [raw]
    elif isinstance(raw, (list, tuple)):
        requested = list(raw)
    else:
        return PIN_CLASSES
    allowed = PIN_CLASS_SET
    chosen = [name for name in PIN_CLASSES if name in requested and name in allowed]
    # An explicit empty list means "no class is active", not "open the set".
    if isinstance(raw, (list, tuple)) and len(raw) == 0:
        return ()
    if not chosen and not isinstance(raw, (list, tuple)):
        return PIN_CLASSES
    return tuple(chosen)


def parse_semantic_pins_config(raw: Any) -> SemanticPinsConfig:
    """Parse ``compression.semantic_pins``. Missing or malformed means disabled."""
    if not isinstance(raw, Mapping):
        return SemanticPinsConfig(enabled=False, classes=PIN_CLASSES)
    classes = _closed_classes(raw.get("classes")) if "classes" in raw else PIN_CLASSES
    return SemanticPinsConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        classes=classes,
    )


def explicit_pin_class(message: Any, allowed: Iterable[str]) -> Optional[str]:
    """Return the closed-set pin class marked on *message*, or None.

    Marks are local metadata. They do not themselves call Jev. A mark outside
    the configured subset of the closed set is ignored.
    """
    if not isinstance(message, dict):
        return None
    allowed_set = frozenset(allowed)
    candidates = []
    for key in PIN_MARK_KEYS:
        if key in message:
            candidates.append(message.get(key))
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        for key in PIN_MARK_KEYS:
            if key in metadata:
                candidates.append(metadata.get(key))
    for raw in candidates:
        name = str(raw or "").strip()
        if name in PIN_CLASS_SET and name in allowed_set:
            return name
    return None


def pin_survives(
    *,
    index: int,
    n: int,
    score: float,
    keep_threshold: float,
    protect_first_n: int,
    protect_last_n: int,
    pinned: bool,
) -> bool:
    """Whether index survives verbatim.

    A pin survives even when it is outside the positional head/tail and the
    Jev keep score is below ``keep_threshold``. An unpinned index survives only
    via position or score.
    """
    if pinned:
        return True
    if protect_first_n > 0 and 0 <= index < protect_first_n:
        return True
    if protect_last_n > 0 and n - protect_last_n <= index < n:
        return True
    try:
        return float(score) >= float(keep_threshold)
    except (TypeError, ValueError):
        return False


def redact_message_for_compaction(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy *message* with compaction-boundary redaction applied.

    Pinning must not reintroduce spans that ``_redact_compaction_text`` strips
    at every other compaction boundary (content, tool args, tool results).
    The surviving text is verbatim after that redaction, not a second copy of
    the secret.
    """
    from agent.context_compressor import _redact_compaction_text

    out = copy.deepcopy(dict(message))
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = _redact_compaction_text(content)
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(_redact_compaction_text(part))
            elif isinstance(part, dict):
                copied = dict(part)
                if isinstance(copied.get("text"), str):
                    copied["text"] = _redact_compaction_text(copied["text"])
                if isinstance(copied.get("content"), str):
                    copied["content"] = _redact_compaction_text(copied["content"])
                parts.append(copied)
            else:
                parts.append(part)
        out["content"] = parts
    tool_calls = out.get("tool_calls")
    if isinstance(tool_calls, list):
        redacted_calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                redacted_calls.append(call)
                continue
            copied = copy.deepcopy(call)
            fn = copied.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                fn["arguments"] = _redact_compaction_text(fn["arguments"])
            redacted_calls.append(copied)
        out["tool_calls"] = redacted_calls
    return out


def split_explicit_pins(
    messages: Sequence[Mapping[str, Any]],
    cfg: SemanticPinsConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split explicit pins out of *messages*.

    Returns ``(remainder, pinned_verbatim)``. Pinned copies are redacted.
    Remainder items are the original dicts (not copied) so a disabled or
    fallback caller can keep the unchanged list. When the feature is off,
    remainder is the input list and pinned is empty.
    """
    if not cfg.enabled:
        return [dict(msg) for msg in messages if isinstance(msg, dict)], []
    remainder: List[Dict[str, Any]] = []
    pinned: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if explicit_pin_class(msg, cfg.classes):
            pinned.append(redact_message_for_compaction(msg))
        else:
            remainder.append(msg)
    return remainder, pinned
