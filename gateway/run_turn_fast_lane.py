"""Compact-context fast lane for no-task Telegram turns (LAB-52).

The LAB-3 fast path (``run_turn_fast_path``) already routes bare acks onto a one-hop
agent turn, but that hop still carries the full session prompt (100k+ tokens). This
module adds a SEPARATE lightweight provider call: minimal system line + last N chat
messages as a compact transcript. The main conversation's cached prefix is never
touched — tools stay on the wire for the fallback path; this lane is an independent
``call_llm`` with its own messages list.

Fail-soft: any error, empty reply, or time-to-first-token over the budget returns
``None`` so the caller falls back to the existing one-hop path unchanged.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

FAST_LANE_MARKER = "fast_lane"
CONFIG_KEY = "gateway.telegram.fast_lane"

FAST_LANE_SYSTEM = (
    "You are Ningning, a warm casual AI assistant. "
    "Reply in the language of the user, short and natural, "
    "no markdown spam, no em dashes (the character U+2014 or U+2013) in the reply."
)

_EM_DASHES = ("\u2014", "\u2013")  # em dash, en dash
_DEFAULT_MAX_MESSAGES = 6
_DEFAULT_MAX_CHARS = 2000
_DEFAULT_TTFT_MS = 8000


def _strip_em_dashes(text: str) -> str:
    for ch in _EM_DASHES:
        text = text.replace(ch, "-")
    return text


def sanitize_fast_lane_reply(text: str) -> str:
    """Post-process model output: no em/en dashes, trimmed."""
    return _strip_em_dashes((text or "").strip())


def _content_text(content: Any) -> Optional[str]:
    """Plain text from a message content field; None when missing or non-text."""
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = str(part.get("text") or "").strip()
                if t:
                    parts.append(t)
            elif isinstance(part, str) and part.strip():
                parts.append(part.strip())
        joined = "\n".join(parts).strip()
        return joined or None
    return None


def _eligible_turn(message: Any) -> Optional[Dict[str, str]]:
    """Keep user/assistant chat text; drop tools, system, and tool-call rows."""
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role not in ("user", "assistant"):
        return None
    if message.get("tool_calls"):
        return None
    text = _content_text(message.get("content"))
    if text is None:
        return None
    # Internal gateway notes / machine prefixes are not conversational context.
    if text.startswith("[hermes:") or text.startswith("[System note:"):
        return None
    return {"role": role, "content": _strip_em_dashes(text)}


def build_compact_transcript(
    history: Optional[Sequence[Any]],
    user_message: str,
    *,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> List[Dict[str, str]]:
    """Build ``[system, ...last N turns...]`` for the fast-lane provider call.

    Always includes the latest user message. Older turns are dropped first to meet
    ``max_messages`` and ``max_chars``; if the latest user message alone exceeds the
    char budget it is truncated (still kept).
    """
    max_messages = max(1, int(max_messages or _DEFAULT_MAX_MESSAGES))
    max_chars = max(1, int(max_chars or _DEFAULT_MAX_CHARS))

    turns: list[Dict[str, str]] = []
    for row in history or []:
        eligible = _eligible_turn(row)
        if eligible is not None:
            turns.append(eligible)

    latest = {
        "role": "user",
        "content": _strip_em_dashes((user_message or "").strip() or "(empty)"),
    }
    # Avoid consecutive user rows (history may already end with the inbound user turn).
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    turns.append(latest)
    # Trailing window always retains the latest user row (it is last).
    if len(turns) > max_messages:
        turns = turns[-max_messages:]

    def _chars(rows: Sequence[Dict[str, str]]) -> int:
        return sum(len(r.get("content") or "") for r in rows)

    while len(turns) > 1 and _chars(turns) > max_chars:
        turns.pop(0)

    if _chars(turns) > max_chars:
        # Latest user message alone is over budget — truncate, never drop.
        keep = max_chars
        content = turns[-1]["content"]
        turns[-1] = {
            "role": "user",
            "content": content[:keep].rstrip() + ("…" if len(content) > keep else ""),
        }

    return [{"role": "system", "content": FAST_LANE_SYSTEM}, *turns]


def load_fast_lane_config(user_config: Any = None) -> Dict[str, Any]:
    """``gateway.telegram.fast_lane`` from user YAML; defaults when absent.

    Gateway does not merge DEFAULT_CONFIG, so missing keys still mean enabled with
    the shipped budgets.
    """
    cfg: Dict[str, Any] = {
        "enabled": True,
        "provider": "",
        "model": "",
        "max_messages": _DEFAULT_MAX_MESSAGES,
        "max_chars": _DEFAULT_MAX_CHARS,
        "ttft_budget_ms": _DEFAULT_TTFT_MS,
    }
    try:
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        tg = gw.get("telegram") if isinstance(gw, dict) else None
        raw = tg.get("fast_lane") if isinstance(tg, dict) else None
        if not isinstance(raw, dict):
            return cfg
        if "enabled" in raw:
            value = raw.get("enabled")
            if isinstance(value, str):
                cfg["enabled"] = value.strip().lower() not in {"false", "0", "no", "off"}
            else:
                cfg["enabled"] = bool(value)
        for key in ("provider", "model"):
            if key in raw and raw[key] is not None:
                cfg[key] = str(raw[key]).strip()
        for key, cast in (
            ("max_messages", int),
            ("max_chars", int),
            ("ttft_budget_ms", int),
        ):
            if key in raw and raw[key] is not None:
                try:
                    cfg[key] = cast(raw[key])
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return cfg


def is_fast_lane_enabled(user_config: Any = None) -> bool:
    return bool(load_fast_lane_config(user_config).get("enabled", True))


def log_fast_lane(
    *,
    provider: str,
    ttft_ms: Optional[float],
    ready_ms: Optional[float],
    fallback: bool,
    chat_id: Any = None,
) -> None:
    """Single measurable line; mirrors telegram_delivery_receipt anonymity."""
    payload = {
        "marker": FAST_LANE_MARKER,
        "provider": provider or "unknown",
        "ttft_ms": None if ttft_ms is None else round(float(ttft_ms), 1),
        "ready_ms": None if ready_ms is None else round(float(ready_ms), 1),
        "fallback": bool(fallback),
    }
    logger.info(
        "[latency] " + FAST_LANE_MARKER
        + " provider=%s ttft_ms=%s ready_ms=%s fallback=%s chat=%s",
        payload["provider"],
        payload["ttft_ms"] if payload["ttft_ms"] is not None else "none",
        payload["ready_ms"] if payload["ready_ms"] is not None else "none",
        "true" if fallback else "false",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX),
        extra={"fast_lane": payload},
    )


def _chunk_delta_text(chunk: Any) -> str:
    """OpenAI-style streamed chunk → content delta (empty when none)."""
    try:
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            choice0 = choices[0] or {}
            delta = choice0.get("delta") or {}
            content = delta.get("content") if isinstance(delta, dict) else None
            return content if isinstance(content, str) else ""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def _consume_stream_with_ttft(
    stream: Any,
    *,
    ttft_budget_ms: int,
    on_delta: Optional[Callable[[Optional[str]], None]],
    started_at: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Consume a streamed completion.

    Returns ``(text, ttft_ms, error)``. ``error == "ttft_timeout"`` when the first
    content token exceeds the budget; any other failure sets a short reason string.
    """
    q: queue.Queue = queue.Queue()

    def _reader() -> None:
        try:
            for chunk in stream:
                q.put(("chunk", chunk))
            q.put(("done", None))
        except Exception as exc:
            q.put(("err", exc))
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    thread = threading.Thread(target=_reader, name="fast-lane-stream", daemon=True)
    thread.start()

    parts: list[str] = []
    ttft_ms: Optional[float] = None
    first = False
    budget_s = max(0.05, float(ttft_budget_ms) / 1000.0)

    def _finish(
        text: Optional[str], ttft: Optional[float], err: Optional[str],
    ) -> tuple[Optional[str], Optional[float], Optional[str]]:
        thread.join(timeout=0.5)
        return text, ttft, err

    while True:
        if not first:
            remaining = budget_s - (clock() - started_at)
            if remaining <= 0:
                return _finish(None, None, "ttft_timeout")
            timeout = remaining
        else:
            timeout = 120.0
        try:
            kind, payload = q.get(timeout=timeout)
        except queue.Empty:
            if not first:
                return _finish(None, None, "ttft_timeout")
            if parts:
                return _finish("".join(parts), ttft_ms, None)
            return _finish(None, ttft_ms, "stream_stall")

        if kind == "err":
            if parts:
                # Already streamed tokens — commit partial rather than double-send via fallback.
                return _finish("".join(parts), ttft_ms, None)
            return _finish(None, ttft_ms, f"stream_error:{type(payload).__name__}")
        if kind == "done":
            break
        text = _chunk_delta_text(payload)
        if not text:
            continue
        text = _strip_em_dashes(text)
        if not text:
            continue
        if not first:
            elapsed_ms = (clock() - started_at) * 1000.0
            if elapsed_ms > float(ttft_budget_ms):
                return _finish(None, elapsed_ms, "ttft_timeout")
            ttft_ms = elapsed_ms
            first = True
        parts.append(text)
        if on_delta is not None:
            on_delta(text)

    joined = "".join(parts)
    if not joined.strip():
        return _finish(None, ttft_ms, "empty_reply")
    return _finish(joined, ttft_ms, None)


def _is_completed_response(obj: Any) -> bool:
    """True when a stream=True call returned a finished ChatCompletion-shaped object."""
    return getattr(obj, "choices", None) is not None and not hasattr(obj, "__next__")


def try_fast_lane(
    *,
    history: Optional[Sequence[Any]],
    user_message: str,
    user_config: Any = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    on_delta: Optional[Callable[[Optional[str]], None]] = None,
    chat_id: Any = None,
    call_llm_fn: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[Dict[str, Any]]:
    """Run the compact fast-lane call, or return ``None`` to fall back.

    On success the dict mirrors a minimal ``run_conversation`` result:
    ``final_response``, ``messages`` (history + user + assistant), ``api_calls``,
    ``agent_persisted=False`` so the gateway writes the turn normally.
    """
    cfg = load_fast_lane_config(user_config)
    if not cfg.get("enabled", True):
        return None

    runtime = dict(main_runtime or {})
    provider = (cfg.get("provider") or runtime.get("provider") or "").strip()
    model = (cfg.get("model") or runtime.get("model") or "").strip()
    if not provider and not model and not runtime.get("api_key"):
        log_fast_lane(provider="none", ttft_ms=None, ready_ms=None, fallback=True, chat_id=chat_id)
        return None

    messages = build_compact_transcript(
        history,
        user_message,
        max_messages=int(cfg.get("max_messages") or _DEFAULT_MAX_MESSAGES),
        max_chars=int(cfg.get("max_chars") or _DEFAULT_MAX_CHARS),
    )
    ttft_budget_ms = int(cfg.get("ttft_budget_ms") or _DEFAULT_TTFT_MS)
    # Whole-call ceiling: TTFT budget plus room to finish a short social reply.
    call_timeout_s = max(15.0, (ttft_budget_ms / 1000.0) + 30.0)

    if call_llm_fn is None:
        from agent.auxiliary_client import call_llm as call_llm_fn  # type: ignore[assignment]

    started = clock()
    provider_label = provider or runtime.get("provider") or "main"
    # Lane-side affinity: aux `_normalize_main_runtime` drops session_id, and the
    # gateway turn thread has no ambient conversation contextvar, so call_llm's
    # shared merge would send no x-opencode-session. Attach it here; caller-pinned
    # extra_headers win over the later aux setdefault merge.
    extra_headers: Optional[Dict[str, str]] = None
    _sid = (runtime.get("session_id") or "").strip() if isinstance(runtime.get("session_id"), str) else ""
    if _sid:
        from agent.opencode_affinity import merge_opencode_session_headers

        _hdr_kwargs: Dict[str, Any] = {}
        merge_opencode_session_headers(
            _hdr_kwargs,
            provider or runtime.get("provider"),
            runtime.get("base_url"),
            _sid,
        )
        got = _hdr_kwargs.get("extra_headers")
        if isinstance(got, dict) and got:
            extra_headers = got
    try:
        stream = call_llm_fn(
            messages=messages,
            stream=True,
            max_tokens=256,
            temperature=0.7,
            timeout=call_timeout_s,
            main_runtime=runtime or None,
            provider=provider or None,
            model=model or None,
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            api_mode=runtime.get("api_mode"),
            extra_headers=extra_headers,
        )
    except Exception as exc:
        logger.info("fast_lane call setup failed: %s", type(exc).__name__)
        log_fast_lane(
            provider=provider_label, ttft_ms=None,
            ready_ms=(clock() - started) * 1000.0, fallback=True, chat_id=chat_id,
        )
        return None

    # Non-stream shims may return a completed response despite stream=True.
    if _is_completed_response(stream):
        from agent.auxiliary_client import extract_content_or_reasoning
        try:
            text = sanitize_fast_lane_reply(extract_content_or_reasoning(stream) or "")
        except Exception:
            text = ""
        ready_ms = (clock() - started) * 1000.0
        if not text:
            log_fast_lane(
                provider=provider_label, ttft_ms=ready_ms, ready_ms=ready_ms,
                fallback=True, chat_id=chat_id,
            )
            return None
        if on_delta is not None:
            on_delta(text)
        log_fast_lane(
            provider=provider_label, ttft_ms=ready_ms, ready_ms=ready_ms,
            fallback=False, chat_id=chat_id,
        )
        return _success_result(history, user_message, text)

    text, ttft_ms, err = _consume_stream_with_ttft(
        stream,
        ttft_budget_ms=ttft_budget_ms,
        on_delta=on_delta,
        started_at=started,
        clock=clock,
    )
    ready_ms = (clock() - started) * 1000.0
    if err or not text:
        log_fast_lane(
            provider=provider_label, ttft_ms=ttft_ms, ready_ms=ready_ms,
            fallback=True, chat_id=chat_id,
        )
        return None

    text = sanitize_fast_lane_reply(text) or _strip_em_dashes(text).strip() or "…"

    log_fast_lane(
        provider=provider_label, ttft_ms=ttft_ms, ready_ms=ready_ms,
        fallback=False, chat_id=chat_id,
    )
    return _success_result(history, user_message, text)


def _success_result(
    history: Optional[Sequence[Any]], user_message: str, text: str,
) -> Dict[str, Any]:
    prior = [m for m in (history or []) if isinstance(m, dict)]
    messages = list(prior) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": text},
    ]
    return {
        "final_response": text,
        "messages": messages,
        "api_calls": 1,
        "failed": False,
        "completed": True,
        "interrupted": False,
        "agent_persisted": False,
        "tools": [],
    }
