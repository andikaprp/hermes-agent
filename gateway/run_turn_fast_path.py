"""No-task conversational fast path: answer a bare ack in one model hop, with no tool round.

LAB-3 measured 67 of 100 live Telegram turns spending more than one model call, and only 11 of
44 casual turns hitting "one call, no tools". Unpaired medians by call count were 12.4 s for
one call, 28.7 s for two and 32.6 s for three, so a social reply that wanders into the agentic
loop costs the user two to three times what it should. The worst recorded case was ``yess``
answered with 24 model calls, 23 tool calls and 206 s of wall clock.

**Why this is a note on the user message and not a smaller tool schema.** Per-conversation
prompt caching is the project's sacred invariant, and the cached prefix is system prompt +
tools + history. Dropping the tool array for one turn, or sending ``tool_choice="none"``,
changes bytes inside that prefix: ``agent/chat_completion_helpers.py`` already documents that
SGLang renders the prompt with ``tools=None`` under ``tool_choice="none"`` and the KV prefix
diverges. A turn that alternates between "tools" and "no tools" would pay a full cache miss in
both directions, which on a long-lived DM costs far more than the calls it saves. The user
message sits at the END of the prefix, so a note prepended there leaves every cached byte
before it intact — the same seam ``_prepend_pending_note`` already uses for model-switch and
``/reload-skills`` notices, and the seam ``agent/AGENTS.md`` names for mid-conversation
injection.

**Why it is a note and not a hard block.** Keeping the tools on the wire means a misrouted
turn degrades to "the model ignored a hint", never to "the work was silently dropped". The
note says so explicitly, which is the fall-through-and-say-so escape hatch.

The gate is structural, then lexical — never the other way around. A longer ack list cannot
tell ``yes`` from ``yes merge it``: both start with the same word. Eligibility is therefore
a sequence of fail-closed shape checks, and only a message that survives every one of them
may consult the frozen ack/social lexicons:

* a URL is never ordinary;
* a bracketed or tagged prefix (``[ASYNC …]``, ``[IMPORTANT: …]``) is a machine work
  trigger, never ordinary. Detected on the raw text, before any punctuation strip, because
  stripping ``[]`` would turn ``[ok]`` into an ack;
* an action verb (merge, run, restart, fix, upgrade, …) anywhere means the message is a
  command, regardless of any ack word sitting next to it;
* an ack/directive/social phrase with anything attached — a verb, a noun, a target — is a
  command. ``yes`` has no object; ``yes, patch it`` does;
* a question that has to consult the world or some named state (``what time is it in …``,
  ``how is it going with the …``) is never ordinary. A trailing ``?`` on a known bare phrase
  (``ok?``) is decoration, not a question;
* directive words (``continue``, ``do it``, ``proceed``, ``done``, …) are never eligible.
  ``done`` is deliberate: a user reporting a finished step usually expects the next one;
* a surviving ack word is eligible only when the assistant's last message did not propose
  or ask for anything, because there an ack IS a go-ahead;
* slash commands and native multimodal content are never eligible.

Do not extend ``_ACK_WORDS`` to chase recall. Ambiguous probes (``test``, ``this``,
``ping``) stay off. Ack words may reach past ``is_trivial_prompt`` (that regex is
English-only; this deployment's primary user sends ``ya`` / ``oke`` / ``okey``). Social
(non-ack) turns still require the trivial-prompt judgement.

Media, quoted replies and group sender prefixes need no separate guard: inbound preprocessing
folds each of them into the message text as a note, and a leading ``[`` is already rejected.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from agent.memory_provider import is_trivial_prompt
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

FAST_PATH_MARKER = "fast_path_turn"
FAST_PATH_OUTCOME_MARKER = "fast_path_outcome"

# config.yaml: gateway.telegram.fast_path (default on). The gateway loads user YAML with no
# DEFAULT_CONFIG merge, so absence of the key MUST still mean enabled — a missing nested dict
# is not an opt-out.
CONFIG_KEY = "gateway.telegram.fast_path"

# Platforms whose DMs route through the fast path. Telegram is where LAB-3 measured the loss;
# the classifier itself is platform-agnostic, so widening this is a one-line change once
# another platform has the numbers to justify it.
FAST_PATH_PLATFORMS = frozenset({"telegram"})
_DM_CHAT_TYPES = frozenset({"dm", "private"})

# Trivial words that instruct work. "done" and "next" are included: from a user they usually
# report a finished step and expect the agent to take the following one. Do not move "done"
# onto the fast path — that is a measured, deliberate exclusion (see the test of the same name).
_DIRECTIVE_WORDS = frozenset({"continue", "go ahead", "do it", "proceed", "next", "done"})

# Acknowledgements that are an approval only in context. Safe ALONE; a go-ahead when the
# assistant just proposed something, which ``_ASSISTANT_PROPOSED_RE`` detects.
# Frozen. Do not add words here to chase ordinary-recall — ``yes`` and ``yes merge it``
# share the same first token, and a longer list cannot tell them apart.
# ``ya`` / ``oke`` / ``okey`` / ``iya`` are Indonesian (and mixed-ID/EN) equivalents of yes/ok;
# they are not in ``TRIVIAL_PROMPT_RE`` (English-only) and match via this set directly.
_ACK_WORDS = frozenset({
    "yes", "y", "yep", "yup", "yeah", "ok", "okay", "oke", "okey", "k", "sure",
    "no", "n", "nope", "nah", "ya", "iya",
    "got it", "lgtm", "fine", "good", "not really",
})

# Imperative verbs of action. Independent of the ack list: a message that contains one of
# these as a whole word is a command even if it also contains ``yes`` / ``ok``.
_ACTION_VERBS = frozenset({
    "merge", "run", "restart", "fix", "upgrade", "deploy", "build", "check",
    "summarize", "update", "delete", "install", "commit", "push", "review", "report",
})
_ACTION_VERB_RE = re.compile(
    r"\b(?:" + "|".join(sorted(re.escape(v) for v in _ACTION_VERBS)) + r")\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

# Machine-generated work triggers arrive as a leading [tag] (delegation batches, cron
# completions, background-process notices). Must run on the raw text: ``_TRAILING``
# contains ``[]``, so a later strip would turn ``[ok]`` into an ack.
_SYSTEM_PREFIX_RE = re.compile(r"^\[")

# Question-words. Auxiliaries (do/is/are) are omitted: ``do it`` is an imperative, and a
# message that is not a known bare phrase already falls through to the full loop.
_INTERROGATIVE_RE = re.compile(
    r"^(what|what's|whats|why|how|how's|hows|when|where|who|which|"
    r"can|could|would|should)\b",
    re.IGNORECASE,
)

# The assistant's last message invited an action, so an ack answers it.
_ASSISTANT_PROPOSED_RE = re.compile(
    r"(shall i\b|should i\b|want me to\b|would you like me to\b|do you want me to\b"
    r"|let me know\b|confirm\b|ready to\b|i can\b|proceed\?|go ahead\?)",
    re.IGNORECASE,
)

# Trailing decoration TRIVIAL_PROMPT_RE tolerates, stripped to recover the bare word.
_TRAILING = " \t\n!?.:;,\"'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()[]{}<>*&^%$#@+=`\u00a0"

# Chat elongation: "yess", "okkk", "hiii". TRIVIAL_PROMPT_RE does not match these (it allows
# trailing punctuation, not a repeated letter), which is why LAB-3's worst recorded turn --
# ``yess`` answered with 24 model calls -- read as a non-trivial prompt. Collapsing a repeated
# RUN AT THE END only is deliberately narrow: it cannot turn a real instruction into an ack.
_ELONGATED_TAIL = re.compile(r"(.)\1+$")

FAST_PATH_NOTE = (
    "[hermes:fast-path] The message below is a short social reply that carries no task. "
    "Answer it directly in one short message and call no tools. If it does turn out to need a "
    "tool, a lookup or a file, ignore this note, say plainly what you are doing, and do the work."
)


def _bare_word(text: str) -> str:
    """The lexicon key for *text*: decoration stripped, trailing elongation collapsed."""
    return _ELONGATED_TAIL.sub(r"\1", text.strip().strip(_TRAILING).lower())


def _tokens(text: str) -> list[str]:
    """Whitespace tokens with per-token decoration and trailing elongation removed."""
    out: list[str] = []
    for raw in text.split():
        tok = _ELONGATED_TAIL.sub(r"\1", raw.strip(_TRAILING).lower())
        if tok:
            out.append(tok)
    return out


def _known_leading_len(tokens: Sequence[str]) -> int:
    """Longest leading ack / directive / trivial phrase length, else 0."""
    max_n = min(3, len(tokens))
    for n in range(max_n, 0, -1):
        phrase = " ".join(tokens[:n])
        if phrase in _ACK_WORDS or phrase in _DIRECTIVE_WORDS or is_trivial_prompt(phrase):
            return n
    return 0


def _has_attached_object(tokens: Sequence[str]) -> bool:
    """True when a known bare phrase is followed by any extra token.

    ``yes`` is an acknowledgement. ``yes merge it`` is an order. The two are made of
    the same ack word; the extra token is the object that makes it a command.
    """
    n = _known_leading_len(tokens)
    return n > 0 and len(tokens) > n


def _is_consultative(stripped: str, tokens: Sequence[str]) -> bool:
    """World/state questions stay on the full loop; ``ok?`` is just a decorated ack."""
    phrase = " ".join(tokens)
    if (
        phrase in _ACK_WORDS
        or phrase in _DIRECTIVE_WORDS
        or is_trivial_prompt(stripped)
        or is_trivial_prompt(phrase)
    ):
        return False
    return bool(_INTERROGATIVE_RE.match(stripped) or "?" in stripped)


def _assistant_proposed(history: Optional[Sequence[Any]]) -> bool:
    """True when the last assistant message asked for, or offered to do, something.

    Unknown shapes read as "proposed": the gate must fail towards the normal loop.
    """
    for message in reversed(list(history or [])):
        if not isinstance(message, dict):
            return True
        role = message.get("role")
        if role != "assistant":
            if role in ("tool", "system"):
                continue
            # A user row directly before this turn means the assistant never replied.
            return False
        if message.get("tool_calls"):
            return True
        content = message.get("content")
        if not isinstance(content, str):
            return True
        stripped = content.strip()
        return stripped.endswith("?") or bool(_ASSISTANT_PROPOSED_RE.search(stripped))
    return False


def is_fast_path_enabled(user_config: Any = None) -> bool:
    """``gateway.telegram.fast_path`` in config.yaml; default on.

    The gateway reads user YAML with no DEFAULT_CONFIG merge, so a missing key is enabled,
    not off. Explicit ``false``/``off``/``0``/``no`` disables without a code change.
    """
    try:
        gw = user_config.get("gateway") if isinstance(user_config, dict) else None
        tg = gw.get("telegram") if isinstance(gw, dict) else None
        if not isinstance(tg, dict) or "fast_path" not in tg:
            return True
        value = tg.get("fast_path")
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def classify_fast_path(
    message: Any, *, history: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    """Return ``"social"`` / ``"ack"`` when this turn needs no tools, else ``None``.

    ``history`` is the conversation the agent is about to see, newest last.
    """
    if not isinstance(message, str):
        return None  # native multimodal content
    stripped = message.strip()
    if not stripped or stripped.startswith("/"):
        return None
    # Shape checks first — the lexicon never sees a URL, a tagged prefix, a verb of
    # action, an ack-with-object, or a consultative question.
    if _URL_RE.search(stripped) or _SYSTEM_PREFIX_RE.match(stripped):
        return None
    if _ACTION_VERB_RE.search(stripped):
        return None
    tokens = _tokens(stripped)
    if not tokens or _has_attached_object(tokens) or _is_consultative(stripped, tokens):
        return None
    phrase = " ".join(tokens)
    if phrase in _DIRECTIVE_WORDS:
        return None
    if phrase in _ACK_WORDS:
        return None if _assistant_proposed(history) else "ack"
    if is_trivial_prompt(stripped) or is_trivial_prompt(phrase):
        return "social"
    return None


def fast_path_reason(
    message: Any, *, platform_key: Optional[str], chat_type: Optional[str],
    history: Optional[Sequence[Any]] = None, user_config: Any = None,
) -> Optional[str]:
    """``classify_fast_path`` restricted to the surfaces the fast path is enabled on."""
    if not is_fast_path_enabled(user_config):
        return None
    if str(platform_key or "").lower() not in FAST_PATH_PLATFORMS:
        return None
    if str(chat_type or "").lower() not in _DM_CHAT_TYPES:
        return None
    return classify_fast_path(message, history=history)


def apply_fast_path_note(message: str, reason: str, *, chat_id: Any = None) -> str:
    """Prepend the note and log the routing decision so it is countable from the logs."""
    logger.info(
        "[latency] " + FAST_PATH_MARKER + " chat=%s reason=%s",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX), reason,
        extra={"fast_path": {"marker": FAST_PATH_MARKER, "reason": reason}},
    )
    return FAST_PATH_NOTE + "\n\n" + message


def _count_tool_calls(result: Any) -> int:
    n = 0
    for message in (result or {}).get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if isinstance(calls, list):
            n += len(calls)
    return n


def log_fast_path_outcome(reason: str, result: Any, *, chat_id: Any = None) -> None:
    """Turn-end marker: whether the fast path was taken, and how many calls it used.

    LAB-3 could not verify "one model call, zero tools" from logs because only the routing
    decision was visible. This line closes that: ``api_calls`` and ``tool_calls`` are the
    turn's real counts, so the next measurement pass can score compliance without guessing.
    """
    api_calls = int((result or {}).get("api_calls") or 0)
    tool_calls = _count_tool_calls(result)
    payload = {
        "marker": FAST_PATH_OUTCOME_MARKER,
        "reason": reason,
        "api_calls": api_calls,
        "tool_calls": tool_calls,
    }
    logger.info(
        "[latency] " + FAST_PATH_OUTCOME_MARKER + " chat=%s reason=%s api_calls=%d tool_calls=%d",
        redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX), reason, api_calls, tool_calls,
        extra={"fast_path": payload},
    )
