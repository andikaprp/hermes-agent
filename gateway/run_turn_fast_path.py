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

The gate is deliberately narrower than the thing it reuses. Eligibility requires
``agent.memory_provider.is_trivial_prompt`` — the shipped single source of truth for "this
prompt carries no semantic signal", already trusted to skip memory recall — and then removes
from it everything that could be an instruction to act:

* directive words (``continue``, ``do it``, ``proceed``, ...) are never eligible;
* ack words (``yes``, ``ok``, ``sure``, ...) are eligible only when the assistant's last
  message did not propose or ask for anything, because there an ack IS a go-ahead;
* slash commands are never eligible;
* anything that is not a plain string (native multimodal content) is never eligible.

Media, quoted replies and group sender prefixes need no separate guard: inbound preprocessing
folds each of them into the message text as a note, and ``TRIVIAL_PROMPT_RE`` is anchored, so
a turn carrying any of them simply stops matching.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from agent.memory_provider import is_trivial_prompt
from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.run_turn")

FAST_PATH_MARKER = "fast_path_turn"

# Platforms whose DMs route through the fast path. Telegram is where LAB-3 measured the loss;
# the classifier itself is platform-agnostic, so widening this is a one-line change once
# another platform has the numbers to justify it.
FAST_PATH_PLATFORMS = frozenset({"telegram"})
_DM_CHAT_TYPES = frozenset({"dm", "private"})

# Trivial words that instruct work. "done" and "next" are included: from a user they usually
# report a finished step and expect the agent to take the following one.
_DIRECTIVE_WORDS = frozenset({"continue", "go ahead", "do it", "proceed", "next", "done"})

# Trivial words that are an approval only in context. Safe alone; a go-ahead when the assistant
# just proposed something, which ``_ASSISTANT_PROPOSED_RE`` detects.
_ACK_WORDS = frozenset({
    "yes", "y", "yep", "yeah", "ok", "okay", "k", "sure", "no", "n", "nope", "nah",
    "got it", "lgtm",
})

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
    word = _bare_word(stripped)
    if not (is_trivial_prompt(stripped) or is_trivial_prompt(word)):
        return None
    if word in _DIRECTIVE_WORDS:
        return None
    if word in _ACK_WORDS:
        return None if _assistant_proposed(history) else "ack"
    return "social"


def fast_path_reason(
    message: Any, *, platform_key: Optional[str], chat_type: Optional[str],
    history: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    """``classify_fast_path`` restricted to the surfaces the fast path is enabled on."""
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
