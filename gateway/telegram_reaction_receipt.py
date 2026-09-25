"""Redacted, local-only reaction receipts for the real Telegram native-reaction seam.

One receipt is emitted per *attempted* reaction, at the point the adapter hands the
request to the Bot API (``setMessageReaction``). The carrier is deliberately
anonymous — it records only the enumerated fields below:

* ``mono``          — ``time.monotonic()`` at emit time (seconds, ms precision)
* ``chat``          — short keyed digest of the chat id (never the raw id)
* ``mid``           — short keyed digest of the reacted-to message id (never the raw id)
* ``emoji``         — the glyph cluster, ``none`` for a clear, ``other`` when the payload
                      could carry text (see :func:`safe_emoji_token`)
* ``phase``         — which lifecycle step attempted it: ``start`` / ``complete`` /
                      ``cancelled``, or ``direct`` for an agent-intent react
* ``outcome``       — ``success``/``failed``
* ``failure_class`` — exception class name of a failed attempt (``none`` on success),
                      sanitized to ``[A-Za-z0-9_]``; local refusals use an allowlisted
                      token (``no_bot``) and anything unrecognized is ``unknown``

There is no field for message text, user content, tokens, or error message strings, and
no generic ``**kwargs`` sink: a receipt can only be built from the keywords above, so a
later edit cannot accidentally append a payload. The chat/message digests come from
:func:`gateway.telegram_delivery_receipt.redacted_token` — one digest convention and one
per-process keyed salt, so a chat renders the same token in the delivery and reaction
receipt streams of a single gateway process (correlatable inside a process, not
enumerable across logs; an unsalted digest of a numeric Telegram id is brute-forceable
in seconds).

Fail-open by construction: construction, field building, and emit are all total
(``contextlib.suppress`` / except-guarded), because a receipt must never break a
delivery path — a reaction is cosmetic and its bookkeeping even more so.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Callable, Optional

from gateway.telegram_delivery_receipt import (
    CHAT_DIGEST_PREFIX,
    MESSAGE_DIGEST_PREFIX,
    MISSING_TOKEN,
    redacted_token,
)

logger = logging.getLogger("gateway.telegram_reaction_receipt")

RECEIPT_LOG_PREFIX = "telegram_reaction_receipt"

# Lifecycle phases. ``direct`` is NOT a lifecycle phase: it marks an explicit
# agent/API react (add_reaction / remove_reaction) so a log reader never mistakes
# agent intent for the automatic ack/outcome swap.
PHASE_START = "start"
PHASE_COMPLETE = "complete"
PHASE_CANCELLED = "cancelled"
PHASE_DIRECT = "direct"
PHASE_OTHER = "other"
PHASES = frozenset({PHASE_START, PHASE_COMPLETE, PHASE_CANCELLED, PHASE_DIRECT})

SUCCESS = "success"
FAILED = "failed"

# A recorded emoji must be a glyph cluster, never a text payload: the react paths accept
# caller-supplied strings (the agent can pass anything as an emoji), and a Bot-API
# rejection must not turn that string into a log line. Only a short all-non-ASCII cluster
# is echoed; everything else is recorded as ``other``.
EMOJI_TOKEN_LIMIT = 8
OTHER_EMOJI = "other"

# Allowlisted local failure tokens (an attempt refused before the Bot API call). The
# ``failed()`` string rung accepts ONLY these, so no caller string can reach the log.
LOCAL_NO_BOT = "no_bot"
LOCAL_FAILURE_TOKENS = frozenset({LOCAL_NO_BOT})
FAILURE_UNKNOWN = "unknown"
FAILURE_CLASS_MAX = 64


def _digest(value: Any, prefix: str) -> str:
    """``redacted_token`` that degrades to ``none`` instead of raising."""
    try:
        return redacted_token(value, prefix=prefix)
    except Exception:
        return MISSING_TOKEN


def safe_emoji_token(value: Any) -> str:
    """Emoji field value: the glyph cluster, ``none`` for a clear, ``other`` if it could be text.

    ``None``/blank → ``none`` (a clear, the documented Bot API way). A short (≤
    :data:`EMOJI_TOKEN_LIMIT` codepoints) all-non-ASCII cluster is echoed verbatim so
    👀/👍/👎 or an agent-chosen emoji stay auditable. Anything containing ASCII (letters,
    digits, punctuation — i.e. a string that can spell a message or a URL) or exceeding
    the limit degrades to ``other``: the receipt can report that a reaction was attempted
    without becoming a channel for content.
    """
    try:
        if value is None:
            return MISSING_TOKEN
        text = str(value)
        if not text.strip():
            return MISSING_TOKEN
        if len(text) > EMOJI_TOKEN_LIMIT or any(ord(character) < 128 for character in text):
            return OTHER_EMOJI
        return text
    except Exception:
        return OTHER_EMOJI


def failure_class_token(error: Any) -> str:
    """Failure class for the log: the exception's class name, else an allowlisted local token.

    Never a message string: an ``Exception`` contributes only ``type(error).__name__``
    (sanitized, bounded), ``None`` contributes ``unknown``, and a bare string contributes
    itself only when it is one of :data:`LOCAL_FAILURE_TOKENS`.
    """
    if error is None:
        return FAILURE_UNKNOWN
    if isinstance(error, str):
        return error if error in LOCAL_FAILURE_TOKENS else FAILURE_UNKNOWN
    name = ""
    with contextlib.suppress(Exception):
        name = type(error).__name__
    name = "".join(
        character
        for character in name
        if character == "_"
        or "a" <= character <= "z"
        or "A" <= character <= "Z"
        or "0" <= character <= "9"
    )[:FAILURE_CLASS_MAX]
    return name or FAILURE_UNKNOWN


def phase_token(phase: Any) -> str:
    """An unrecognized phase degrades to ``other`` rather than leaking the raw value."""
    try:
        return phase if phase in PHASES else PHASE_OTHER
    except Exception:
        return PHASE_OTHER


class TelegramReactionReceipt:
    """One attempted reaction's anonymous receipt. Emits at most once, and never raises.

    Emit on the success path with :meth:`succeeded`, and from the failure path with
    :meth:`failed` (the emit is idempotent, so a retry/log-then-fail ordering cannot
    double-log).
    """

    __slots__ = ("_clock", "_chat", "_mid", "_emoji", "_phase", "_emitted", "_logger")

    def __init__(
        self,
        *,
        chat_id: Any,
        message_id: Any,
        emoji: Any,
        phase: Any,
        clock: Callable[[], float] = time.monotonic,
        receipt_logger: Optional[logging.Logger] = None,
    ) -> None:
        self._clock = clock
        self._chat = _digest(chat_id, CHAT_DIGEST_PREFIX)
        self._mid = _digest(message_id, MESSAGE_DIGEST_PREFIX)
        self._emoji = safe_emoji_token(emoji)
        self._phase = phase_token(phase)
        self._emitted = False
        self._logger = receipt_logger or logger

    @property
    def emitted(self) -> bool:
        return self._emitted

    def succeeded(self) -> None:
        """Record a reaction the Bot API accepted."""
        self._emit(SUCCESS, None)

    def failed(self, error: Any = None) -> None:
        """Record a failed attempt; no-op when a receipt already emitted.

        ``error`` is an exception (class name is recorded) or one of
        :data:`LOCAL_FAILURE_TOKENS` for a refusal made before the platform call.
        """
        self._emit(FAILED, failure_class_token(error))

    def fields(self, outcome: str, failure_class: Optional[str] = None) -> "dict[str, Any]":
        """The receipt payload — exactly the allowlisted keys, nothing derived from content."""
        return {
            "mono": self._mono(),
            "chat": self._chat,
            "mid": self._mid,
            "emoji": self._emoji,
            "phase": self._phase,
            "outcome": outcome,
            "failure_class": failure_class or MISSING_TOKEN,
        }

    def render(self, outcome: str, failure_class: Optional[str] = None) -> str:
        """Log line body: ``telegram_reaction_receipt mono=… chat=… … outcome=…``."""
        payload = self.fields(outcome, failure_class)
        return RECEIPT_LOG_PREFIX + " " + " ".join(f"{key}={value}" for key, value in payload.items())

    def _mono(self) -> float:
        try:
            return round(float(self._clock()), 3)
        except Exception:
            return 0.0

    def _emit(self, outcome: str, failure_class: Optional[str]) -> None:
        if self._emitted:
            return
        self._emitted = True
        with contextlib.suppress(Exception):
            payload = self.fields(outcome, failure_class)
            self._logger.info(
                RECEIPT_LOG_PREFIX + " %s",
                " ".join(f"{key}={value}" for key, value in payload.items()),
                extra={"reaction_receipt": payload},
            )
