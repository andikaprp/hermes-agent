"""Redacted, local-only delivery receipts for the real Telegram outbound send seam.

One receipt is emitted per platform send attempt, at the point where the adapter
hands a message to the Bot API. The carrier is deliberately anonymous — it records
only the enumerated fields below:

* ``mono``    — ``time.monotonic()`` at emit time (seconds, ms precision)
* ``chat``    — short keyed digest of the chat id (never the raw id)
* ``attempt`` — 1-based attempt number; ``0`` = refused locally, no platform call
* ``mid``     — short keyed digest of the platform message id Telegram returned
                (``none`` when no id came back)
* ``anchor``  — ``present``/``absent``: a ``reply_to_message_id`` anchor was on the send
* ``thread``  — ``present``/``absent``: a thread routing id (``message_thread_id``, or
                the DM-topic ``direct_messages_topic_id`` when that route is used)
* ``outcome`` — ``success``/``failure``

There is no field for message text, user content, tokens, or error strings, and no
generic ``**kwargs`` sink: a receipt can only be built from the keywords above, so a
later edit cannot accidentally append a payload. Digests share the per-process salt
with :mod:`gateway.delivery_outcome` so Telegram receipts and gateway-wide outcomes
correlate inside one process (deliberate trade: correlatable inside a process, not
enumerable across logs — an unsalted digest of a numeric Telegram id is
brute-forceable in seconds). Emitting never raises: a logging failure must not break a
live send.

Scope: the seams that deliver a *new* message — the legacy ``sendMessage`` chunk path
and the Bot API 10.1 ``sendRichMessage`` path. Edits, ephemeral draft frames, and
control-style sends (prompts/pickers) deliver no new message and emit nothing.

Telegram ``outcome=success`` means provider acceptance (a ``message_id``) — gateway
stage ``delivered``, never ``confirmed`` (bots have no user-read acknowledgement here).
See :class:`gateway.delivery_outcome.DeliveryOutcome` for the shared stage model.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Callable, Optional

# Digests live on the shared gateway outcome module so every channel shares one salt.
from gateway.delivery_outcome import (  # noqa: F401 — re-export for existing importers
    CHAT_DIGEST_PREFIX,
    DIGEST_CHARS,
    MESSAGE_DIGEST_PREFIX,
    MISSING_TOKEN,
    redacted_token,
)

logger = logging.getLogger("gateway.telegram_delivery_receipt")

RECEIPT_LOG_PREFIX = "telegram_delivery_receipt"
PRESENT = "present"
ABSENT = "absent"
SUCCESS = "success"
FAILURE = "failure"


def _presence(value: Any) -> str:
    return PRESENT if value else ABSENT


class TelegramDeliveryReceipt:
    """One send attempt's anonymous receipt. Emits at most once, and never raises.

    Emit on the success path with :meth:`succeeded`, then call :meth:`failed` from a
    ``finally`` so every terminal path (return, retry ``continue``, or raise) is
    recorded exactly once — the emit is idempotent.
    """

    __slots__ = ("_clock", "_chat", "_attempt", "_anchor", "_thread", "_emitted", "_logger")

    def __init__(
        self,
        *,
        chat_id: Any,
        attempt: int,
        reply_anchor: Any,
        thread_id: Any,
        clock: Callable[[], float] = time.monotonic,
        receipt_logger: Optional[logging.Logger] = None,
    ) -> None:
        self._clock = clock
        self._chat = redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX)
        self._attempt = int(attempt)
        self._anchor = _presence(reply_anchor)
        self._thread = _presence(thread_id)
        self._emitted = False
        self._logger = receipt_logger or logger

    @property
    def emitted(self) -> bool:
        return self._emitted

    def succeeded(self, message_id: Any = None) -> None:
        """Record a delivered send (Telegram returned ``message_id``)."""
        self._emit(SUCCESS, message_id)

    def failed(self) -> None:
        """Record a failed attempt; no-op when a receipt for this attempt already emitted."""
        self._emit(FAILURE, None)

    def fields(self, outcome: str, message_id: Any = None) -> "dict[str, Any]":
        """The receipt payload — exactly the allowlisted keys, nothing derived from content."""
        return {
            "mono": round(float(self._clock()), 3),
            "chat": self._chat,
            "attempt": self._attempt,
            "mid": redacted_token(message_id, prefix=MESSAGE_DIGEST_PREFIX),
            "anchor": self._anchor,
            "thread": self._thread,
            "outcome": outcome,
        }

    def render(self, outcome: str, message_id: Any = None) -> str:
        """Log line body: ``telegram_delivery_receipt mono=… chat=… … outcome=…``."""
        payload = self.fields(outcome, message_id)
        return RECEIPT_LOG_PREFIX + " " + " ".join(f"{key}={value}" for key, value in payload.items())

    def _emit(self, outcome: str, message_id: Any) -> None:
        if self._emitted:
            return
        self._emitted = True
        payload = self.fields(outcome, message_id)
        with contextlib.suppress(Exception):
            self._logger.info(
                RECEIPT_LOG_PREFIX + " %s",
                " ".join(f"{key}={value}" for key, value in payload.items()),
                extra={"delivery_receipt": payload},
            )
