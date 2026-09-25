"""Time-to-first-content markers for a streamed gateway turn.

LAB-3 found that time-to-first-token was not derivable from any shipped log: grepping the
gateway and agent logs for ``ttft``/``time_to_first_token``/``first_chunk`` returned nothing,
so "the user sat in silence for N seconds" could only be inferred from the delivery receipt of
what turned out to be the turn's ONLY message. These two markers close that gap by recording
the two instants the receipt cannot show:

* ``stream_first_delta``  — the provider's first streamed token reached the consumer
* ``stream_first_visible`` — the first frame/send actually reached the platform

``since_first_delta_ms`` on the second line is the consumer's own contribution to perceived
latency (debounce + transport), separate from the provider's think time, which is
``since_open_ms`` on the first line.

Chat ids are redacted with the same keyed per-process digest the Telegram delivery receipts
use, so a marker correlates with its receipts inside one gateway process without publishing a
raw chat id to the log. Emitting never raises: a logging failure must not break a live turn.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Callable, Optional

from gateway.telegram_delivery_receipt import CHAT_DIGEST_PREFIX, redacted_token

logger = logging.getLogger("gateway.stream_consumer")

FIRST_DELTA_MARKER = "stream_first_delta"
FIRST_VISIBLE_MARKER = "stream_first_visible"


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


class StreamLatencyMarks:
    """Per-consumer first-delta / first-visible stopwatch; each marker emits at most once."""

    __slots__ = ("_clock", "_turn_id", "_chat", "_opened", "_first_delta", "_first_visible", "_logger")

    def __init__(
        self,
        *,
        turn_id: str,
        chat_id: Any,
        clock: Callable[[], float] = time.monotonic,
        marker_logger: Optional[logging.Logger] = None,
    ) -> None:
        self._clock = clock
        self._turn_id = turn_id
        self._chat = redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX)
        self._opened = clock()
        self._first_delta: Optional[float] = None
        self._first_visible: Optional[float] = None
        self._logger = marker_logger or logger

    @property
    def first_delta_seen(self) -> bool:
        return self._first_delta is not None

    @property
    def first_visible_seen(self) -> bool:
        return self._first_visible is not None

    def note_delta(self) -> None:
        """Record (once) that the first streamed token arrived."""
        if self._first_delta is not None:
            return
        self._first_delta = self._clock()
        self._emit(FIRST_DELTA_MARKER, {"since_open_ms": _ms(self._first_delta - self._opened)})

    def note_visible(self) -> None:
        """Record (once) that the first frame/send reached the platform.

        Callers emit this from the drain loop, where every transport (native frame, draft
        frame, plain first send) has converged on one success path.
        """
        if self._first_visible is not None:
            return
        self._first_visible = self._clock()
        fields = {"since_open_ms": _ms(self._first_visible - self._opened)}
        if self._first_delta is not None:
            fields["since_first_delta_ms"] = _ms(self._first_visible - self._first_delta)
        self._emit(FIRST_VISIBLE_MARKER, fields)

    def _emit(self, marker: str, fields: "dict[str, Any]") -> None:
        payload = {"turn": self._turn_id, "chat": self._chat, **fields}
        with contextlib.suppress(Exception):
            self._logger.info(
                "[latency] " + marker + " %s",
                " ".join(f"{key}={value}" for key, value in payload.items()),
                extra={"stream_latency": {"marker": marker, **payload}},
            )
