"""Private, local-only monotonic timing for gateway turns.

The carrier deliberately has no identity, content, route, tool, or provider fields.
It only emits named elapsed milliseconds at DEBUG level.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

logger = logging.getLogger("gateway.timing")
_current_turn_timing: ContextVar["TurnTiming | None"] = ContextVar("gateway_turn_timing", default=None)


@contextmanager
def bind_turn_timing(timing: "TurnTiming") -> Iterator[None]:
    token = _current_turn_timing.set(timing)
    try:
        yield
    finally:
        _current_turn_timing.reset(token)


def current_turn_timing() -> "TurnTiming":
    return _current_turn_timing.get() or TurnTiming()


class TurnTiming:
    """One turn's monotonic timing carrier, safe to attach to an inbound event."""

    required_phases = (
        "telegram_quiet_window",
        "adapter_wait",
        "queue_wait",
        "gateway_prep",
        "cached_agent_model_resolution",
        "context_memory_pre_llm",
        "api_start",
        "api_first_chunk",
        "api_end",
        "tool_time",
        "final_delivery",
    )

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._last = self._started
        self._marks: dict[str, float] = {}
        self._logged = False

    def mark(self, phase: str) -> None:
        if phase not in self.required_phases or phase in self._marks:
            return
        now = max(self._last, self._clock())
        self._last = now
        self._marks[phase] = now

    def snapshot(self) -> "OrderedDict[str, int | None]":
        previous = self._started
        values: "OrderedDict[str, int | None]" = OrderedDict()
        for phase in self.required_phases:
            marked = self._marks.get(phase)
            if marked is None:
                values[f"{phase}_ms"] = None
            else:
                values[f"{phase}_ms"] = max(0, round((marked - previous) * 1000))
                previous = marked
        return values

    def finish_delivery(self) -> None:
        """Mark ``final_delivery`` at the gateway deliver-decision point, then emit once."""
        self.mark("final_delivery")
        self.log_terminal()

    def log_terminal(self) -> None:
        """Emit only fixed phase names and elapsed milliseconds; never caller data."""
        if self._logged:
            return
        self._logged = True
        values = self.snapshot()
        logger.debug(
            "gateway_turn_timing %s",
            " ".join(f"{name}={value if value is not None else 'na'}" for name, value in values.items()),
            extra={"turn_timing": values},
        )
