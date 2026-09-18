"""Conversational DM quiet window (2s / 5s cap) on TelegramAdapter."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._conversational_dm_batching = True
    adapter._text_batch_quiet_seconds = 2.0
    adapter._text_batch_max_wait_seconds = 5.0
    adapter._text_batch_delay_seconds = 0.3
    adapter._text_batch_split_delay_seconds = 1.0
    adapter._SPLIT_THRESHOLD = 3500
    adapter._TEXT_BATCH_FAST_LEN = 80
    adapter._TEXT_BATCH_SHORT_LEN = 400
    adapter._TEXT_BATCH_FAST_DELAY_S = 0.18
    adapter._TEXT_BATCH_SHORT_DELAY_S = 0.24
    adapter.handle_message = AsyncMock()
    return adapter


def _event(chat_type="dm"):
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type=chat_type),
    )


def test_dm_quiet_window_two_seconds():
    adapter = _adapter()
    pending = _event("dm")
    pending._batch_opened_mono = 100.0
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.4
    try:
        assert adapter._text_batch_delay_for(pending) == 2.0
    finally:
        mod.time.monotonic = orig


def test_dm_absolute_cap():
    adapter = _adapter()
    pending = _event("dm")
    pending._batch_opened_mono = 100.0
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 104.5
    try:
        assert adapter._text_batch_delay_for(pending) == 0.5
    finally:
        mod.time.monotonic = orig


def test_group_keeps_adaptive_fast_path():
    adapter = _adapter()
    pending = _event("group")
    delay = adapter._text_batch_delay_for(pending)
    assert delay <= 0.3


def test_quiet_window_knobs_have_class_defaults_so_the_flush_task_cannot_crash():
    """Every knob ``_text_batch_delay_for`` reads must resolve without ``__init__``.

    It runs inside the flush task, where an AttributeError becomes a never-retrieved task
    exception and the buffered burst is dropped silently. The class defaults must also agree
    with what ``__init__`` writes, or the pre-init window would batch on different rules.
    """
    from plugins.platforms.telegram.adapter import TelegramAdapter

    pristine = object.__new__(TelegramAdapter)
    delay = pristine._text_batch_delay_for(_event("dm"))
    assert isinstance(delay, float) and delay >= 0.0

    configured = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    for knob in ("_conversational_dm_batching", "_text_batch_quiet_seconds", "_text_batch_max_wait_seconds"):
        assert getattr(pristine, knob) == getattr(configured, knob)
