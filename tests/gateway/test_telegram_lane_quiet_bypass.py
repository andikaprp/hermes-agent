"""Quiet-window bypass for lane-eligible single-bubble Telegram DMs.

A single ack/social DM that the LAB-3 shape classifier would route onto the
fast path (and LAB-52 lane) must not pay ``text_batch_quiet_seconds``. Multi-
bubble bursts and non-eligible text keep the full coalescing window.
"""
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter(**kwargs):
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
    adapter._fast_lane_quiet_bypass = True
    for key, value in kwargs.items():
        setattr(adapter, key, value)
    return adapter


def _event(text="hi", chat_type="dm", bubbles=1):
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type=chat_type),
    )
    event._batch_opened_mono = 100.0
    event._batch_bubble_count = bubbles
    return event


def test_lane_eligible_single_bubble_dm_skips_quiet_window():
    """Ack/social single bubble → delay 0 (no 2s quiet wait)."""
    adapter = _adapter()
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.0
    try:
        assert adapter._text_batch_delay_for(_event("ok", bubbles=1)) == 0.0
        assert adapter._text_batch_delay_for(_event("hi", bubbles=1)) == 0.0
        assert adapter._text_batch_delay_for(_event("ya", bubbles=1)) == 0.0
    finally:
        mod.time.monotonic = orig


def test_non_eligible_single_bubble_keeps_full_quiet_window():
    adapter = _adapter()
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.4
    try:
        assert adapter._text_batch_delay_for(
            _event("please merge the PR", bubbles=1)) == 2.0
        assert adapter._text_batch_delay_for(
            _event("what time is it in Tokyo?", bubbles=1)) == 2.0
    finally:
        mod.time.monotonic = orig


def test_multi_bubble_burst_keeps_quiet_window_even_when_merged_text_is_social():
    """Second bubble must re-arm coalescing — never early-flush a merge."""
    adapter = _adapter()
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.4
    try:
        # Even if the merged text would classify as social, bubble_count > 1 wins.
        assert adapter._text_batch_delay_for(_event("hi", bubbles=2)) == 2.0
        assert adapter._text_batch_delay_for(_event("ok", bubbles=3)) == 2.0
    finally:
        mod.time.monotonic = orig


def test_group_chats_unaffected_by_lane_quiet_bypass():
    adapter = _adapter()
    delay = adapter._text_batch_delay_for(_event("ok", chat_type="group", bubbles=1))
    assert delay <= 0.3


def test_fast_lane_disabled_keeps_quiet_window_for_eligible_single():
    adapter = _adapter(_fast_lane_quiet_bypass=False)
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.4
    try:
        assert adapter._text_batch_delay_for(_event("ok", bubbles=1)) == 2.0
    finally:
        mod.time.monotonic = orig


def test_skip_helper_never_raises_on_bad_pending():
    adapter = _adapter()
    assert adapter._should_skip_dm_quiet_window(None) is False
    assert adapter._should_skip_dm_quiet_window(object()) is False  # type: ignore[arg-type]
