"""Sub-1s visible signal for every Telegram message type.

The quiet window used to sit in front of the turn (median ~2.7s bubble→flush).
The first thing a user can see must be Telegram's native typing indicator,
armed at inbound — before that wait, before media download, and not gated by
``reaction_style``. The turn itself opens inside 1s: conversational DM quiet
defaults to 300ms (config escape: ``extra.text_batch_quiet_seconds``), and the
ack/social bypass still flushes immediately.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter(**extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="test-token", extra=dict(extra))
    adapter._bot = AsyncMock()
    adapter._running = True
    adapter._drop_delayed_deliveries = False
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._held_inbound_events = []
    adapter._held_inbound_redispatch_task = None
    adapter.HELD_INBOUND_MAX = 64
    adapter._text_batch_delay_seconds = 0.3
    adapter._text_batch_split_delay_seconds = 1.0
    adapter._text_batch_quiet_seconds = 2.0  # pin the old window so typing must not wait for it
    adapter._text_batch_max_wait_seconds = 5.0
    adapter._conversational_dm_batching = True
    adapter._fast_lane_quiet_bypass = False
    adapter._SPLIT_THRESHOLD = 4000
    adapter._TEXT_BATCH_FAST_LEN = 320
    adapter._TEXT_BATCH_FAST_DELAY_S = 0.18
    adapter._TEXT_BATCH_SHORT_LEN = 1024
    adapter._TEXT_BATCH_SHORT_DELAY_S = 0.24
    adapter.handle_message = AsyncMock()
    adapter.send_typing = AsyncMock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._is_user_authorized_from_message = lambda msg: True
    adapter._should_process_message = lambda msg, is_command=False: True
    adapter._gate_or_observe = lambda *args, **kwargs: True
    adapter._build_triggered_event = AsyncMock(side_effect=lambda msg, update, msg_type: _event(
        getattr(msg, "text", "") or "", chat_type=getattr(getattr(msg, "chat", None), "type", "private")))
    adapter._build_message_event = lambda msg, msg_type, update_id=None: _event(
        getattr(msg, "text", "") or getattr(msg, "caption", "") or "", msg_type=msg_type,
        chat_type=getattr(getattr(msg, "chat", None), "type", "private"))
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._max_doc_bytes = 20 * 1024 * 1024
    return adapter


def _event(text="please merge the PR", chat_type="dm", msg_type=MessageType.TEXT, chat_id="42"):
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type),
    )


def _update(msg):
    return SimpleNamespace(update_id=7, message=msg, effective_message=msg, channel_post=None)


def _message(text="please merge the PR", *, chat_type="private", chat_id=42, **media):
    msg = SimpleNamespace(
        message_id=9,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        venue=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )
    for key, value in media.items():
        setattr(msg, key, value)
    return msg


async def _yield_typing():
    """One loop turn: the inbound typing task must have run; the quiet window must not."""
    await asyncio.sleep(0)


# ── quiet-window collapse ──────────────────────────────────────────────


def test_default_quiet_window_is_300ms_for_task_dms():
    """A task DM must not inherit the old 2s pre-turn wait when the key is omitted."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    assert adapter._text_batch_quiet_seconds == 0.3
    pending = _event("please merge the PR and explain the risk")
    pending._batch_opened_mono = 100.0
    pending._batch_bubble_count = 1
    pending._last_chunk_len = len(pending.text)
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.0
    try:
        assert adapter._text_batch_delay_for(pending) == 0.3
    finally:
        mod.time.monotonic = orig


def test_quiet_window_config_escape_restores_a_longer_wait():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t",
        extra={"conversational_dm_batching": True, "text_batch_quiet_seconds": 2},
    ))
    assert adapter._text_batch_quiet_seconds == 2.0
    pending = _event("please merge the PR")
    pending._batch_opened_mono = 100.0
    pending._batch_bubble_count = 1
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.0
    try:
        assert adapter._text_batch_delay_for(pending) == 2.0
    finally:
        mod.time.monotonic = orig


def test_ack_and_social_bypass_still_opens_immediately():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    adapter._fast_lane_quiet_bypass = True
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.0
    try:
        for text in ("ok", "hi", "ya"):
            pending = _event(text)
            pending._batch_opened_mono = 100.0
            pending._batch_bubble_count = 1
            assert adapter._text_batch_delay_for(pending) == 0.0
    finally:
        mod.time.monotonic = orig


def test_multi_bubble_still_rearms_the_quiet_window():
    """A second bubble must not early-flush. The window is the (now short) quiet period."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    adapter._fast_lane_quiet_bypass = True
    pending = _event("ok")
    pending._batch_opened_mono = 100.0
    pending._batch_bubble_count = 2
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.4
    try:
        assert adapter._text_batch_delay_for(pending) == adapter._text_batch_quiet_seconds
        assert adapter._text_batch_delay_for(pending) > 0
    finally:
        mod.time.monotonic = orig


def test_client_split_keeps_the_split_delay_under_the_short_quiet_window():
    """A near-limit chunk is a client split, not a pause. 300ms would truncate the paste."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    adapter._fast_lane_quiet_bypass = False
    pending = _event("x" * adapter._SPLIT_THRESHOLD)
    pending._batch_opened_mono = 100.0
    pending._batch_bubble_count = 1
    pending._last_chunk_len = adapter._SPLIT_THRESHOLD
    import plugins.platforms.telegram.adapter as mod

    orig = mod.time.monotonic
    mod.time.monotonic = lambda: 100.0
    try:
        delay = adapter._text_batch_delay_for(pending)
    finally:
        mod.time.monotonic = orig
    assert delay == adapter._text_batch_split_delay_seconds
    assert delay >= 1.0


def test_group_text_is_not_pulled_into_the_dm_quiet_window():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    pending = _event("ok", chat_type="group")
    pending._batch_bubble_count = 1
    assert adapter._text_batch_delay_for(pending) <= 0.3


@pytest.mark.asyncio
async def test_task_turn_opens_within_one_second_of_enqueue():
    """Real clock, no Telegram traffic: task DM flush must not pay the old 2s window."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="t", extra={"conversational_dm_batching": True},
    ))
    adapter._fast_lane_quiet_bypass = False
    opened = asyncio.Event()

    async def _opened(event):
        opened.set()

    adapter.handle_message = _opened
    event = _event("please merge the PR and explain the risk")
    t0 = time.monotonic()
    adapter._enqueue_text_event(event)
    await asyncio.wait_for(opened.wait(), timeout=0.9)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert opened.is_set()


# ── instant typing indicator ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_dm_types_before_the_quiet_window():
    adapter = _adapter()
    t0 = time.monotonic()
    await adapter._handle_text_message(_update(_message("please merge the PR")), SimpleNamespace())
    await _yield_typing()
    elapsed = time.monotonic() - t0
    adapter.send_typing.assert_awaited()
    adapter.handle_message.assert_not_called()
    assert elapsed < 0.1
    chat_id = adapter.send_typing.await_args.args[0]
    assert str(chat_id) == "42"


@pytest.mark.asyncio
async def test_text_group_types_before_dispatch():
    adapter = _adapter()
    t0 = time.monotonic()
    await adapter._handle_text_message(
        _update(_message("please look at this", chat_type="group", chat_id=-100)), SimpleNamespace())
    await _yield_typing()
    assert time.monotonic() - t0 < 0.1
    adapter.send_typing.assert_awaited()
    assert str(adapter.send_typing.await_args.args[0]) == "-100"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_command_types_before_handle_message():
    adapter = _adapter()
    seen = {}

    async def _handle(event):
        await asyncio.sleep(0)
        seen["typed"] = adapter.send_typing.await_count

    adapter.handle_message = _handle
    t0 = time.monotonic()
    await adapter._handle_command(_update(_message("/status")), SimpleNamespace())
    assert time.monotonic() - t0 < 0.1
    assert seen.get("typed", 0) >= 1


@pytest.mark.asyncio
async def test_location_and_venue_type_before_handle_message():
    adapter = _adapter()
    seen = {}

    async def _handle(event):
        await asyncio.sleep(0)
        seen["typed"] = adapter.send_typing.await_count

    adapter.handle_message = _handle
    pin = SimpleNamespace(latitude=1.2, longitude=3.4)
    for label, media in (
        ("location", {"location": pin}),
        ("venue", {"venue": SimpleNamespace(location=pin, title="Cafe", address="1 St")}),
    ):
        adapter.send_typing.reset_mock()
        seen.clear()
        t0 = time.monotonic()
        await adapter._handle_location_message(_update(_message("", **media)), SimpleNamespace())
        assert time.monotonic() - t0 < 0.1, label
        assert seen.get("typed", 0) >= 1, label


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "voice", "audio", "video", "sticker", "document"])
async def test_media_types_type_before_download(kind):
    adapter = _adapter()
    started = asyncio.Event()

    async def _hang():
        started.set()
        await asyncio.Event().wait()

    file_obj = SimpleNamespace(
        get_file=_hang, file_size=100, file_name="a.bin", mime_type="application/octet-stream",
        file_path="a.jpg", emoji="😀", set_name="", is_animated=False, is_video=False,
        file_unique_id=f"uniq-{kind}",
    )
    media = {
        "photo": {"photo": [file_obj]},
        "voice": {"voice": file_obj},
        "audio": {"audio": file_obj},
        "video": {"video": file_obj},
        "sticker": {"sticker": file_obj},
        "document": {"document": file_obj},
    }[kind]
    t0 = time.monotonic()
    task = asyncio.create_task(adapter._handle_media_message(_update(_message("", **media)), SimpleNamespace()))
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        await _yield_typing()
        assert time.monotonic() - t0 < 0.1
        adapter.send_typing.assert_awaited()
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reaction_style_content_does_not_disable_typing():
    adapter = _adapter(reaction_style="content", reactions=True)
    await adapter._handle_text_message(_update(_message("please merge the PR")), SimpleNamespace())
    await _yield_typing()
    adapter.send_typing.assert_awaited()


@pytest.mark.asyncio
async def test_extra_typing_indicator_false_suppresses_inbound_typing():
    adapter = _adapter(typing_indicator=False)
    await adapter._handle_text_message(_update(_message("please merge the PR")), SimpleNamespace())
    await _yield_typing()
    adapter.send_typing.assert_not_awaited()


@pytest.mark.asyncio
async def test_typing_failure_is_swallowed_and_does_not_block_the_turn():
    adapter = _adapter()

    async def _boom(*args, **kwargs):
        raise RuntimeError("telegram down")

    adapter.send_typing = _boom
    await adapter._handle_text_message(_update(_message("please merge the PR")), SimpleNamespace())
    await _yield_typing()
    # Flush task is still the quiet window, not an exception from typing.
    assert adapter._pending_text_batch_tasks


@pytest.mark.asyncio
async def test_one_typing_action_per_accepted_bubble():
    adapter = _adapter()
    await adapter._handle_text_message(_update(_message("first bubble")), SimpleNamespace())
    await adapter._handle_text_message(_update(_message("second bubble")), SimpleNamespace())
    await _yield_typing()
    assert adapter.send_typing.await_count == 2


@pytest.mark.asyncio
async def test_unauthorized_text_does_not_type():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda msg: False
    adapter._log_blocked_user = lambda *args, **kwargs: None
    await adapter._handle_text_message(_update(_message("secret")), SimpleNamespace())
    await _yield_typing()
    adapter.send_typing.assert_not_awaited()
