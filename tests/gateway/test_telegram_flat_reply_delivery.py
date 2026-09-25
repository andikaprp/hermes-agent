"""Telegram ordinary conversations send flat unless the user selected an older reply."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_event, merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _event(text: str, message_id: str, *, reply_to_message_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"),
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
    )


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=999))
    return adapter


@pytest.mark.asyncio
async def test_coalesced_telegram_bubbles_form_one_flat_outbound_turn():
    pending = {}
    for text, message_id in (("how", "101"), ("r", "102"), ("u", "103"), ("sayang", "104")):
        merge_pending_message_event(pending, "telegram:123", _event(text, message_id), merge_text=True)

    turn = pending["telegram:123"]
    assert turn.text == "how\nr\nu\nsayang"
    assert _reply_anchor_for_event(turn) is None
    assert _thread_metadata_for_event(turn) is None

    adapter = _adapter()
    await adapter.send("123", "combined answer", reply_to=_reply_anchor_for_event(turn), metadata=_thread_metadata_for_event(turn))

    assert adapter._bot.send_message.await_count == 1
    assert adapter._bot.send_message.call_args.kwargs["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_normal_telegram_message_sends_flat():
    event = _event("good", "105")
    adapter = _adapter()

    await adapter.send("123", "good", reply_to=_reply_anchor_for_event(event), metadata=_thread_metadata_for_event(event))

    assert adapter._bot.send_message.call_args.kwargs["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_explicit_reply_to_older_telegram_message_preserves_anchor():
    event = _event("what about this?", "105", reply_to_message_id="42")
    adapter = _adapter()

    await adapter.send("123", "answer", reply_to=_reply_anchor_for_event(event), metadata=_thread_metadata_for_event(event))

    assert _reply_anchor_for_event(event) == "42"
    assert adapter._bot.send_message.call_args.kwargs["reply_to_message_id"] == 42


def test_queued_follow_up_ordinary_turn_has_no_reply_anchor():
    pending = {}
    merge_pending_message_event(pending, "telegram:123", _event("and then?", "201"), merge_text=True)
    follow_up = pending["telegram:123"]
    assert follow_up.message_id == "201"
    assert _reply_anchor_for_event(follow_up) is None


def test_queued_follow_up_explicit_older_reply_keeps_anchor():
    pending = {}
    merge_pending_message_event(pending, "telegram:123", _event("first", "201"), merge_text=True)
    merge_pending_message_event(
        pending, "telegram:123", _event("about that", "202", reply_to_message_id="42"), merge_text=True,
    )
    follow_up = pending["telegram:123"]
    assert follow_up.text == "first\nabout that"
    assert follow_up.message_id == "202"
    assert _reply_anchor_for_event(follow_up) == "42"


def test_coalesced_ordinary_bubbles_do_not_promote_latest_id_to_reply_anchor():
    pending = {}
    merge_pending_message_event(pending, "telegram:123", _event("hi", "10"), merge_text=True)
    merge_pending_message_event(pending, "telegram:123", _event("again", "11"), merge_text=True)
    turn = pending["telegram:123"]
    assert turn.message_id == "11"
    assert turn.reply_to_message_id is None
    assert _reply_anchor_for_event(turn) is None
