"""Regression coverage for the redacted Telegram outbound delivery receipt.

The receipt exists to make reply-anchor (``reply_to_message_id``) and thread-id
presence provable from local logs for *real* sends — and to be safe to keep in a
production log: it must never carry message text, user content, tokens, or raw
chat/message ids.

``FakeTelegram*`` mirrors python-telegram-bot's hierarchy
(``BadRequest → NetworkError → TelegramError → Exception``) so error
classification is deterministic and does not depend on the installed SDK.
"""

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.telegram_delivery_receipt import (
    DIGEST_CHARS,
    RECEIPT_LOG_PREFIX,
    TelegramDeliveryReceipt,
    redacted_token,
)

RECEIPT_LOGGER = "gateway.telegram_delivery_receipt"
ALLOWED_FIELDS = ("mono", "chat", "attempt", "mid", "anchor", "thread", "outcome")
CHAT_ID = "775566675"
RAW_MESSAGE_ID = 987654321
SECRET_BODY = "SECRET-REPLY-BODY hunter2 https://private.example/path"
CONTENT_FRAGMENTS = ("SECRET-REPLY-BODY", "hunter2", "https://private.example/path")


class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


def _install_fake_telegram(monkeypatch):
    telegram = types.ModuleType("telegram")
    telegram.Update = object
    telegram.Bot = object
    telegram.Message = object
    error = types.ModuleType("telegram.error")
    error.NetworkError = FakeNetworkError
    error.BadRequest = FakeBadRequest
    error.TimedOut = FakeTimedOut
    telegram.error = error
    constants = types.ModuleType("telegram.constants")
    constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    telegram.constants = constants
    for name, module in (("telegram", telegram), ("telegram.error", error), ("telegram.constants", constants)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def _fake_telegram(monkeypatch):
    _install_fake_telegram(monkeypatch)


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    adapter._fallback_ips = []
    # Legacy sendMessage path by default; the rich-seam test opts back in.
    adapter._rich_messages_enabled = False
    return adapter


def _receipts(caplog):
    return [record for record in caplog.records if record.name == RECEIPT_LOGGER]


def _assert_no_content_leak(records):
    """No receipt record may carry message text, user content, or raw ids."""
    for record in records:
        rendered = record.getMessage()
        payload = getattr(record, "delivery_receipt", {})
        for fragment in CONTENT_FRAGMENTS:
            assert fragment not in rendered
            assert fragment not in repr(payload)
        assert CHAT_ID not in rendered
        assert str(RAW_MESSAGE_ID) not in rendered


# ── receipt at the real outbound seam ───────────────────────────────────────


@pytest.mark.asyncio
async def test_send_receipt_reports_absent_anchor_and_thread_without_content(caplog):
    adapter = _make_adapter()
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=RAW_MESSAGE_ID)))

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(chat_id=CHAT_ID, content=SECRET_BODY)

    assert result.success is True
    records = _receipts(caplog)
    assert len(records) == 1
    line = records[0].getMessage()
    assert line.startswith(RECEIPT_LOG_PREFIX + " ")
    assert "anchor=absent" in line
    assert "thread=absent" in line
    assert "outcome=success" in line
    assert "attempt=1" in line
    assert "mono=" in line
    assert "chat=c" in line
    assert "mid=m" in line
    _assert_no_content_leak(records)
    payload = records[0].delivery_receipt
    assert tuple(payload) == ALLOWED_FIELDS
    assert payload["anchor"] == "absent"
    assert payload["thread"] == "absent"
    assert payload["outcome"] == "success"
    assert payload["chat"] == redacted_token(CHAT_ID, prefix="c")
    assert payload["mid"] == redacted_token(RAW_MESSAGE_ID, prefix="m")


@pytest.mark.asyncio
async def test_send_receipt_reports_present_anchor_and_thread(caplog):
    adapter = _make_adapter()
    calls = []

    async def mock_send_message(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(message_id=5551212)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(
            chat_id=CHAT_ID, content=SECRET_BODY, reply_to="462", metadata={"thread_id": "270453"})

    assert result.success is True
    assert calls[0]["reply_to_message_id"] == 462
    assert calls[0]["message_thread_id"] == 270453
    records = _receipts(caplog)
    assert len(records) == 1
    line = records[0].getMessage()
    assert "anchor=present" in line
    assert "thread=present" in line
    assert "outcome=success" in line
    _assert_no_content_leak(records)


@pytest.mark.asyncio
async def test_failed_send_receipt_records_failure_without_content(caplog):
    adapter = _make_adapter()

    async def reject(**_kwargs):
        raise FakeBadRequest("Bad Request: chat not found")

    adapter._bot = SimpleNamespace(send_message=reject)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(chat_id=CHAT_ID, content=SECRET_BODY)

    assert result.success is False
    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].delivery_receipt
    assert payload["outcome"] == "failure"
    assert payload["mid"] == "none"
    assert payload["attempt"] == 1
    assert payload["anchor"] == "absent"
    _assert_no_content_leak(records)


@pytest.mark.asyncio
async def test_retry_emits_one_receipt_per_platform_attempt(caplog):
    """A stale-thread send retried without the thread records both attempts."""
    adapter = _make_adapter()
    calls = []

    async def stale_thread(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("message_thread_id") is not None:
            raise FakeBadRequest("Bad Request: message thread not found")
        return SimpleNamespace(message_id=777888)

    adapter._bot = SimpleNamespace(send_message=stale_thread)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(chat_id="-100123", content=SECRET_BODY, metadata={"thread_id": "99999"})

    assert result.success is True
    payloads = [record.delivery_receipt for record in _receipts(caplog)]
    assert [(p["attempt"], p["outcome"], p["thread"]) for p in payloads] == [
        (1, "failure", "present"),
        (2, "failure", "present"),
        (3, "success", "absent"),
    ]
    _assert_no_content_leak(_receipts(caplog))


@pytest.mark.asyncio
async def test_dm_topic_refusal_emits_receipt_for_local_refusal(caplog):
    """A send refused before any Bot API call is recorded as attempt=0 failure."""
    adapter = _make_adapter()
    calls = []

    async def mock_send_message(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(message_id=1)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(
            chat_id=CHAT_ID, content=SECRET_BODY,
            metadata={"thread_id": "270453", "telegram_dm_topic_reply_fallback": True})

    assert result.success is False
    assert calls == []
    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].delivery_receipt
    assert payload["attempt"] == 0
    assert payload["outcome"] == "failure"
    assert payload["anchor"] == "absent"
    assert payload["thread"] == "present"
    _assert_no_content_leak(records)


@pytest.mark.asyncio
async def test_rich_send_seam_emits_receipt_with_anchor_presence(caplog):
    """The Bot API 10.1 sendRichMessage seam is receipted too (it owns anchor routing)."""
    adapter = _make_adapter()
    adapter._rich_messages_enabled = True
    do_api_request = AsyncMock(return_value={"message_id": 424242})
    adapter._bot = SimpleNamespace(do_api_request=do_api_request)
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        result = await adapter.send(
            chat_id=CHAT_ID, content=table, reply_to="462", metadata={"thread_id": "270453"})

    assert result.success is True
    assert do_api_request.await_count == 1
    assert do_api_request.await_args.args[0] == "sendRichMessage"
    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].delivery_receipt
    assert payload["outcome"] == "success"
    assert payload["anchor"] == "present"
    assert payload["thread"] == "present"
    assert payload["mid"] == redacted_token(424242, prefix="m")
    assert "| a | b |" not in records[0].getMessage()


# ── carrier invariants ─────────────────────────────────────────────────────


def test_receipt_line_shape_is_exact(caplog):
    receipt = TelegramDeliveryReceipt(
        chat_id="123", attempt=1, reply_anchor=None, thread_id=None, clock=lambda: 1.5)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        receipt.failed()

    line = _receipts(caplog)[0].getMessage()
    assert line == (
        f"{RECEIPT_LOG_PREFIX} mono=1.5 chat={redacted_token('123', prefix='c')} "
        "attempt=1 mid=none anchor=absent thread=absent outcome=failure"
    )


def test_receipt_emits_once_and_exposes_only_allowlisted_fields(caplog):
    receipt = TelegramDeliveryReceipt(
        chat_id="123", attempt=2, reply_anchor=True, thread_id="77", clock=lambda: 5.0)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        receipt.succeeded("999")
        receipt.failed()

    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].delivery_receipt
    assert tuple(payload) == ALLOWED_FIELDS
    assert payload == {
        "mono": 5.0,
        "chat": redacted_token("123", prefix="c"),
        "attempt": 2,
        "mid": redacted_token("999", prefix="m"),
        "anchor": "present",
        "thread": "present",
        "outcome": "success",
    }


def test_redacted_token_is_stable_short_and_never_the_raw_id():
    first = redacted_token(CHAT_ID, prefix="c")
    assert first == redacted_token(CHAT_ID, prefix="c")
    assert first != redacted_token("775566676", prefix="c")
    assert first.startswith("c")
    assert len(first) == 1 + DIGEST_CHARS
    assert CHAT_ID not in first
    assert redacted_token(None, prefix="m") == "none"
    assert redacted_token("", prefix="m") == "none"
