"""Behaviour contracts for the redacted Telegram native-reaction receipt.

The reaction receipt exists so a *real* ``setMessageReaction`` attempt is provable from
local logs (LAB-6: reactions actually leave the process on the inbound lifecycle) — and so
that proof is safe to keep in a production log. These tests exercise the seams that make
those claims, not the source text of the module:

* one receipt per attempt, carrying ONLY the allowlisted fields;
* success and failure (with the failure class) both recorded;
* no message text, no raw ids, and no caller-supplied string reaching the log — including
  the emoji slot (the react paths accept arbitrary caller strings);
* the react-only path reacts without ever sending text;
* a receipt error never breaks the reaction path (fail-open).
"""

import logging
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.telegram.adapter as telegram_adapter_module
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from gateway.telegram_reaction_receipt import (
    LOCAL_NO_BOT,
    PHASE_DIRECT,
    RECEIPT_LOG_PREFIX,
    TelegramReactionReceipt,
    failure_class_token,
    redacted_token,
    safe_emoji_token,
)

RECEIPT_LOGGER = "gateway.telegram_reaction_receipt"
ALLOWED_FIELDS = ("mono", "chat", "mid", "emoji", "phase", "outcome", "failure_class")
CHAT_ID = "909987907"
RAW_MESSAGE_ID = 4242
SECRET_PAYLOAD = "SECRET-REPLY-BODY hunter2 https://private.example/path"
CONTENT_FRAGMENTS = ("SECRET-REPLY-BODY", "hunter2", "https://private.example/path")
ACK_EMOJI = "\U0001f440"
OK_EMOJI = "\U0001f44d"


class ReactionRejected(Exception):
    """Stands in for a Bot API rejection (``REACTION_INVALID`` / not permitted)."""


def _make_adapter() -> "telegram_adapter_module.TelegramAdapter":
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(chat_id: str = CHAT_ID, message_id: str = str(RAW_MESSAGE_ID)) -> MessageEvent:
    return MessageEvent(
        text=SECRET_PAYLOAD,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="private",
            user_id="42", user_name="TestUser"),
        message_id=message_id,
    )


def _receipts(caplog):
    return [record for record in caplog.records if record.name == RECEIPT_LOGGER]


def _assert_no_content_leak(records):
    """No receipt record may carry message text, user content, or raw ids."""
    for record in records:
        rendered = record.getMessage()
        payload = getattr(record, "reaction_receipt", {})
        for fragment in CONTENT_FRAGMENTS:
            assert fragment not in rendered
            assert fragment not in repr(payload)
        assert CHAT_ID not in rendered
        assert str(RAW_MESSAGE_ID) not in rendered


@pytest.fixture(autouse=True)
def _reactions_on(monkeypatch):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")


# ── receipt at the reaction seam ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_reaction_emits_success_receipt_with_hashed_ids(caplog):
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        assert await adapter._set_reaction(CHAT_ID, str(RAW_MESSAGE_ID), OK_EMOJI) is True

    records = _receipts(caplog)
    assert len(records) == 1
    line = records[0].getMessage()
    assert line.startswith(RECEIPT_LOG_PREFIX + " ")
    assert "outcome=success" in line
    assert "failure_class=none" in line
    assert "emoji=" + OK_EMOJI in line
    assert "mono=" in line
    payload = records[0].reaction_receipt
    assert tuple(payload) == ALLOWED_FIELDS
    assert payload["outcome"] == "success"
    assert payload["emoji"] == OK_EMOJI
    assert payload["phase"] == PHASE_DIRECT
    assert payload["failure_class"] == "none"
    assert payload["chat"] == redacted_token(CHAT_ID, prefix="c")
    assert payload["mid"] == redacted_token(str(RAW_MESSAGE_ID), prefix="m")
    _assert_no_content_leak(records)


@pytest.mark.asyncio
async def test_failed_reaction_receipt_records_failure_class_without_content(caplog):
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(
        side_effect=ReactionRejected(f"Bad Request: REACTION_INVALID {SECRET_PAYLOAD}"))

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        assert await adapter._set_reaction(CHAT_ID, str(RAW_MESSAGE_ID), OK_EMOJI) is False

    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].reaction_receipt
    assert payload["outcome"] == "failed"
    assert payload["failure_class"] == "ReactionRejected"
    assert payload["emoji"] == OK_EMOJI
    _assert_no_content_leak(records)
    assert "ReactionRejected" in records[0].getMessage()
    # the error *message* never reaches the receipt, only its class name
    assert "Bad Request" not in records[0].getMessage()
    assert "REACTION_INVALID" not in records[0].getMessage()


@pytest.mark.asyncio
async def test_refused_before_platform_call_records_allowlisted_failure_token(caplog):
    """No bot yet: nothing left the process, and the receipt says so with a local token."""
    adapter = _make_adapter()
    adapter._bot = None

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        assert await adapter._set_reaction(CHAT_ID, str(RAW_MESSAGE_ID), OK_EMOJI) is False

    records = _receipts(caplog)
    assert len(records) == 1
    assert records[0].reaction_receipt["failure_class"] == LOCAL_NO_BOT
    assert records[0].reaction_receipt["outcome"] == "failed"


@pytest.mark.asyncio
async def test_bad_target_id_records_failure_class_without_raw_id(caplog):
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        assert await adapter._set_reaction(CHAT_ID, "SECRET-NOT-A-NUMBER", OK_EMOJI) is False

    records = _receipts(caplog)
    assert len(records) == 1
    assert records[0].reaction_receipt["failure_class"] == "ValueError"
    assert "SECRET-NOT-A-NUMBER" not in records[0].getMessage()


# ── lifecycle phases: start / complete / cancelled ──────────────────────────


@pytest.mark.asyncio
async def test_processing_start_receipt_records_start_phase_for_plain_dm(caplog):
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        await adapter.on_processing_start(_make_event())

    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].reaction_receipt
    assert (payload["phase"], payload["emoji"], payload["outcome"]) == ("start", ACK_EMOJI, "success")
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=int(CHAT_ID), message_id=RAW_MESSAGE_ID, reaction=ACK_EMOJI)


@pytest.mark.asyncio
async def test_processing_complete_receipt_records_complete_phase_and_swap(caplog):
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    payload = _receipts(caplog)[0].reaction_receipt
    assert (payload["phase"], payload["emoji"], payload["outcome"]) == ("complete", OK_EMOJI, "success")


@pytest.mark.asyncio
async def test_processing_complete_cancelled_receipt_records_clear_with_none_emoji(caplog):
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        await adapter.on_processing_complete(_make_event(), ProcessingOutcome.CANCELLED)

    records = _receipts(caplog)
    assert len(records) == 1
    payload = records[0].reaction_receipt
    assert payload["phase"] == "cancelled"
    assert payload["emoji"] == "none"
    assert payload["outcome"] == "success"
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=int(CHAT_ID), message_id=RAW_MESSAGE_ID, reaction=None)


@pytest.mark.asyncio
async def test_disabled_reactions_attempt_nothing_and_emit_nothing(caplog, monkeypatch):
    """Gated off is not an attempt: no Bot API call and no receipt."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        await adapter.on_processing_start(_make_event())
        await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    assert _receipts(caplog) == []
    adapter._bot.set_message_reaction.assert_not_awaited()


# ── react-only path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_react_only_path_receipts_without_ever_sending_text(caplog):
    adapter = _make_adapter()
    adapter.send = AsyncMock()

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        added = await adapter.add_reaction(CHAT_ID, OK_EMOJI, message_id=str(RAW_MESSAGE_ID))
        removed = await adapter.remove_reaction(CHAT_ID, message_id=str(RAW_MESSAGE_ID))

    assert added is True
    assert removed is True
    adapter.send.assert_not_awaited()
    adapter.send.assert_not_called()
    payloads = [record.reaction_receipt for record in _receipts(caplog)]
    assert [(p["phase"], p["emoji"], p["outcome"]) for p in payloads] == [
        (PHASE_DIRECT, OK_EMOJI, "success"),
        (PHASE_DIRECT, "none", "success"),
    ]


# ── carrier invariants ──────────────────────────────────────────────────────


def test_receipt_line_shape_is_exact(caplog):
    receipt = TelegramReactionReceipt(
        chat_id="123", message_id="456", emoji=OK_EMOJI, phase="start", clock=lambda: 1.5)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        receipt.succeeded()

    line = _receipts(caplog)[0].getMessage()
    assert line == (
        f"{RECEIPT_LOG_PREFIX} mono=1.5 chat={redacted_token('123', prefix='c')} "
        f"mid={redacted_token('456', prefix='m')} emoji={OK_EMOJI} phase=start "
        "outcome=success failure_class=none"
    )


def test_receipt_emits_once_and_exposes_only_allowlisted_fields(caplog):
    receipt = TelegramReactionReceipt(
        chat_id="123", message_id="456", emoji=None, phase="cancelled", clock=lambda: 5.0)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        receipt.failed(ReactionRejected("boom"))
        receipt.succeeded()

    records = _receipts(caplog)
    assert len(records) == 1
    assert tuple(records[0].reaction_receipt) == ALLOWED_FIELDS
    assert records[0].reaction_receipt == {
        "mono": 5.0,
        "chat": redacted_token("123", prefix="c"),
        "mid": redacted_token("456", prefix="m"),
        "emoji": "none",
        "phase": "cancelled",
        "outcome": "failed",
        "failure_class": "ReactionRejected",
    }


def test_receipt_never_raises_when_the_logger_does(caplog, monkeypatch):
    """Fail-open: a logging failure is swallowed and the emit flag is still consumed."""
    receipt = TelegramReactionReceipt(
        chat_id="123", message_id="456", emoji=OK_EMOJI, phase="start", clock=lambda: 5.0)

    def boom(*_args, **_kwargs):
        raise RuntimeError("logger is broken")

    monkeypatch.setattr(receipt._logger, "info", boom)
    receipt.succeeded()
    assert receipt.emitted is True


@pytest.mark.asyncio
async def test_receipt_construction_failure_never_breaks_the_reaction_path(monkeypatch, caplog):
    """A broken receipt module must not turn a successful reaction into a failure."""
    adapter = _make_adapter()

    class Exploding:
        def __init__(self, **_kwargs):
            raise RuntimeError("receipt module unavailable")

    monkeypatch.setattr(telegram_adapter_module, "TelegramReactionReceipt", Exploding)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        assert await adapter._set_reaction(CHAT_ID, str(RAW_MESSAGE_ID), OK_EMOJI) is True
        assert await adapter._set_reaction(CHAT_ID, "not-a-number", OK_EMOJI) is False

    assert _receipts(caplog) == []


# ── field sanitizers (the no-leak contract) ─────────────────────────────────


def test_safe_emoji_token_echoes_glyphs_and_neutralizes_anything_that_could_be_text():
    assert safe_emoji_token(None) == "none"
    assert safe_emoji_token("") == "none"
    assert safe_emoji_token("   ") == "none"
    assert safe_emoji_token(ACK_EMOJI) == ACK_EMOJI
    assert safe_emoji_token(OK_EMOJI) == OK_EMOJI
    # a text payload in the emoji slot can never reach the log
    assert safe_emoji_token(SECRET_PAYLOAD) == "other"
    assert safe_emoji_token("🔥" * 9) == "other"
    assert "SECRET-REPLY-BODY" not in safe_emoji_token(SECRET_PAYLOAD)

    class Exploding:
        def __str__(self):
            raise RuntimeError("nope")

    assert safe_emoji_token(Exploding()) == "other"


def test_failure_class_token_accepts_class_names_and_allowlisted_tokens_only():
    assert failure_class_token(None) == "unknown"
    assert failure_class_token(ReactionRejected("x")) == "ReactionRejected"
    assert failure_class_token(LOCAL_NO_BOT) == LOCAL_NO_BOT
    # an arbitrary caller string is NOT a failure class
    assert failure_class_token(SECRET_PAYLOAD) == "unknown"
    assert failure_class_token("RuntimeError") == "unknown"


def test_receipt_phase_outside_the_allowlist_degrades_to_other(caplog):
    receipt = TelegramReactionReceipt(
        chat_id="123", message_id="456", emoji=OK_EMOJI, phase=SECRET_PAYLOAD, clock=lambda: 1.0)

    with caplog.at_level(logging.INFO, logger=RECEIPT_LOGGER):
        receipt.succeeded()

    line = _receipts(caplog)[0].getMessage()
    assert "phase=other" in line
    assert "SECRET-REPLY-BODY" not in line


def test_receipt_payload_matches_the_delivery_receipt_digest_convention():
    """One digest convention: the same chat hashes identically in both receipt streams."""
    from gateway.telegram_delivery_receipt import redacted_token as delivery_token

    assert redacted_token(CHAT_ID, prefix="c") == delivery_token(CHAT_ID, prefix="c")
