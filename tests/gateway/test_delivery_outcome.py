"""Gateway-wide delivery outcome model — stages, evidence gates, channel seams.

Contracts (not snapshots):
* Stages advance only with allowed evidence; weaker evidence → unavailable.
* Local file save is never delivered.
* Adapter call is attempted only; provider accept → delivered, never confirmed.
* Web chat SSE write does not establish delivered/confirmed.
* Privacy: no message body, raw chat ids, or secrets in the log payload.
* Telegram receipt semantics remain (success/failure + digests share the salt).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.delivery_outcome import (
    CHANNEL_LOCAL,
    CHANNEL_SLACK,
    CHANNEL_TELEGRAM,
    CHANNEL_WEB_CHAT,
    EVIDENCE_ADAPTER_CALL,
    EVIDENCE_CLIENT_ACK,
    EVIDENCE_FILTERED,
    EVIDENCE_LOCAL_FILE,
    EVIDENCE_PROVIDER_ACCEPT,
    EVIDENCE_SSE_WRITE,
    EVIDENCE_UNAVAILABLE,
    LOG_PREFIX,
    MISSING_TOKEN,
    STAGE_ATTEMPTED,
    STAGE_CONFIRMED,
    STAGE_DELIVERED,
    STAGE_PREPARED,
    STATUS_FAILED,
    STATUS_FILTERED,
    STATUS_LOCAL_SAVED,
    STATUS_UNAVAILABLE,
    DeliveryOutcome,
    normalize_channel,
    observe_send_result,
    redacted_token,
)
from gateway.platforms.base import SendResult
from gateway.telegram_delivery_receipt import (
    RECEIPT_LOG_PREFIX,
    TelegramDeliveryReceipt,
    redacted_token as telegram_redacted_token,
)

OUTCOME_LOGGER = "gateway.delivery_outcome"
SECRET_BODY = "SECRET-REPLY-BODY hunter2 https://private.example/path"
CHAT_ID = "775566675"
RAW_MID = "987654321"
CONTENT_FRAGMENTS = ("SECRET-REPLY-BODY", "hunter2", "https://private.example/path", CHAT_ID, RAW_MID)


def _outcomes(caplog):
    return [r for r in caplog.records if r.name == OUTCOME_LOGGER]


def _assert_no_leak(records):
    for record in records:
        rendered = record.getMessage()
        payload = getattr(record, "delivery_outcome", {})
        for fragment in CONTENT_FRAGMENTS:
            assert fragment not in rendered
            assert fragment not in str(payload)


# --- model / transitions -------------------------------------------------


def test_normalize_channel_maps_api_server_to_web_chat():
    assert normalize_channel(Platform.API_SERVER) == CHANNEL_WEB_CHAT
    assert normalize_channel("web") == CHANNEL_WEB_CHAT
    assert normalize_channel(Platform.TELEGRAM) == CHANNEL_TELEGRAM
    assert normalize_channel(Platform.SLACK) == CHANNEL_SLACK
    assert normalize_channel(Platform.LOCAL) == CHANNEL_LOCAL


def test_stage_evidence_gates_and_unavailable(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    outcome = DeliveryOutcome(channel="telegram", chat_id=CHAT_ID, clock=lambda: 1.5)
    outcome.prepared()
    assert outcome.stage == STAGE_PREPARED

    # Adapter call → attempted only.
    outcome.attempted(evidence=EVIDENCE_ADAPTER_CALL)
    assert outcome.stage == STAGE_ATTEMPTED

    # SSE write must not establish delivered.
    outcome.delivered(RAW_MID, evidence=EVIDENCE_SSE_WRITE)
    assert outcome.stage == STAGE_ATTEMPTED
    assert outcome.status == STATUS_UNAVAILABLE

    # Provider accept with mid → delivered, not confirmed.
    outcome2 = DeliveryOutcome(channel="slack", chat_id=CHAT_ID, clock=lambda: 2.0)
    outcome2.prepared()
    outcome2.attempted()
    outcome2.delivered(RAW_MID, evidence=EVIDENCE_PROVIDER_ACCEPT)
    assert outcome2.stage == STAGE_DELIVERED
    assert outcome2.status is None

    # Provider evidence cannot confirm.
    outcome2.confirmed(evidence=EVIDENCE_PROVIDER_ACCEPT)
    assert outcome2.stage == STAGE_DELIVERED
    assert outcome2.status == STATUS_UNAVAILABLE

    # Client ack can confirm.
    outcome2.confirmed(evidence=EVIDENCE_CLIENT_ACK)
    assert outcome2.stage == STAGE_CONFIRMED
    assert outcome2.status is None

    _assert_no_leak(_outcomes(caplog))


def test_success_without_message_id_is_unavailable_not_delivered(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    outcome = observe_send_result(
        channel="slack", chat_id=CHAT_ID, result=SendResult(success=True))
    assert outcome.stage == STAGE_ATTEMPTED
    assert outcome.status == STATUS_UNAVAILABLE
    assert outcome.fields()["mid"] == MISSING_TOKEN


def test_failed_and_filtered_are_explicit(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    failed = DeliveryOutcome(channel="telegram", chat_id=CHAT_ID)
    failed.prepared()
    failed.attempted()
    failed.failed()
    assert failed.status == STATUS_FAILED

    filtered = DeliveryOutcome(channel="discord", chat_id=CHAT_ID)
    filtered.prepared()
    filtered.filtered(evidence=EVIDENCE_FILTERED)
    assert filtered.status == STATUS_FILTERED
    assert filtered.stage == STAGE_PREPARED


def test_local_saved_is_never_delivered(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    outcome = DeliveryOutcome(channel="local", chat_id=None)
    outcome.prepared()
    outcome.local_saved()
    assert outcome.status == STATUS_LOCAL_SAVED
    assert outcome.evidence == EVIDENCE_LOCAL_FILE
    assert outcome.stage == STAGE_PREPARED
    assert outcome.stage != STAGE_DELIVERED


def test_payload_allowlist_and_log_prefix(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    outcome = DeliveryOutcome(channel="telegram", chat_id=CHAT_ID, clock=lambda: 9.0)
    outcome.prepared()
    records = _outcomes(caplog)
    assert records
    payload = records[-1].delivery_outcome
    assert set(payload) == {
        "mono", "channel", "chat", "attempt", "mid", "stage", "status", "evidence"}
    assert LOG_PREFIX in records[-1].getMessage()
    _assert_no_leak(records)


def test_digest_shared_with_telegram_receipt():
    assert redacted_token(CHAT_ID, prefix="c") == telegram_redacted_token(CHAT_ID, prefix="c")
    receipt = TelegramDeliveryReceipt(
        chat_id=CHAT_ID, attempt=1, reply_anchor=True, thread_id=None, clock=lambda: 1.0)
    assert "outcome=success" in receipt.render("success", RAW_MID)
    assert RECEIPT_LOG_PREFIX in receipt.render("success", RAW_MID)


# --- delivery router seams (local / telegram / slack / filtered) ---------


@pytest.mark.asyncio
async def test_router_local_save_emits_local_saved_not_delivered(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    router = DeliveryRouter(GatewayConfig())
    router.output_dir = tmp_path / "out"
    results = await router.deliver(SECRET_BODY, [DeliveryTarget(platform=Platform.LOCAL)])
    entry = results["local"]
    assert entry["success"] is True
    assert entry["delivered"] is False
    assert entry["result"]["delivered"] is False
    assert entry["delivery_outcome"]["status"] == STATUS_LOCAL_SAVED
    assert entry["delivery_outcome"]["stage"] == STAGE_PREPARED
    _assert_no_leak(_outcomes(caplog))


@pytest.mark.asyncio
async def test_router_telegram_provider_accept_is_delivered_not_confirmed(
        tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    adapter = MagicMock()
    adapter.splits_long_messages = False
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id=RAW_MID))

    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget(platform=Platform.TELEGRAM, chat_id=CHAT_ID, is_explicit=True)
    results = await router.deliver(SECRET_BODY, [target], metadata={"job_id": "j1"})
    entry = results[target.to_string()]
    assert entry["success"] is True
    do = entry["delivery_outcome"]
    assert do["channel"] == CHANNEL_TELEGRAM
    assert do["stage"] == STAGE_DELIVERED
    assert do["evidence"] == EVIDENCE_PROVIDER_ACCEPT
    assert do["status"] == MISSING_TOKEN  # no confirmed
    assert do["mid"] == redacted_token(RAW_MID, prefix="m")
    assert do["chat"] == redacted_token(CHAT_ID, prefix="c")
    adapter.send.assert_awaited()
    _assert_no_leak(_outcomes(caplog))


@pytest.mark.asyncio
async def test_router_slack_failure_is_failed_not_delivered(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    adapter = MagicMock()
    adapter.splits_long_messages = False
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="boom"))

    router = DeliveryRouter(GatewayConfig(), adapters={Platform.SLACK: adapter})
    target = DeliveryTarget(platform=Platform.SLACK, chat_id="C123", is_explicit=True)
    results = await router.deliver("hi", [target], metadata={"job_id": "j1"})
    entry = results[target.to_string()]
    assert entry["success"] is False
    do = entry["delivery_outcome"]
    assert do["channel"] == CHANNEL_SLACK
    assert do["status"] == STATUS_FAILED
    assert do["stage"] == STAGE_ATTEMPTED


@pytest.mark.asyncio
async def test_router_filtered_silence_not_marked_delivered(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="should-not-send"))
    router = DeliveryRouter(
        GatewayConfig(filter_silence_narration=True),
        adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget(platform=Platform.DISCORD, chat_id="ch1", is_explicit=True)
    results = await router.deliver("*(silent)*", [target])  # no job_id → filtered
    entry = results[target.to_string()]
    assert entry.get("filtered") == "silence_narration"
    assert entry.get("delivered") is False
    assert entry["delivery_outcome"]["status"] == STATUS_FILTERED
    adapter.send.assert_not_awaited()


# --- send_final_ledgered shared seam -------------------------------------


@pytest.mark.asyncio
async def test_send_final_ledgered_emits_outcome_for_slack(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    from gateway.platforms.base import BasePlatformAdapter

    class _Adapter(BasePlatformAdapter):
        @property
        def name(self):
            return "slack"

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="ts.1")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    config = PlatformConfig(enabled=True)
    adapter = _Adapter(config, Platform.SLACK)
    adapter._connected = True
    monkeypatch.setattr(
        adapter, "_record_delivery_obligation", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_final_delivery_adapter", lambda source: adapter)

    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.SLACK, chat_id="C9", thread_id=None),
        text="hi", message_id="m1", ledger_message_id=None)
    result, used = await adapter.send_final_ledgered(
        event, "agent:main:slack:channel:C9", "hello", {}, reply_to=None)
    assert result.success is True
    assert used is adapter
    payloads = [r.delivery_outcome for r in _outcomes(caplog)]
    assert payloads
    assert any(p["stage"] == STAGE_DELIVERED and p["channel"] == CHANNEL_SLACK for p in payloads)
    assert all(
        p.get("stage") != STAGE_CONFIRMED
        or p.get("evidence") == EVIDENCE_CLIENT_ACK
        for p in payloads
    )


@pytest.mark.asyncio
async def test_send_final_ledgered_records_failed_when_send_raises(caplog, monkeypatch):
    """Raised _send_with_retry must emit status=failed and re-raise unchanged."""
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    from gateway.platforms.base import BasePlatformAdapter

    class _Boom(RuntimeError):
        pass

    class _Adapter(BasePlatformAdapter):
        @property
        def name(self):
            return "slack"

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            raise AssertionError("send must not be reached; _send_with_retry is stubbed")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    # Channel-agnostic seam: Slack stands in for any platform adapter.
    adapter = _Adapter(PlatformConfig(enabled=True), Platform.SLACK)
    adapter._connected = True
    monkeypatch.setattr(
        adapter, "_record_delivery_obligation", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_final_delivery_adapter", lambda source: adapter)

    boom = _Boom("network/provider failure")

    async def _raise_send(**kwargs):
        raise boom

    monkeypatch.setattr(adapter, "_send_with_retry", _raise_send)

    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.SLACK, chat_id="C9", thread_id=None),
        text="hi", message_id="m1", ledger_message_id=None)
    with pytest.raises(_Boom) as raised:
        await adapter.send_final_ledgered(
            event, "agent:main:slack:channel:C9", "hello", {}, reply_to=None)
    assert raised.value is boom

    payloads = [r.delivery_outcome for r in _outcomes(caplog)]
    assert any(
        p["stage"] == STAGE_ATTEMPTED
        and p["status"] == STATUS_FAILED
        and p["channel"] == CHANNEL_SLACK
        for p in payloads
    )
    _assert_no_leak(_outcomes(caplog))


def test_record_attempt_failed_never_raises():
    """Safe failure recording must not surface bookkeeping errors to the send path."""
    outcome = DeliveryOutcome(channel="discord", chat_id="ch1")
    outcome.prepared()
    outcome.attempted()
    # Even if emit/logger is broken, record_attempt_failed stays quiet.
    broken = logging.getLogger("gateway.delivery_outcome.broken")
    outcome._logger = broken
    outcome._logger.info = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("log boom"))
    outcome.record_attempt_failed()
    assert outcome.status == STATUS_FAILED


# --- web chat SSE evidence -----------------------------------------------


def test_web_chat_sse_write_does_not_claim_delivered_or_confirmed(caplog):
    caplog.set_level(logging.INFO, logger=OUTCOME_LOGGER)
    outcome = DeliveryOutcome(channel=CHANNEL_WEB_CHAT, chat_id="sess_1")
    outcome.prepared()
    outcome.attempted(evidence=EVIDENCE_SSE_WRITE)
    assert outcome.stage == STAGE_ATTEMPTED
    outcome.unavailable(for_stage=STAGE_DELIVERED, evidence=EVIDENCE_SSE_WRITE)
    outcome.unavailable(for_stage=STAGE_CONFIRMED, evidence=EVIDENCE_SSE_WRITE)
    assert outcome.status == STATUS_UNAVAILABLE
    # Inventing provider_accept from an SSE write must be refused.
    denied = DeliveryOutcome(channel=CHANNEL_WEB_CHAT, chat_id="sess_2")
    denied.attempted(evidence=EVIDENCE_SSE_WRITE)
    denied.delivered("frame-1", evidence=EVIDENCE_SSE_WRITE)
    assert denied.stage == STAGE_ATTEMPTED
    assert denied.status == STATUS_UNAVAILABLE
    _assert_no_leak(_outcomes(caplog))
