"""The seam: the agent-side ``react_to_message`` tool → the REAL Telegram adapter.

``tests/tools/test_react_to_message_messaging.py`` proves the tool's messaging surface against a
duck-typed fake adapter; ``tests/gateway/test_telegram_reactions.py`` proves the adapter's own
reaction API. Neither proves the two halves are wired to each other. This file does.

Every test builds a REAL ``TelegramAdapter`` (only its Bot API is a recording async fake), resolves
it through the gateway runner that ``tools/send_message_senders.py::_live_adapter`` actually walks,
dispatches the tool through the registry the way the model's call arrives, and asserts the Bot API
call the user would see on their own bubble. The second half is the same real adapter under the live
config shape ``platforms.telegram.extra.{reactions, reaction_style}``: ``reaction_style: content``
must leave the automatic lifecycle receipts silent while the agent's deliberate reaction still lands.
"""

import contextlib
import json
from unittest.mock import AsyncMock

import pytest

# Importing the module registers the ``react_to_message`` tool with the registry.
import tools.react_to_message_tool as reactions  # noqa: F401
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from tools.registry import registry

CHAT_ID = "123"
MESSAGE_ID = "555"
HEART = "\u2764\ufe0f"
EYES = "\U0001f440"
THUMB = "\U0001f44d"


# ── real adapter / real runner construction ──────────────────────────


def _real_adapter(**extra):
    """A REAL ``TelegramAdapter`` whose only fake is the Bot API.

    Same construction style as ``tests/gateway/test_telegram_reactions.py::_make_adapter``: the
    class under test is the production one, so a passing assertion means the production
    ``add_reaction``/``remove_reaction``/``on_processing_*`` bodies ran — not a stand-in's.
    """
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter.config.extra.update(extra)
    return adapter


def _live_config_extra(monkeypatch, **extra):
    """The documented live config shape ``platforms.telegram.extra.{reactions, reaction_style}``
    through the adapter's own YAML hook (never the env bridge), so these tests run the same
    resolution the live gateway runs. Both env vars are cleared before the hook and again after it,
    so neither a developer's exported value nor a bridged value can decide the outcome here."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    resolved = _apply_yaml_config({}, {"extra": extra}) or {}
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    return resolved


def _bind_gateway_runner(monkeypatch, adapter):
    """Point ``gateway.run._gateway_runner_ref`` at a real ``GatewayRunner`` whose adapter map holds
    the real *adapter* — the resolver ``_live_adapter`` reads in production.

    ``_live_adapter`` walks the real chain (``runner._authorization_adapter`` →
    ``_adapters_for_profile`` → the primary adapter map); only the active-profile lookup is pinned to
    ``default``, because the per-test HERMES_HOME sandbox is neither the default home nor
    ``profiles/<name>`` and the profile name is not the seam under test.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)  # bare instance, the way the gateway tests build runners
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    return runner


@contextlib.contextmanager
def _telegram_turn(monkeypatch, adapter, *, message_id=MESSAGE_ID):
    """A gateway Telegram turn as the gateway binds it (``gateway/session_context.py``), with the
    real adapter behind it. The session-store mirror is not this seam (the messaging tests cover
    it), so the DB is kept out of the way and only the platform call is under assertion."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _bind_gateway_runner(monkeypatch, adapter)
    monkeypatch.setattr(reactions, "_open_session_db", lambda: None)
    tokens = set_session_vars(
        platform="telegram", source="telegram", chat_id=CHAT_ID, session_key="telegram:123",
        session_id="telegram:123", message_id=message_id,
    )
    try:
        yield adapter
    finally:
        clear_session_vars(tokens)


def _dispatch(args):
    """The model's tool call, through the registry."""
    raw = registry.dispatch("react_to_message", args)
    return json.loads(raw if isinstance(raw, str) else json.dumps(raw))


def _event(chat_id=CHAT_ID, message_id=MESSAGE_ID):
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="private",
            user_id="42", user_name="TestUser",
        ),
        message_id=message_id,
    )


# ── the agent's reaction, through the real adapter ───────────────────


class TestTheToolReachesTheRealAdapter:
    def test_heart_lands_on_the_triggering_message(self, monkeypatch):
        """``react_to_message`` → registry → tool → ``_live_adapter`` → real TelegramAdapter →
        ``bot.set_message_reaction``, naming this turn's chat and triggering message."""
        adapter = _real_adapter()

        with _telegram_turn(monkeypatch, adapter):
            result = _dispatch({"emoji": HEART})

        adapter._bot.set_message_reaction.assert_awaited_once_with(
            chat_id=123, message_id=int(MESSAGE_ID), reaction=HEART)
        assert result == {"success": True, "platform": "telegram",
                          "message_id": MESSAGE_ID, "emoji": HEART}

    def test_empty_emoji_retracts_through_the_real_adapter(self, monkeypatch):
        """An empty emoji is the retract verb, and the real wrapper clears via ``reaction=None``."""
        adapter = _real_adapter()

        with _telegram_turn(monkeypatch, adapter):
            result = _dispatch({"emoji": ""})

        adapter._bot.set_message_reaction.assert_awaited_once_with(
            chat_id=123, message_id=int(MESSAGE_ID), reaction=None)
        assert result["success"] is True

    def test_the_real_adapter_earns_the_tool_on_this_turn(self, monkeypatch):
        """The surface gate the toolset fold reads answers True for a real adapter (it would say
        False — and the tool would never be handed to the turn — for one without the API)."""
        adapter = _real_adapter()

        with _telegram_turn(monkeypatch, adapter):
            assert reactions.check_react_requirements() is True

    def test_a_turn_with_no_triggering_id_refuses_instead_of_guessing(self, monkeypatch):
        """No triggering id (Telegram keeps no per-chat "latest inbound"): the real adapter is
        never handed a guessed message — the call fails closed before the Bot API."""
        adapter = _real_adapter()

        with _telegram_turn(monkeypatch, adapter, message_id=""):
            result = _dispatch({"emoji": HEART})

        assert result.get("success") is not True
        assert "no id on the platform" in json.dumps(result)
        adapter._bot.set_message_reaction.assert_not_awaited()


# ── the automatic lifecycle receipts under the live reaction policy ──


class TestLifecycleReactionsUnderTheLivePolicy:
    @pytest.mark.asyncio
    async def test_content_style_keeps_the_lifecycle_silent(self, monkeypatch):
        """``reaction_style: content`` (the user's live config): start + SUCCESS end produce ZERO
        Bot API calls — the automatic 👀/👍 receipts are off, not merely reordered."""
        adapter = _real_adapter(**_live_config_extra(
            monkeypatch, **{"reactions": True, "reaction_style": "content"}))
        event = _event()

        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        adapter._bot.set_message_reaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_content_style_is_silent_for_every_outcome(self, monkeypatch):
        """The silence holds for FAILURE and CANCELLED too (no 👎, no clear-only call)."""
        adapter = _real_adapter(**_live_config_extra(
            monkeypatch, **{"reactions": True, "reaction_style": "content"}))
        event = _event()

        for outcome in (ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED):
            await adapter.on_processing_start(event)
            await adapter.on_processing_complete(event, outcome)

        adapter._bot.set_message_reaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_receipt_style_still_posts_eyes_then_thumb(self, monkeypatch):
        """Control: the same real adapter under ``receipt`` keeps the pre-existing behaviour, so
        the silence above is the style's doing and not an inert adapter."""
        adapter = _real_adapter(**_live_config_extra(
            monkeypatch, **{"reactions": True, "reaction_style": "receipt"}))
        event = _event()

        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

        assert [c.kwargs["reaction"] for c in adapter._bot.set_message_reaction.await_args_list] == [EYES, THUMB]
        assert [c.kwargs["message_id"] for c in adapter._bot.set_message_reaction.await_args_list] == [int(MESSAGE_ID)] * 2

    def test_content_style_still_lets_the_agent_react_deliberately(self, monkeypatch):
        """The whole point of ``content``: the SAME real adapter, on the SAME live config, makes
        zero automatic calls and still carries the agent's own reaction through the tool."""
        adapter = _real_adapter(**_live_config_extra(
            monkeypatch, **{"reactions": True, "reaction_style": "content"}))

        with _telegram_turn(monkeypatch, adapter):
            result = _dispatch({"emoji": HEART})

        adapter._bot.set_message_reaction.assert_awaited_once_with(
            chat_id=123, message_id=int(MESSAGE_ID), reaction=HEART)
        assert result["success"] is True
