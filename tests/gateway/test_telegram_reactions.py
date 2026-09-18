"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(chat_id: str = "123", message_id: str = "456") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="private",
            user_id="42",
            user_name="TestUser",
        ),
        message_id=message_id,
    )


# ── _reactions_enabled ───────────────────────────────────────────────


def test_reactions_disabled_by_default(monkeypatch):
    """Telegram reactions should be disabled by default."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_enabled_when_set_true(monkeypatch):
    """Setting TELEGRAM_REACTIONS=true enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


def test_explicit_env_wins_over_materialized_yaml_default(monkeypatch):
    """TELEGRAM_REACTIONS=true must beat the stock ``reactions: false`` in config.yaml (#109032).

    Fresh installs materialize the whole default config tree, so ``_apply_yaml_config`` seeds
    ``extra["reactions"] = False`` even when the user never chose a value; the reader must still
    honour the explicitly set env var, like ``yaml_env_setter`` documents for the bridge.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False
    assert adapter._reactions_enabled() is True


def test_scoped_miss_does_not_leak_default_profile_env(monkeypatch):
    """Under multiplex a scoped miss must not read another profile's process-env value (#72348)."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")  # default profile's bridged value
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False  # this profile's own YAML
    set_multiplex_active(True)
    token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "222:b2"})
    try:
        assert adapter._reactions_enabled() is False
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


@pytest.mark.asyncio
async def test_set_reaction_calls_bot_api(monkeypatch):
    """_set_reaction should call bot.set_message_reaction with correct args."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._set_reaction("123", "456", "\U0001f440")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    """Should handle events without chat_id or message_id gracefully."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=None),
        message_id=None,
    )

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── on_processing_complete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_reaction(monkeypatch):
    """Cancelled processing should clear the in-progress reaction.

    Without this clear, the 👀 reaction lingers on the user's message
    indefinitely (until another agent run swaps it for 👍/👎). On a
    ``/stop`` that ends a session, that reaction never gets cleaned up.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    # set_message_reaction with reaction=None clears all reactions on the
    # message (Bot API documented semantics; equivalent to Bot API 10.0's
    # deleteMessageReaction but works on PTB 22.6 already).
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


@pytest.mark.asyncio
async def test_clear_reactions_handles_api_error_gracefully(monkeypatch):
    """API errors during clear should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._clear_reactions("123", "456")
    assert result is False


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.reactions to TELEGRAM_REACTIONS env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"


# ── _reaction_style resolution ───────────────────────────────────────


def test_style_defaults_to_receipt_when_key_absent(monkeypatch):
    """No ``reaction_style`` anywhere = the pre-existing 👀/👍 receipts (back-compat)."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    adapter = _make_adapter()
    assert adapter._reaction_style() == "receipt"


def test_style_reads_yaml_extra(monkeypatch):
    """``extra.reaction_style`` from config.yaml (per profile) is the YAML rung."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "content"
    assert adapter._reaction_style() == "content"


def test_style_env_wins_over_yaml(monkeypatch):
    """An explicit TELEGRAM_REACTION_STYLE beats the YAML value, like TELEGRAM_REACTIONS does."""
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "content")
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "receipt"
    assert adapter._reaction_style() == "content"


def test_style_unknown_value_warns_once_and_falls_back_to_receipt(monkeypatch, caplog):
    """A typo'd style must not crash a turn: one warning per adapter, then ``receipt``."""
    from plugins.platforms.telegram import adapter as telegram_adapter

    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "reciept"

    with caplog.at_level(logging.WARNING, logger=telegram_adapter.logger.name):
        assert [adapter._reaction_style() for _ in range(3)] == ["receipt"] * 3

    warnings = [r for r in caplog.records if "unknown reaction_style" in r.message]
    assert len(warnings) == 1


def test_reactions_false_master_switch_wins_over_style(monkeypatch):
    """``reactions: false`` stays authoritative — no style value re-enables the receipts."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "receipt")
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "receipt"

    assert adapter._reactions_enabled() is False
    assert adapter._lifecycle_reactions_enabled() is False


# ── lifecycle reactions per style ────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,final",
    [(ProcessingOutcome.SUCCESS, "\U0001f44d"), (ProcessingOutcome.FAILURE, "\U0001f44e"),
     (ProcessingOutcome.CANCELLED, None)],
)
async def test_receipt_style_keeps_lifecycle_reactions(monkeypatch, outcome, final):
    """Style ``receipt`` = exactly today's behaviour: 👀 on start, then 👍/👎 (None = cleared)."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)

    assert [c.kwargs["reaction"] for c in adapter._bot.set_message_reaction.await_args_list] == ["\U0001f440", final]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_content_style_posts_no_lifecycle_reactions(monkeypatch, outcome):
    """Style ``content``: zero automatic calls, whatever the outcome — the agent reacts only if it
    decides to (``send_message action="react"``)."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "content"
    event = _make_event()

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_off_style_posts_no_lifecycle_reactions(monkeypatch):
    """Style ``off`` (via env) behaves like reactions disabled."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "off")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── agent-facing add_reaction / remove_reaction ──────────────────────


@pytest.mark.asyncio
async def test_add_reaction_wraps_set_reaction():
    """``send_message action="react"`` reaches the Bot API through the existing setter."""
    adapter = _make_adapter()

    assert await adapter.add_reaction("123", "\U0001f44d", message_id="456") is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(chat_id=123, message_id=456, reaction="\U0001f44d")


@pytest.mark.asyncio
async def test_remove_reaction_clears_via_set_reaction():
    adapter = _make_adapter()

    assert await adapter.remove_reaction("123", message_id="456") is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(chat_id=123, message_id=456, reaction=None)


@pytest.mark.asyncio
async def test_agent_reaction_without_message_id_fails_closed():
    """Telegram keeps no per-chat "latest inbound": an omitted id fails rather than guessing."""
    adapter = _make_adapter()

    assert await adapter.add_reaction("123", "\U0001f44d") is False
    assert await adapter.remove_reaction("123") is False
    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_reaction_is_not_gated_by_the_lifecycle_switches(monkeypatch):
    """Deliberate reactions are not receipts: the master switch and the style leave them alone."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    adapter.config.extra["reaction_style"] = "content"

    assert adapter._lifecycle_reactions_enabled() is False
    assert await adapter.add_reaction("123", "\U0001f44d", message_id="456") is True


# ── reaction_style in config.yaml ────────────────────────────────────


def test_yaml_nested_extra_style_reaches_adapter(monkeypatch):
    """The live config shape ``platforms.telegram.extra.{reactions,reaction_style}`` must reach
    PlatformConfig.extra (the nested form is passed through by _apply_yaml_config, not bridged to env)."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)

    extras = _apply_yaml_config({}, {"extra": {"reactions": True, "reaction_style": "content"}})

    adapter = _make_adapter()
    adapter.config.extra.update(extras or {})
    assert adapter._reactions_enabled() is True
    assert adapter._lifecycle_reactions_enabled() is False


def test_yaml_flat_reaction_style_bridges_env(monkeypatch):
    """The documented flat form ``telegram.reaction_style`` also seeds extra and bridges the env var."""
    import os
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    # setenv (not delenv) so monkeypatch registers cleanup: _apply_yaml_config writes os.environ
    # directly and would otherwise leak the bridged value into the rest of the session.
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "")

    extras = _apply_yaml_config({}, {"reaction_style": "content"})

    assert extras["reaction_style"] == "content"
    assert os.getenv("TELEGRAM_REACTION_STYLE") == "content"
