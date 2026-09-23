"""Runtime burst grouping at the Telegram reaction seam.

Consecutive inbound bubbles from the same chat within a short window are one
burst. A reaction aimed at a non-final bubble must not hit the Bot API while
that burst is still open; it is deferred and lands on the final bubble once
the window closes. Lifecycle receipts keep their own path (``_set_reaction``)
and are not regrouped. The window is a config.yaml knob
(``extra.reaction_burst_window``, seconds) — no new env var.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource

CHAT = "123"
OTHER_CHAT = "999"
HEART = "\u2764\ufe0f"
EYES = "\U0001f440"


class _Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _adapter(clock: _Clock, **extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter.gateway_runner = None
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock(return_value=True)
    adapter.config.extra.update(extra)
    # Injectable clock. Production uses time.monotonic when this is absent.
    adapter._reaction_burst_clock = clock
    return adapter


def _bubble(message_id, chat_id=CHAT):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type="private", title=None, full_name="Test"),
        from_user=SimpleNamespace(id=42, full_name="TestUser", is_bot=False),
        message_id=message_id,
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=None,
        date=None,
    )


def _note(adapter, message_id, chat_id=CHAT):
    """The production inbound path: building the event records the bubble."""
    return adapter._build_message_event(_bubble(message_id, chat_id), MessageType.TEXT)


def _fired(adapter):
    return [
        (c.kwargs["chat_id"], c.kwargs["message_id"], c.kwargs["reaction"])
        for c in adapter._bot.set_message_reaction.await_args_list
    ]


# ── open burst: non-final must not fire ──────────────────────────────


@pytest.mark.asyncio
async def test_reaction_on_a_non_final_bubble_does_not_fire_while_the_burst_is_open():
    """Two bubbles inside the window are one open burst. Reacting to the first
    must not call the Bot API on that bubble (or at all) while the window is open."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.4
    _note(adapter, 102)

    clock.t = 0.5
    accepted = await adapter.add_reaction(CHAT, HEART, message_id="101")

    assert accepted is True
    assert _fired(adapter) == []


@pytest.mark.asyncio
async def test_deferred_reaction_lands_on_the_final_bubble_when_the_burst_closes():
    """The held reaction is retargeted: once the window elapses it fires once,
    on the final bubble, and never on the non-final one."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.4
    _note(adapter, 102)
    clock.t = 0.5
    await adapter.add_reaction(CHAT, HEART, message_id="101")
    assert _fired(adapter) == []

    clock.t = 0.4 + 2.0 + 0.01  # default window is 2s after the last bubble
    await adapter.flush_due_reaction_bursts()

    assert _fired(adapter) == [(int(CHAT), 102, HEART)]


@pytest.mark.asyncio
async def test_a_later_bubble_retargets_the_deferred_reaction():
    """While the burst stays open, a newer bubble becomes the final one.
    The deferred reaction follows it instead of sticking to the previous last."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.3
    _note(adapter, 102)
    clock.t = 0.4
    await adapter.add_reaction(CHAT, HEART, message_id="101")
    clock.t = 0.8
    _note(adapter, 103)
    assert _fired(adapter) == []

    clock.t = 0.8 + 2.0 + 0.01
    await adapter.flush_due_reaction_bursts()

    assert _fired(adapter) == [(int(CHAT), 103, HEART)]


@pytest.mark.asyncio
async def test_reaction_on_the_current_final_bubble_fires_immediately():
    """The constraint is the non-final bubble. The current final may fire now."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.4
    _note(adapter, 102)

    assert await adapter.add_reaction(CHAT, HEART, message_id="102") is True

    assert _fired(adapter) == [(int(CHAT), 102, HEART)]


@pytest.mark.asyncio
async def test_retract_on_a_non_final_bubble_does_not_fire_while_open():
    """An empty retract is the same seam: it must not clear the non-final bubble
    while the burst is open. When the burst closes, the clear lands on the final."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.2
    _note(adapter, 102)

    assert await adapter.remove_reaction(CHAT, message_id="101") is True
    assert _fired(adapter) == []

    clock.t = 0.2 + 2.0 + 0.01
    await adapter.flush_due_reaction_bursts()

    assert _fired(adapter) == [(int(CHAT), 102, None)]


# ── window, chat isolation, back-compat ──────────────────────────────


@pytest.mark.asyncio
async def test_bubbles_outside_the_window_are_separate_bursts():
    """A gap larger than the window closes the first burst. Reacting to its
    (now final) bubble fires immediately — it is not held for the next thought."""
    clock = _Clock(0.0)
    adapter = _adapter(clock, reaction_burst_window=1.0)
    _note(adapter, 101)
    clock.t = 1.5  # > 1.0s after 101
    _note(adapter, 102)

    assert await adapter.add_reaction(CHAT, HEART, message_id="101") is True

    assert _fired(adapter) == [(int(CHAT), 101, HEART)]


@pytest.mark.asyncio
async def test_window_from_config_extra_groups_only_inside_that_window():
    """``extra.reaction_burst_window`` (seconds) is the knob. 0.4s groups a
    0.3s pair and does not group a 0.5s pair."""
    clock = _Clock(0.0)
    close = _adapter(clock, reaction_burst_window=0.4)
    _note(close, 101)
    clock.t = 0.3
    _note(close, 102)
    clock.t = 0.35
    await close.add_reaction(CHAT, HEART, message_id="101")
    assert _fired(close) == []

    clock.t = 10.0
    far = _adapter(clock, reaction_burst_window=0.4)
    _note(far, 201)
    clock.t = 10.5  # 0.5s > 0.4s
    _note(far, 202)
    assert await far.add_reaction(CHAT, HEART, message_id="201") is True
    assert _fired(far) == [(int(CHAT), 201, HEART)]


@pytest.mark.asyncio
async def test_default_window_is_two_seconds_and_ignores_env(monkeypatch):
    """No key in config.yaml → 2.0s. A HERMES_* or TELEGRAM_* env var must not
    override that (config.yaml only)."""
    monkeypatch.setenv("HERMES_REACTION_BURST_WINDOW", "0.1")
    monkeypatch.setenv("TELEGRAM_REACTION_BURST_WINDOW", "0.1")
    clock = _Clock(0.0)
    adapter = _adapter(clock)  # no extra key
    _note(adapter, 101)
    clock.t = 1.5  # inside 2.0s, outside the env's 0.1s
    _note(adapter, 102)
    clock.t = 1.6
    await adapter.add_reaction(CHAT, HEART, message_id="101")

    assert _fired(adapter) == []


@pytest.mark.asyncio
async def test_different_chats_do_not_form_one_burst():
    """Grouping is per chat. A lone bubble in another chat is its own final."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101, CHAT)
    clock.t = 0.2
    _note(adapter, 202, OTHER_CHAT)

    assert await adapter.add_reaction(OTHER_CHAT, HEART, message_id="202") is True
    assert await adapter.add_reaction(CHAT, HEART, message_id="101") is True

    assert _fired(adapter) == [
        (int(OTHER_CHAT), 202, HEART),
        (int(CHAT), 101, HEART),
    ]


@pytest.mark.asyncio
async def test_a_message_that_was_never_noted_still_fires():
    """No burst context (tests, CLI, a message we did not see inbound) keeps
    today's behaviour: the named id is reacted to immediately."""
    clock = _Clock(0.0)
    adapter = _adapter(clock)

    assert await adapter.add_reaction(CHAT, HEART, message_id="456") is True

    assert _fired(adapter) == [(int(CHAT), 456, HEART)]


@pytest.mark.asyncio
async def test_lifecycle_receipts_are_not_regrouped(monkeypatch):
    """Receipt style still posts 👀 on the event's own bubble, even when that
    bubble is not the final one of an open burst. Burst grouping is the agent
    reaction seam, not the lifecycle receipts."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.3
    _note(adapter, 102)
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="private",
            user_id="42", user_name="TestUser",
        ),
        message_id="101",
    )

    await adapter.on_processing_start(event)

    assert _fired(adapter) == [(int(CHAT), 101, EYES)]


def test_absent_reaction_style_still_defaults_to_receipt(monkeypatch):
    """Back-compat stays: a config that never set reaction_style is receipt.
    Burst grouping must not change that default."""
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    adapter = _adapter(_Clock())
    assert adapter._reaction_style() == "receipt"


def test_yaml_flat_window_is_seeded_without_an_env_bridge(monkeypatch):
    """``telegram.reaction_burst_window`` reaches extra. It must not invent a
    TELEGRAM_* / HERMES_* env var."""
    import os
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    monkeypatch.delenv("TELEGRAM_REACTION_BURST_WINDOW", raising=False)
    monkeypatch.delenv("HERMES_REACTION_BURST_WINDOW", raising=False)

    extras = _apply_yaml_config({}, {"reaction_burst_window": 0.75}) or {}

    assert extras.get("reaction_burst_window") == 0.75
    assert os.getenv("TELEGRAM_REACTION_BURST_WINDOW") is None
    assert os.getenv("HERMES_REACTION_BURST_WINDOW") is None


def test_yaml_nested_extra_window_reaches_the_adapter(monkeypatch):
    """The live shape ``platforms.telegram.extra.reaction_burst_window`` is what
    the adapter reads. No env bridge."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    monkeypatch.delenv("TELEGRAM_REACTION_BURST_WINDOW", raising=False)
    extras = _apply_yaml_config({}, {"extra": {"reaction_burst_window": 0.75}}) or {}
    adapter = _adapter(_Clock())
    adapter.config.extra.update(extras)

    assert adapter._reaction_burst_window_s() == 0.75


@pytest.mark.asyncio
async def test_production_clock_flushes_the_held_reaction_when_the_window_closes():
    """No injected clock: ``add_reaction`` arms a real sleep and the Bot API
    call lands on the final bubble after the window, without an explicit flush."""
    import asyncio

    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter.gateway_runner = None
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock(return_value=True)
    adapter.config.extra["reaction_burst_window"] = 0.05
    _note(adapter, 101)
    _note(adapter, 102)

    assert await adapter.add_reaction(CHAT, HEART, message_id="101") is True
    assert _fired(adapter) == []

    await asyncio.sleep(0.25)
    assert _fired(adapter) == [(int(CHAT), 102, HEART)]


@pytest.mark.asyncio
async def test_tool_reaction_on_a_non_final_bubble_does_not_fire_while_open(monkeypatch):
    """``react_to_message`` → real ``add_reaction``. Targeting the first bubble
    of an open burst must not reach ``set_message_reaction`` until the burst
    closes, and then only on the final bubble."""
    import tools.react_to_message_tool as reactions  # noqa: F401 — registers the tool
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.registry import registry

    clock = _Clock(0.0)
    adapter = _adapter(clock)
    _note(adapter, 101)
    clock.t = 0.4
    _note(adapter, 102)

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(reactions, "_open_session_db", lambda: None)

    clock.t = 0.5
    tokens = set_session_vars(
        platform="telegram", source="telegram", chat_id=CHAT, session_key="telegram:123",
        session_id="telegram:123", message_id="101",
    )
    try:
        raw = registry.dispatch("react_to_message", {"emoji": HEART})
    finally:
        clear_session_vars(tokens)

    result = json.loads(raw if isinstance(raw, str) else json.dumps(raw))
    assert result.get("success") is True
    assert _fired(adapter) == []

    clock.t = 0.4 + 2.0 + 0.01
    await adapter.flush_due_reaction_bursts()
    assert _fired(adapter) == [(int(CHAT), 102, HEART)]
