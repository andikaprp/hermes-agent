"""The agent's reaction surface on a messaging session (Telegram first).

One tool, two surfaces. A chat turn whose live adapter implements the reaction API reacts ON the
platform — a real emoji on the user's own bubble — while a desktop turn keeps the session store +
renderer path it always had. Which sessions are handed the tool at all is the toolset resolver's
call (never a process env var): a CLI/cron/api_server turn, or a platform whose adapter has no
reaction API, never sees it. The reaction policy itself lives in the tool description (pinned at
the bottom so a rewrite has to be deliberate).
"""

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.react_to_message_tool as reactions
from tools import desktop_ui
from tools.registry import registry

PLATFORM = "telegram"


class _FakeTelegramAdapter:
    """Stand-in for a platform adapter with the reaction API (its own slice adds it)."""

    def __init__(self, result: object = True):
        self.calls = []
        self._result = result

    def toolsets_for_source(self, source):
        return None

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return self._result

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("remove", chat_id, message_id))
        return self._result


class _NoReactionAdapter:
    """A platform whose bot API has no reactions at all."""

    def toolsets_for_source(self, source):
        return None


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform(PLATFORM): adapter})


@contextlib.contextmanager
def _telegram_turn(adapter):
    """A Telegram turn as the gateway binds it (triggering message id 555)."""
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform=PLATFORM, source=PLATFORM, chat_id="123", session_key="telegram:123",
        session_id="telegram:123", message_id="555",
    )
    try:
        yield adapter
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def live_telegram(monkeypatch):
    """Bind a Telegram turn behind ``adapter`` (default: one that reports success)."""

    def _bind(adapter=None):
        adapter = adapter if adapter is not None else _FakeTelegramAdapter()
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter))
        return _telegram_turn(adapter)

    return _bind


def _react(args):
    """Dispatch through the registry, the way the model's call arrives."""
    raw = registry.dispatch("react_to_message", args)
    return json.loads(raw if isinstance(raw, str) else json.dumps(raw))


class TestPlatformSurface:
    def test_reacts_on_the_triggering_message(self, live_telegram, monkeypatch):
        monkeypatch.setattr(reactions, "_open_session_db", lambda: None)

        with live_telegram() as adapter:
            result = _react({"emoji": "👍"})

        assert adapter.calls == [("add", "123", "👍", "555")]
        assert result == {"success": True, "platform": PLATFORM, "message_id": "555", "emoji": "👍"}

    def test_messages_back_targets_the_previous_user_message(self, live_telegram, monkeypatch):
        db = MagicMock()
        db.latest_message_row_id.return_value = 42
        db.message_platform_id.return_value = "444"
        monkeypatch.setattr(reactions, "_open_session_db", lambda: db)

        with live_telegram() as adapter:
            result = _react({"emoji": "😂", "messages_back": 1})

        assert adapter.calls == [("add", "123", "😂", "444")]
        db.latest_message_row_id.assert_called_once_with("telegram:123", role="user", offset=1)
        assert result["message_id"] == "444"
        # Mirrored into the store under the agent's author, like the desktop tapback.
        db.set_message_reaction.assert_called_once_with("telegram:123", 42, "😂", author="agent")

    def test_an_empty_emoji_retracts(self, live_telegram, monkeypatch):
        monkeypatch.setattr(reactions, "_open_session_db", lambda: None)

        with live_telegram() as adapter:
            result = _react({"emoji": ""})

        assert adapter.calls == [("remove", "123", "555")]
        assert result["success"] is True

    def test_a_platform_refusal_reaches_the_model(self, live_telegram, monkeypatch):
        monkeypatch.setattr(reactions, "_open_session_db", lambda: None)
        adapter = _FakeTelegramAdapter(result={"error": "Bad Request: message to react not found"})

        with live_telegram(adapter):
            result = _react({"emoji": "👀"})

        assert result.get("success") is not True
        assert "not found" in json.dumps(result)


class _Src:
    """The session's source, as the resolver receives it."""

    def __init__(self, chat_id):
        self.chat_id = chat_id


class TestSessionExposure:
    """Exposure is decided per SESSION by the toolset resolver (root AGENTS.md surface rule)."""

    @staticmethod
    def _runner(adapter):
        from gateway.run import GatewayRunner

        gr = object.__new__(GatewayRunner)
        gr._adapter_for_source = lambda source: adapter
        return gr

    def _resolve(self, adapter):
        from gateway.run import GatewayRunner

        return GatewayRunner._resolve_enabled_toolsets_for_source(
            self._runner(adapter), {}, _Src("telegram:123"), PLATFORM)

    def test_a_reaction_capable_adapter_earns_the_toolset(self):
        assert "message_reactions" in self._resolve(_FakeTelegramAdapter())

    def test_a_platform_without_the_reaction_api_does_not(self):
        assert "message_reactions" not in self._resolve(_NoReactionAdapter())

    def test_no_live_adapter_does_not(self):
        # Standalone/cron: the resolver has no adapter for the source at all.
        assert "message_reactions" not in self._resolve(None)

    def test_other_surfaces_never_fold_it(self):
        from hermes_cli.tools_config import _get_platform_tools
        from toolsets import resolve_toolset

        for platform in ("cli", "cron", "api_server", "desktop"):
            assert "message_reactions" not in _get_platform_tools({}, platform), platform
        # Not a platform-bundle member either: a session only gets it from the fold above.
        assert "react_to_message" not in resolve_toolset("hermes-telegram")

    def test_a_telegram_session_sees_the_tool_and_a_cli_session_does_not(self, live_telegram):
        from hermes_cli.tools_config import _get_platform_tools
        from model_tools import get_tool_definitions

        def names(toolsets):
            return {t["function"]["name"] for t in get_tool_definitions(
                quiet_mode=True, enabled_toolsets=toolsets)}

        with live_telegram():
            assert "react_to_message" in names(["message_reactions"])  # the folded selection
        # A CLI session's selection never carries the toolset (the fold tests above), and the
        # desktop opt-in stays off by default — so the tool cannot reach that schema.
        assert "react_to_message" not in names(sorted(_get_platform_tools({}, "cli")))
        assert reactions.check_react_requirements() is False


class TestDesktopSurface:
    """The desktop path is unchanged: store first, renderer event when a bridge exists."""

    def test_persists_and_paints(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        db = MagicMock()
        db.latest_message_row_id.return_value = 7
        db.set_message_reaction.return_value = [{"emoji": "❤️", "author": "agent"}]
        monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
        painted = []
        monkeypatch.setattr(desktop_ui, "emit", lambda event, payload: painted.append((event, payload)) or True)
        tokens = set_session_vars(platform="desktop", source="desktop", session_key="s-1", session_id="s-1")
        try:
            result = _react({"emoji": "❤️"})
        finally:
            clear_session_vars(tokens)

        assert db.set_message_reaction.call_args.args == ("s-1", 7, "❤️")
        assert db.set_message_reaction.call_args.kwargs == {"author": "agent"}
        assert painted == [("message.reaction", {"row_id": 7, "reactions": [{"emoji": "❤️", "author": "agent"}],
                                                 "role": "user"})]
        assert result["success"] is True and result["row_id"] == 7


class TestPolicy:
    def test_the_description_carries_the_reaction_policy(self):
        description = reactions.REACT_TO_MESSAGE_SCHEMA["description"]

        for rule in (
            "Never react to every message by default",
            "treat consecutive bubbles sent close together as one thought",
            "react only to the final bubble",
            "👀 when taking a new request or investigating",
            "👍 when the user confirms, approves, corrects, or steers existing work",
            "✅ when requested work is fully done",
            "❤️ for thanks, warmth, or appreciation",
            "😂 or 💀 for something genuinely funny",
            "😢/🫂/😩 for a rough moment depending on the emotion",
            "❗ for something important or impressive",
            "❓ only when the natural response is genuine confusion",
            "If unsure, skip the reaction and reply normally",
            "Never narrate a reaction",
        ):
            assert rule in description, rule
