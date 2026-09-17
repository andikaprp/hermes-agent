"""``send_message(action='react', reaction_only=True)`` — the react-as-the-reply claim.

The claim is the only way a caller can say "this reaction IS my reply". It is passed to the
adapter only where the adapter can judge it against the inbound message (Telegram); everywhere
else the kwarg is dropped so the reaction behaves exactly as before.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.send_message_tool as smt
from gateway.config import Platform, PlatformConfig


class _FakePhotonAdapter:
    """A reaction-capable adapter with no guarded react-as-reply support."""

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "emoji": emoji}


def _runner_with(platform, adapter):
    return SimpleNamespace(adapters={platform: adapter})


def _call(args):
    return json.loads(smt.send_message_tool(args))


def _telegram_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from unittest.mock import AsyncMock

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def test_schema_advertises_the_claim_as_an_optional_boolean():
    props = smt.SEND_MESSAGE_SCHEMA["parameters"]["properties"]
    assert props["reaction_only"]["type"] == "boolean"
    assert "reaction_only" not in smt.SEND_MESSAGE_SCHEMA["parameters"]["required"]


def test_plain_react_never_passes_the_claim(monkeypatch):
    """Default path unchanged: no claim, no extra kwarg, same result shape."""
    adapter = _telegram_adapter()
    adapter.config.extra = {}
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform.TELEGRAM, adapter)):
        result = _call({"action": "react", "target": "telegram:123", "emoji": "👍", "message_id": "456"})
    assert result == {"success": True, "message_id": "456"}


def test_claim_is_dropped_for_adapters_that_cannot_judge_it(monkeypatch):
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform("photon"), adapter)):
        result = _call({"action": "react", "target": "photon:+155****4567", "emoji": "❤️",
                        "message_id": "1", "reaction_only": True})
    assert result["success"] is True
    assert adapter.calls == [("add", "+155****4567", "❤️", "1")]


def test_claim_against_a_question_returns_text_required(monkeypatch):
    adapter = _telegram_adapter()
    adapter.config.extra = {}
    adapter._reaction_inbound_text = {"123": "what's the status?"}
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "content")
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform.TELEGRAM, adapter)):
        result = _call({"action": "react", "target": "telegram:123", "emoji": "❤",
                        "message_id": "456", "reaction_only": True})
    assert result["success"] is True          # the glyph still lands on the message
    assert result["text_required"] is True    # …but it may not be the answer
    assert result["reaction_only"] is False
    assert result["reason"] == "question_asked"
    assert adapter._bot.set_message_reaction.call_args.kwargs["reaction"] == "❤"


def test_claim_against_a_social_message_is_granted(monkeypatch):
    adapter = _telegram_adapter()
    adapter.config.extra = {}
    adapter._reaction_inbound_text = {"123": "thank you so much 🙏"}
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_STYLE", "content")
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform.TELEGRAM, adapter)):
        result = _call({"action": "react", "target": "telegram:123", "emoji": "🙏",
                        "message_id": "456", "reaction_only": True})
    assert result["reaction_only"] is True
    assert result["text_required"] is False
    assert result["reason"] == "social"


def test_claim_is_refused_under_the_default_style(monkeypatch):
    """No style configured means the capability is off, so the claim can never grant silence."""
    adapter = _telegram_adapter()
    adapter.config.extra = {}
    adapter._reaction_inbound_text = {"123": "thank you so much 🙏"}
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_STYLE", raising=False)
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform.TELEGRAM, adapter)):
        result = _call({"action": "react", "target": "telegram:123", "emoji": "🙏",
                        "message_id": "456", "reaction_only": True})
    assert result["text_required"] is True
    assert result["reason"] == "style_disabled"


def test_unreact_ignores_the_claim(monkeypatch):
    adapter = _telegram_adapter()

    async def _remove(chat_id, message_id=None):
        return {"success": True, "message_id": str(message_id)}

    adapter.remove_reaction = _remove
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(Platform.TELEGRAM, adapter)):
        result = _call({"action": "unreact", "target": "telegram:123", "message_id": "456",
                        "reaction_only": True})
    assert result["success"] is True


PROFILE_ENV_KEYS = ("TELEGRAM_REACTIONS", "TELEGRAM_REACTION_STYLE")


@pytest.fixture(autouse=True)
def _clean_env():
    """Keep this module from leaking a style/reactions value into the rest of the worker."""
    before = {key: os.environ.get(key) for key in PROFILE_ENV_KEYS}
    yield
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
