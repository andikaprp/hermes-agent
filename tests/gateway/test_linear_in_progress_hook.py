"""Linear in-progress turn-note hook: always-on agent:start surface for Labs issues.

The standing convention ("check in-progress issues at the start of any real task")
lives here as a builtin gateway hook so it cannot be skipped by a fresh session:
at ``agent:start`` the hook lists up to five in-progress LAB-* issues as a compact
turn-context note. Fail-closed everywhere: no token, expired token, network error,
timeout or malformed payload all degrade to no note and never block a turn.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from gateway.builtin_hooks import linear_in_progress as mod
from gateway.hooks import HookRegistry

_CANNED_ISSUES = [
    {"id": "LAB-48", "title": "Stop em/en dashes reaching Telegram"},
    {"id": "LAB-49", "title": "Voice channel UX polish"},
    {"id": "LAB-50", "title": "Quarterly report"},
    {"id": "LAB-51", "title": "Docs refresh"},
    {"id": "LAB-52", "title": "Bot onboarding flow"},
    {"id": "PROJ-53", "title": "Different team: must be filtered out"},
]


def _canned_mcp_result(issues=None):
    """An MCP ``tools/call`` result shaped like the Linear server's ``list_issues``."""
    return {
        "result": {
            "content": [{"type": "text", "text": json.dumps({"issues": issues or _CANNED_ISSUES})}]
        }
    }


def _fake_mcp_call(*args, **kwargs):
    return _canned_mcp_result()


@pytest.fixture(autouse=True)
def _clear_mod_cache():
    mod._clear_cache()
    yield
    mod._clear_cache()


def _token_file(home: Path) -> Path:
    path = home / "mcp-tokens" / "linear.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "access_token": "tok-sample",
        "expires_at": time.time() + 3600,
        "expires_in": 3600,
    }), encoding="utf-8")
    return path


def test_fetch_in_progress_calls_list_issues_with_the_team_state_filter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mod, "_mcp_call",
        lambda token, tool, arguments, timeout: (calls.append((tool, arguments)) or _fake_mcp_call()),
    )

    rows = mod.fetch_in_progress("tok")

    assert calls[0][0] == "list_issues"
    assert calls[0][1] == {"state": "started", "team": "Labs", "limit": 6}
    assert ("LAB-48", "Stop em/en dashes reaching Telegram") in rows


def test_parse_mcp_issues_filters_by_team_prefix_and_cleans_titles():
    rows = mod.parse_mcp_issues(_canned_mcp_result())

    assert ("LAB-48", "Stop em/en dashes reaching Telegram") in rows
    assert all(identifier.startswith("LAB-") for identifier, _ in rows)
    assert len(rows) == 5  # PROJ-53 is filtered out by the team identifier prefix


def test_parse_mcp_issues_survives_malformed_payloads():
    assert mod.parse_mcp_issues(None) == []
    assert mod.parse_mcp_issues({"result": {}}) == []
    assert mod.parse_mcp_issues({"result": {"content": []}}) == []
    assert mod.parse_mcp_issues({"result": {"content": [{"type": "text", "text": "not json"}]}}) == []
    assert mod.parse_mcp_issues({"result": {"content": [{"type": "text", "text": json.dumps({"issues": [None, {}, {"id": 3}]})}]}}) == []


def test_build_note_caps_at_five_and_marks_extra():
    rows = [(f"LAB-{i}", f"issue {i}") for i in range(7)]

    note = mod.build_note(rows)

    assert note is not None
    assert "LAB-1" in note and "LAB-4" in note
    assert "LAB-5" not in note
    assert "(+2 more)" in note


def test_build_note_returns_none_for_empty():
    assert mod.build_note([]) is None


def test_handle_returns_note_with_token_and_fetch(tmp_path, monkeypatch):
    _token_file(tmp_path)
    monkeypatch.setattr(mod, "_read_cached_access_token", lambda home: "tok")
    monkeypatch.setattr(mod, "_mcp_call", _fake_mcp_call)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    note = asyncio.run(mod.handle("agent:start", {}))

    assert note is not None
    assert "LAB-48" in note
    assert "check for a matching issue" in note


def test_handle_fails_closed_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_read_cached_access_token", lambda home: None)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    assert asyncio.run(mod.handle("agent:start", {})) is None


def test_handle_fails_closed_on_fetch_error(tmp_path, monkeypatch):
    _token_file(tmp_path)
    monkeypatch.setattr(mod, "_read_cached_access_token", lambda home: "tok")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "_mcp_call", boom)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    assert asyncio.run(mod.handle("agent:start", {})) is None


def test_handle_ignores_other_events(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    assert asyncio.run(mod.handle("agent:end", {})) is None
    assert asyncio.run(mod.handle("", {})) is None


def test_builtin_hook_is_registered_for_agent_start(tmp_path, monkeypatch):
    _token_file(tmp_path)
    monkeypatch.setattr(mod, "_read_cached_access_token", lambda home: "tok")
    monkeypatch.setattr(mod, "_mcp_call", _fake_mcp_call)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)

    registry = HookRegistry()
    registry.discover_and_load()

    names = [hook["name"] for hook in registry.loaded_hooks]
    assert "linear-in-progress" in names

    results = asyncio.run(registry.emit_collect("agent:start", {}))
    assert results and "LAB-48" in results[0]


def test_builtin_registration_is_idempotent(tmp_path):
    registry = HookRegistry()
    registry.discover_and_load()
    first = len(registry._handlers.get("agent:start", []))
    registry.discover_and_load()
    second = len(registry._handlers.get("agent:start", []))

    assert first == 1 and second == 1