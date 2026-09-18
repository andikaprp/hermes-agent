"""LAB-52 compact fast lane: transcript builder + fail-soft fallback.

These tests are red on main (module absent). They pin the behaviour contract:
count + char budgets, always-include latest user message, no em/en dashes in
builder output, and provider error → fall back to the one-hop path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gateway.run_turn_fast_lane import (
    FAST_LANE_MARKER,
    FAST_LANE_SYSTEM,
    build_compact_transcript,
    is_fast_lane_enabled,
    load_fast_lane_config,
    log_fast_lane,
    sanitize_fast_lane_reply,
    try_fast_lane,
)


def _history(*pairs):
    """Build alternating user/assistant rows from (role, text) pairs."""
    return [{"role": role, "content": text} for role, text in pairs]


# ── compact transcript builder ────────────────────────────────────────────


def test_builder_includes_system_and_latest_user():
    msgs = build_compact_transcript([], "ya")
    assert msgs[0] == {"role": "system", "content": FAST_LANE_SYSTEM}
    assert msgs[-1] == {"role": "user", "content": "ya"}
    assert "\u2014" not in FAST_LANE_SYSTEM
    assert "\u2013" not in FAST_LANE_SYSTEM


def test_builder_respects_message_count_limit():
    history = _history(
        ("user", "one"), ("assistant", "a1"),
        ("user", "two"), ("assistant", "a2"),
        ("user", "three"), ("assistant", "a3"),
        ("user", "four"), ("assistant", "a4"),
    )
    msgs = build_compact_transcript(history, "latest", max_messages=3, max_chars=10_000)
    # system + 3 turns
    assert len(msgs) == 4
    assert msgs[-1]["content"] == "latest"
    assert all(m["role"] in ("system", "user", "assistant") for m in msgs)


def test_builder_always_keeps_latest_user_when_over_char_budget():
    history = _history(
        ("user", "x" * 500), ("assistant", "y" * 500),
        ("user", "z" * 500), ("assistant", "w" * 500),
    )
    msgs = build_compact_transcript(history, "KEEP_ME", max_messages=6, max_chars=50)
    assert msgs[-1]["role"] == "user"
    assert "KEEP_ME" in msgs[-1]["content"]
    body_chars = sum(len(m["content"]) for m in msgs[1:])
    assert body_chars <= 50


def test_builder_truncates_oversized_latest_user_rather_than_dropping_it():
    msgs = build_compact_transcript([], "Z" * 500, max_messages=6, max_chars=40)
    assert len(msgs) == 2
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"].startswith("Z")
    assert len(msgs[-1]["content"]) <= 41  # budget + optional ellipsis


def test_builder_strips_tool_and_internal_rows():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result"},
        {"role": "system", "content": "ignore"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "[hermes:fast-path] note\n\nok"},
    ]
    msgs = build_compact_transcript(history, "bye", max_messages=10, max_chars=10_000)
    roles = [m["role"] for m in msgs[1:]]
    assert "tool" not in roles
    assert "system" not in roles[1:] or all(m["role"] != "system" for m in msgs[1:])
    assert not any("[hermes:" in (m.get("content") or "") for m in msgs[1:])
    assert msgs[-1]["content"] == "bye"
    assert {"role": "user", "content": "hi"} in msgs
    assert {"role": "assistant", "content": "hello"} in msgs


def test_builder_output_never_contains_em_or_en_dashes():
    history = _history(
        ("user", "wait\u2014really"),
        ("assistant", "yes\u2013ok"),
    )
    msgs = build_compact_transcript(history, "cool\u2014thanks")
    blob = "\n".join(m["content"] for m in msgs)
    assert "\u2014" not in blob
    assert "\u2013" not in blob


def test_sanitize_reply_strips_em_dashes():
    assert "\u2014" not in sanitize_fast_lane_reply("hi\u2014there\u2013friend")
    assert sanitize_fast_lane_reply("  ok  ") == "ok"


# ── config gate ───────────────────────────────────────────────────────────


def test_fast_lane_defaults_on_when_absent():
    assert is_fast_lane_enabled({}) is True
    assert is_fast_lane_enabled(None) is True
    cfg = load_fast_lane_config({})
    assert cfg["max_messages"] == 6
    assert cfg["max_chars"] == 2000
    assert cfg["ttft_budget_ms"] == 8000


@pytest.mark.parametrize("raw", [False, "false", "off", "0", "no"])
def test_fast_lane_off_tokens(raw):
    assert is_fast_lane_enabled({"gateway": {"telegram": {"fast_lane": {"enabled": raw}}}}) is False


def test_fast_lane_config_from_user_yaml_loader(tmp_path):
    from hermes_cli.config_effective import load_user_config_effective

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "gateway:\n  telegram:\n    fast_lane:\n      enabled: false\n      max_messages: 4\n",
        encoding="utf-8",
    )
    cfg = load_user_config_effective(cfg_path)
    assert is_fast_lane_enabled(cfg) is False
    loaded = load_fast_lane_config(cfg)
    assert loaded["max_messages"] == 4


def test_default_config_declares_fast_lane_block():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    block = DEFAULT_CONFIG["gateway"]["telegram"]["fast_lane"]
    assert block["enabled"] is True
    assert block["max_messages"] >= 1
    assert block["max_chars"] >= 1
    assert block["ttft_budget_ms"] >= 1
    assert block["base_url"] == ""
    assert block["api_key"] == ""


# ── metrics line ──────────────────────────────────────────────────────────


def test_fast_lane_log_line_shape(caplog):
    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        log_fast_lane(
            provider="openrouter", ttft_ms=120.4, ready_ms=400.1,
            fallback=False, chat_id="998877",
        )
    assert FAST_LANE_MARKER in caplog.text
    assert "provider=openrouter" in caplog.text
    assert "ttft_ms=120.4" in caplog.text
    assert "ready_ms=400.1" in caplog.text
    assert "fallback=false" in caplog.text
    assert "998877" not in caplog.text


# ── try_fast_lane: success + fallback ─────────────────────────────────────


def _chunk(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_try_fast_lane_streams_and_returns_session_shaped_result():
    deltas: list[str] = []

    def fake_call_llm(**kwargs):
        assert kwargs.get("stream") is True
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][-1]["content"] == "ya"
        return iter([_chunk("sip"), _chunk("!")])

    result = try_fast_lane(
        history=_history(("assistant", "halo")),
        user_message="ya",
        main_runtime={"provider": "test", "model": "m", "api_key": "k"},
        on_delta=lambda t: deltas.append(t) if t else None,
        call_llm_fn=fake_call_llm,
    )
    assert result is not None
    assert result["final_response"] == "sip!"
    assert result["api_calls"] == 1
    assert result["agent_persisted"] is False
    assert result["messages"][-2] == {"role": "user", "content": "ya"}
    assert result["messages"][-1]["content"] == "sip!"
    assert deltas == ["sip", "!"]


def test_try_fast_lane_falls_back_on_provider_error(caplog):
    def boom(**kwargs):
        raise RuntimeError("upstream down")

    with caplog.at_level(logging.INFO, logger="gateway.run_turn"):
        result = try_fast_lane(
            history=[],
            user_message="hi",
            main_runtime={"provider": "test", "model": "m", "api_key": "k"},
            call_llm_fn=boom,
            chat_id="42",
        )
    assert result is None
    assert "fallback=true" in caplog.text
    assert "42" not in caplog.text


def test_try_fast_lane_falls_back_on_ttft_budget():
    class _Clock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    clock = _Clock()

    def slow_stream(**kwargs):
        def _gen():
            clock.now = 9.0  # past 8000ms budget before first content
            yield _chunk("too late")
        return _gen()

    result = try_fast_lane(
        history=[],
        user_message="hi",
        user_config={"gateway": {"telegram": {"fast_lane": {"ttft_budget_ms": 8000}}}},
        main_runtime={"provider": "test", "model": "m", "api_key": "k"},
        call_llm_fn=slow_stream,
        clock=clock,
    )
    assert result is None


def test_try_fast_lane_disabled_returns_none_without_calling_provider():
    called = []

    def fake(**kwargs):
        called.append(1)
        return iter([_chunk("x")])

    result = try_fast_lane(
        history=[],
        user_message="hi",
        user_config={"gateway": {"telegram": {"fast_lane": {"enabled": False}}}},
        main_runtime={"provider": "test", "model": "m", "api_key": "k"},
        call_llm_fn=fake,
    )
    assert result is None
    assert called == []


def test_runner_falls_back_to_one_hop_when_fast_lane_errors(monkeypatch):
    """Fail-soft seam: provider error must reach ``_run_conversation_with_approval``."""
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    class _Stub:
        session_store = None

        def _adapter_for_source(self, source):
            return None

    class _Source:
        platform = Platform.TELEGRAM
        chat_id = "4242"
        chat_type = "dm"

    ctx = TurnContext(
        source=_Source(), message="ya", session_key="telegram:4242",
        history=[], _run_still_current=lambda: True,
        user_config={"gateway": {"telegram": {"fast_path": True, "fast_lane": {"enabled": True}}}},
        fast_path_taken="ack",
    )
    runner = TurnRunner(_Stub(), ctx)
    seen = {"conv": 0}

    def fake_conv(*a, **k):
        seen["conv"] += 1
        return {
            "final_response": "from-one-hop", "messages": [], "api_calls": 1,
            "agent_persisted": True,
        }

    monkeypatch.setattr(
        "gateway.run_turn_fast_lane.try_fast_lane",
        lambda **kw: None,  # simulated provider/TTFT failure
    )
    monkeypatch.setattr(runner, "_run_conversation_with_approval", fake_conv)

    out = runner._try_fast_lane_or_conversation(
        agent=None, agent_history=[], observed_group_context=None,
        persist_msg="ya", persist_ts=None, stream_delta_cb=None,
        model="m", runtime_kwargs={"provider": "p", "api_key": "k"},
    )
    assert out["final_response"] == "from-one-hop"
    assert seen["conv"] == 1


def test_builder_does_not_emit_consecutive_user_rows():
    history = _history(("user", "old"), ("assistant", "a"), ("user", "also inbound"))
    msgs = build_compact_transcript(history, "also inbound", max_messages=10, max_chars=10_000)
    roles = [m["role"] for m in msgs]
    assert roles[-1] == "user"
    assert "user,user" not in ",".join(roles)


def test_runner_updates_cached_agent_messages_on_lane_success(monkeypatch):
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    class _Stub:
        session_store = None

        def _adapter_for_source(self, source):
            return None

    class _Source:
        platform = Platform.TELEGRAM
        chat_id = "4242"
        chat_type = "dm"

    class _Agent:
        _session_messages = [{"role": "assistant", "content": "stale"}]

    ctx = TurnContext(
        source=_Source(), message="ya", session_key="telegram:4242",
        history=[], _run_still_current=lambda: True,
        user_config={"gateway": {"telegram": {"fast_path": True, "fast_lane": {"enabled": True}}}},
        fast_path_taken="ack",
    )
    runner = TurnRunner(_Stub(), ctx)
    agent = _Agent()
    lane_result = {
        "final_response": "sip",
        "messages": [
            {"role": "assistant", "content": "halo"},
            {"role": "user", "content": "ya"},
            {"role": "assistant", "content": "sip"},
        ],
        "api_calls": 1, "agent_persisted": False, "completed": True, "failed": False,
    }
    monkeypatch.setattr("gateway.run_turn_fast_lane.try_fast_lane", lambda **kw: lane_result)
    monkeypatch.setattr(
        runner, "_run_conversation_with_approval",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fallback")),
    )
    out = runner._try_fast_lane_or_conversation(
        agent=agent, agent_history=[{"role": "assistant", "content": "halo"}],
        observed_group_context=None, persist_msg="ya", persist_ts=None,
        stream_delta_cb=None, model="m", runtime_kwargs={"provider": "p", "api_key": "k"},
    )
    assert out is lane_result
    assert agent._session_messages == lane_result["messages"]


def test_try_fast_lane_sets_opencode_session_affinity_contextvar():
    """LAB-52: opencode-go lane must send x-opencode-session (else HTTP 400
    MissingSessionID). The lane sets the runtime contextvar with the session id
    and does NOT pass main_runtime down (aux normalization strips it)."""
    captured: list[dict] = []

    def fake_call_llm(**kwargs):
        from agent.auxiliary_client import _runtime_main_value

        captured.append(kwargs)
        captured.append(_runtime_main_value("session_id"))
        return iter([_chunk("sip")])

    result = try_fast_lane(
        history=[],
        user_message="ya",
        main_runtime={
            "provider": "opencode-go",
            "model": "glm-5",
            "api_key": "k",
            "base_url": "https://opencode.ai/zen/go/v1",
            "api_mode": "chat_completions",
            "session_id": "sess-affinity-lab52",
        },
        call_llm_fn=fake_call_llm,
    )
    assert result is not None
    assert len(captured) == 2
    kw = captured[0]
    assert "main_runtime" not in kw
    assert captured[1] == "sess-affinity-lab52"


def test_runner_passes_session_identity_into_fast_lane_runtime(monkeypatch):
    """Runner must put ctx.session_id into main_runtime for the lane."""
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    class _Stub:
        session_store = None

        def _adapter_for_source(self, source):
            return None

    class _Source:
        platform = Platform.TELEGRAM
        chat_id = "4242"
        chat_type = "dm"

    seen: dict = {}

    def capture_lane(**kwargs):
        seen["main_runtime"] = dict(kwargs.get("main_runtime") or {})
        return {
            "final_response": "sip",
            "messages": [{"role": "user", "content": "ya"}, {"role": "assistant", "content": "sip"}],
            "api_calls": 1, "agent_persisted": False, "completed": True, "failed": False,
        }

    ctx = TurnContext(
        source=_Source(), message="ya", session_key="telegram:4242",
        session_id="sess-affinity-lab52",
        history=[], _run_still_current=lambda: True,
        user_config={"gateway": {"telegram": {"fast_path": True, "fast_lane": {"enabled": True}}}},
        fast_path_taken="ack",
    )
    runner = TurnRunner(_Stub(), ctx)
    monkeypatch.setattr("gateway.run_turn_fast_lane.try_fast_lane", capture_lane)
    monkeypatch.setattr(
        runner, "_run_conversation_with_approval",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fallback")),
    )
    runner._try_fast_lane_or_conversation(
        agent=None, agent_history=[], observed_group_context=None,
        persist_msg="ya", persist_ts=None, stream_delta_cb=None,
        model="glm-5",
        runtime_kwargs={
            "provider": "opencode-go",
            "base_url": "https://opencode.ai/zen/go/v1",
            "api_key": "k",
            "api_mode": "chat_completions",
        },
    )
    assert seen["main_runtime"].get("session_id") == "sess-affinity-lab52"
    assert seen["main_runtime"].get("provider") == "opencode-go"


def test_lane_affinity_via_contextvar_and_no_main_runtime(monkeypatch):
    """The lane must NOT pass main_runtime down (aux normalizes it away and the
    request then misses x-opencode-session); it must set the runtime contextvar
    with the session id instead. Regression for LAB-52 400 MissingSessionID."""
    from gateway.run_turn_fast_lane import try_fast_lane

    seen: dict = {}

    def fake_call_llm(**kwargs):
        from agent.auxiliary_client import _runtime_main_value

        seen["passed_main_runtime"] = "main_runtime" in kwargs
        seen["session_id"] = _runtime_main_value("session_id")
        raise RuntimeError("stop-here")

    try:
        try_fast_lane(
            history=[], user_message="ok",
            user_config={
                "gateway": {"telegram": {"fast_lane": {
                    "enabled": True, "provider": "opencode-go", "model": "glm-5",
                    "max_messages": 6, "max_chars": 2000, "ttft_budget_ms": 8000,
                }}}
            },
            main_runtime={
                "provider": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1",
                "api_mode": "chat_completions", "session_id": "ses-1", "model": "glm-5",
            },
            call_llm_fn=fake_call_llm,
            chat_id="4242",
        )
    except RuntimeError:
        pass
    assert seen.get("passed_main_runtime") is False
    assert seen.get("session_id") == "ses-1"


def test_local_fast_lane_skips_opencode_affinity_contextvar():
    """LAB-53: local / 127.0.0.1 OpenAI-compatible lane must NOT set the
    runtime contextvar or emit x-opencode-session (zero affinity machinery).
    OpenCode path keeps affinity; this only gates non-OpenCode targets.
    """
    from agent.auxiliary_client import _runtime_main_value
    from agent.opencode_affinity import OPENCODE_SESSION_HEADER, opencode_session_headers

    seen: dict = {}

    def fake_call_llm(**kwargs):
        seen["kwargs"] = dict(kwargs)
        seen["session_id"] = _runtime_main_value("session_id")
        seen["affinity_headers"] = opencode_session_headers(
            kwargs.get("provider"), kwargs.get("base_url"), seen["session_id"] or None,
        )
        return iter([_chunk("sip")])

    result = try_fast_lane(
        history=[],
        user_message="ya",
        user_config={
            "gateway": {"telegram": {"fast_lane": {
                "enabled": True,
                "provider": "local",
                "model": "qwen-local",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "sk-local",
                "max_messages": 6,
                "max_chars": 2000,
                "ttft_budget_ms": 8000,
            }}}
        },
        main_runtime={
            "provider": "opencode-go",
            "model": "glm-5",
            "api_key": "k",
            "base_url": "https://opencode.ai/zen/go/v1",
            "api_mode": "chat_completions",
            "session_id": "sess-should-not-leak",
        },
        call_llm_fn=fake_call_llm,
    )
    assert result is not None
    assert result["final_response"] == "sip"
    assert seen["kwargs"].get("provider") == "local"
    assert seen["kwargs"].get("base_url") == "http://127.0.0.1:8080/v1"
    assert seen["kwargs"].get("api_key") == "sk-local"
    assert seen["session_id"] in (None, "")
    assert OPENCODE_SESSION_HEADER not in (seen["affinity_headers"] or {})
    assert "main_runtime" not in seen["kwargs"]
