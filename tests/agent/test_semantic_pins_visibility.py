"""Ningning-only, never Rancana

Config-gated semantic pins and the query-aware visibility ladder.

These tests are written against the new API. On the base commit they fail
(missing module, missing attribute, or the old behaviour). On this branch
they pass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_compressor_jev import JevScorerConfig
from agent.semantic_pins import (
    PIN_CLASSES,
    SemanticPinsConfig,
    parse_semantic_pins_config,
    pin_survives,
    redact_message_for_compaction,
)
from agent.visibility_ladder import (
    SCORING_MODES,
    VisibilityLadderConfig,
    VisibilityLedger,
    parse_visibility_ladder_config,
    prepare_compressible_window,
    render_level,
    score_visibility,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG, OPTIONAL_ENV_VARS
from tests.agent.test_compaction_redaction_boundaries import (
    OAUTH_URL,
    SECRET,
    _assert_clean,
)


PIN_PHRASE = "PIN-VERBATIM-PHRASE current failing test"
CONTROL_PHRASE = "CONTROL-SHOULD-DROP old plan"


def _compressor(**kwargs) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.5,
            protect_first_n=kwargs.pop("protect_first_n", 0),
            protect_last_n=kwargs.pop("protect_last_n", 1),
            quiet_mode=True,
            config_context_length=100_000,
            **kwargs,
        )
    compressor.context_length = 100_000
    compressor.threshold_tokens = 1_000
    compressor._tail_token_budget = 20
    return compressor


def _middle_transcript() -> list[dict]:
    """Pin and control sit in the middle, outside protect_first_n and protect_last_n."""
    messages: list[dict] = [{"role": "system", "content": "system prompt stays"}]
    messages.append(
        {
            "role": "user",
            "content": PIN_PHRASE,
            "semantic_pin": "active_task",
        }
    )
    messages.append({"role": "assistant", "content": CONTROL_PHRASE})
    for index in range(6):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"middle filler {index}"})
    messages.append({"role": "assistant", "content": "tail assistant reply"})
    messages.append({"role": "user", "content": "latest user question"})
    return messages


def _joined(messages) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(str(content))
        calls = msg.get("tool_calls")
        if calls:
            parts.append(str(calls))
    return "\n".join(parts)


def _compress(compressor: ContextCompressor, messages: list[dict], **kwargs):
    with patch.object(
        compressor,
        "_generate_summary",
        return_value="summary of older work only",
    ):
        return compressor.compress(messages, current_tokens=90_000, force=True, **kwargs)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload=None, error=None, status_code=200):
        self.payload = payload or {}
        self.error = error
        self.status_code = status_code
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return _Response(self.payload, self.status_code)


def _answers_for(body, choice="short", confidence=0.95, pin="none"):
    state = body["state"]
    answers = {}
    for index in range(len(state)):
        answers[f"v{index}"] = {"choice": choice, "confidence": confidence}
        answers[f"p{index}"] = {"choice": pin, "confidence": confidence}
    return {"answers": answers}


def test_semantic_pins_config_gate_defaults_off_and_closed():
    block = DEFAULT_CONFIG["compression"]["semantic_pins"]
    assert block["enabled"] is False
    assert block["classes"] == list(PIN_CLASSES)
    parsed = parse_semantic_pins_config(None)
    assert parsed.enabled is False
    assert parsed.classes == PIN_CLASSES
    narrowed = parse_semantic_pins_config(
        {"enabled": True, "classes": ["active_task", "steal_secrets", "standing_instruction"]}
    )
    assert narrowed.classes == ("active_task", "standing_instruction")
    assert "steal_secrets" not in narrowed.classes
    assert "HERMES_" not in open(
        "agent/semantic_pins.py", encoding="utf-8"
    ).read()


def test_no_new_hermes_environment_variables():
    banned = [
        key for key in OPTIONAL_ENV_VARS
        if "SEMANTIC_PIN" in key or "VISIBILITY_LADDER" in key or "CACHE_REUSE" in key
    ]
    assert banned == []
    assert "HERMES_" not in open("agent/visibility_ladder.py", encoding="utf-8").read()


def test_pinned_span_survives_independent_of_position_and_jev_score():
    assert pin_survives(
        index=5, n=10, score=0.0, keep_threshold=1.2,
        protect_first_n=0, protect_last_n=1, pinned=True,
    )
    assert not pin_survives(
        index=5, n=10, score=0.0, keep_threshold=1.2,
        protect_first_n=0, protect_last_n=1, pinned=False,
    )
    # Index 5 is outside a head of 0 and a tail of 1 (only index 9).
    assert 5 >= 0
    assert 5 < 10 - 1

    compressor = _compressor(
        protect_first_n=0,
        protect_last_n=1,
        semantic_pins=SemanticPinsConfig(enabled=True),
        jev_scorer=JevScorerConfig(enabled=True, keep_threshold=1.2),
    )

    def _stub_everything(messages, **kwargs):
        from agent.context_compressor_jev import JevThinResult

        stubbed = [
            {"role": msg.get("role") or "assistant", "content": "STUBBED-BY-JEV"}
            for msg in messages
        ]
        return JevThinResult(
            messages=stubbed, n_scored=len(messages), n_kept=0, fallback=False,
        )

    messages = _middle_transcript()
    with patch(
        "agent.context_compressor_jev.thin_compressible_window",
        side_effect=_stub_everything,
    ):
        out = _compress(compressor, messages)
    joined = _joined(out)
    assert PIN_PHRASE in joined
    assert CONTROL_PHRASE not in joined
    assert compressor.protect_first_n == 0
    assert compressor.protect_last_n == 1


def test_pinning_cannot_reintroduce_redaction_boundary_spans(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    shapes = [
        {"role": "user", "content": f"token {SECRET} url {OAUTH_URL} keep-phrase", "semantic_pin": "goal_scaffold"},
        {
            "role": "assistant",
            "content": None,
            "semantic_pin": "active_task",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "terminal",
                        "arguments": f'{{"command": "curl {OAUTH_URL}", "note": "{SECRET}"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "semantic_pin": "standing_instruction",
            "content": f"got {SECRET} via {OAUTH_URL}",
        },
        {
            "role": "assistant",
            "semantic_pin": "goal_scaffold",
            "content": f"fallback summary leaked {SECRET} and {OAUTH_URL}",
        },
    ]
    redacted = [redact_message_for_compaction(msg) for msg in shapes]
    joined = _joined(redacted)
    _assert_clean(joined)
    assert "keep-phrase" in joined

    compressor = _compressor(
        protect_first_n=0,
        protect_last_n=1,
        semantic_pins=SemanticPinsConfig(enabled=True),
    )
    compressor._previous_summary = f"Old summary leaked {SECRET} and {OAUTH_URL}"
    messages = _middle_transcript()
    messages[1]["content"] = f"{PIN_PHRASE} token {SECRET} url {OAUTH_URL}"
    out = _compress(
        compressor,
        messages,
        focus_topic=f"focus {SECRET} {OAUTH_URL}",
    )
    body = _joined(out)
    assert SECRET not in body
    assert "opaque-code-123" not in body
    assert "access_token=opaque-token-456" not in body
    assert PIN_PHRASE in body


def test_visibility_ladder_gate_monotonic_and_not_deleted():
    block = DEFAULT_CONFIG["compression"]["visibility_ladder"]
    assert block["enabled"] is False
    assert block["scoring"] == "slow_path_only"
    assert set(SCORING_MODES) == {"slow_path_only", "async_precompute"}
    parsed = parse_visibility_ladder_config(None)
    assert parsed.enabled is False
    assert parsed.scoring == "slow_path_only"
    unknown = parse_visibility_ladder_config({"enabled": True, "scoring": "hot_path"})
    assert unknown.scoring == "slow_path_only"

    samples = ["hi", "word " * 40, "hit\n" * 2400]
    for text in samples:
        lengths = []
        for level in ("hidden", "short", "long", "full"):
            rendered = render_level(text, level)
            lengths.append(0 if rendered is None else len(rendered))
        assert lengths == sorted(lengths), (text[:20], lengths)
        assert lengths[0] == 0
        assert render_level(text, "hidden") is None
        assert render_level(text, "full") == text
        short = render_level(text, "short")
        assert short is not None and "\n" not in short

    text = "grep-hit\n" * 2400
    msg = {
        "role": "tool",
        "content": text,
        "tool_call_id": "grep-2400",
        "visibility_id": "grep-2400",
    }
    ledger = VisibilityLedger()
    ledger.observe([msg])
    before = ledger.count()
    ledger.set_level("grep-2400", "hidden")
    assert ledger.project() == []
    assert ledger.count() == before
    assert ledger.recover("grep-2400")["content"] == text
    ledger.set_level("grep-2400", "short")
    sent = ledger.project()
    assert len(sent) == 1
    assert 0 < len(sent[0]["content"]) <= len(text)
    assert ledger.recover("grep-2400")["content"] == text
    assert not hasattr(ledger, "delete")


def test_visibility_scoring_is_off_the_fast_lane_hot_path(monkeypatch):
    posts = []

    def _boom(*args, **kwargs):
        posts.append(1)
        raise AssertionError("fast lane posted to System One")

    monkeypatch.setattr("agent.context_compressor_jev._post_systemone", _boom)
    from gateway.run_turn_fast_lane import FAST_LANE_SYSTEM, build_compact_transcript

    out = build_compact_transcript(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
        "hello",
    )
    assert posts == []
    assert out[0] == {"role": "system", "content": FAST_LANE_SYSTEM}
    assert out[-1]["content"] == "hello"

    monkeypatch.setattr(
        "agent.visibility_ladder._on_fast_lane_hot_path",
        lambda: True,
    )
    monkeypatch.setattr(
        "agent.context_compressor_jev.resolve_typesafe_api_key",
        lambda: "test-key",
    )
    turns = [{"role": "user", "content": "bulk grep"}]
    result = prepare_compressible_window(
        turns,
        pins=SemanticPinsConfig(enabled=False),
        ladder=VisibilityLadderConfig(enabled=True, scoring="slow_path_only"),
        question="what failed?",
        http_client=_Client(error=AssertionError("should not post")),
    )
    assert result.fallback is True
    assert result.reason == "fast_lane_hot_path"
    assert result.turns is turns
    assert result.pinned_verbatim == []


def test_visibility_request_is_query_aware_and_slow_path(monkeypatch):
    monkeypatch.setattr(
        "agent.context_compressor_jev.resolve_typesafe_api_key",
        lambda: "test-key",
    )
    client = _Client()

    def post(url, json=None, headers=None, timeout=None):
        client.calls.append(json)
        return _Response(_answers_for(json, choice="long", confidence=0.9))

    client.post = post
    turns = [{"role": "tool", "content": "x" * 2000, "tool_call_id": "g1"}]
    levels, pins, reason = score_visibility(
        turns,
        question="show the failing test in current file",
        ladder=VisibilityLadderConfig(enabled=True, scoring="slow_path_only"),
        want_pins=False,
        http_client=client,
    )
    assert reason == ""
    assert levels == ["long"]
    assert pins == [None]
    instructions = client.calls[0]["questions"]["v0"]["instructions"]
    assert "failing test" in instructions
    assert client.calls[0]["questions"]["v0"]["type"] == "choice"


def test_async_precompute_still_scores_off_the_caller_stack(monkeypatch):
    monkeypatch.setattr(
        "agent.context_compressor_jev.resolve_typesafe_api_key",
        lambda: "test-key",
    )
    seen = {}

    class Client:
        def post(self, url, json=None, headers=None, timeout=None):
            seen["thread"] = __import__("threading").current_thread().name
            return _Response(_answers_for(json, choice="full", confidence=0.99))

    levels, _, reason = score_visibility(
        [{"role": "user", "content": "keep me"}],
        question="current goal",
        ladder=VisibilityLadderConfig(enabled=True, scoring="async_precompute"),
        want_pins=False,
        http_client=Client(),
    )
    assert reason == ""
    assert levels == ["full"]
    assert seen["thread"].startswith("lab52-vis")


@pytest.mark.parametrize("kind", ["error", "429", "timeout", "missing_key"])
def test_hard_fallback_on_jev_failure(kind, monkeypatch):
    turns = [{"role": "user", "content": "original span", "semantic_pin": "active_task"}]
    if kind == "missing_key":
        monkeypatch.setattr(
            "agent.context_compressor_jev.resolve_typesafe_api_key",
            lambda: "",
        )
        client = _Client(error=AssertionError("posted without a key"))
    else:
        monkeypatch.setattr(
            "agent.context_compressor_jev.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        if kind == "error":
            client = _Client(error=RuntimeError("boom"))
        elif kind == "429":
            client = _Client(status_code=429, payload={})
        else:
            client = _Client(error=TimeoutError("request timeout"))
    result = prepare_compressible_window(
        turns,
        pins=SemanticPinsConfig(enabled=True),
        ladder=VisibilityLadderConfig(enabled=True, scoring="slow_path_only"),
        question="current question",
        http_client=client,
    )
    assert result.fallback is True
    assert result.reason == kind
    assert result.turns is turns
    assert result.pinned_verbatim == []
    if kind == "missing_key":
        assert client.calls == []


def test_below_threshold_counts_as_success_not_failure(caplog):
    from agent.context_compressor_jev import is_jev_scorer_success, log_jev_scorer

    assert is_jev_scorer_success(fallback=True, reason="below_threshold") is True
    assert is_jev_scorer_success(fallback=True, reason="timeout") is False
    assert is_jev_scorer_success(fallback=True, reason="429") is False
    assert is_jev_scorer_success(fallback=True, reason="missing_key") is False
    assert is_jev_scorer_success(fallback=True, reason="error") is False

    caplog.set_level("INFO")
    log_jev_scorer(
        n_scored=3,
        n_kept=1,
        n_pinned=2,
        level_chars={"hidden": 0, "short": 12, "long": 80, "full": 400},
        ttft_ms=1.2,
        ready_ms=3.4,
        fallback=True,
        reason="below_threshold",
    )
    text = caplog.text
    assert "n_pinned=2" in text
    assert "level_hidden=0" in text
    assert "level_short=12" in text
    assert "level_long=80" in text
    assert "level_full=400" in text
    assert "fallback=true" in text
    assert "reason=below_threshold" in text

    turns = [{"role": "user", "content": "keep-original"}]
    with patch(
        "agent.context_compressor_jev.resolve_typesafe_api_key",
        return_value="test-key",
    ):
        client = _Client()

        def post(url, json=None, headers=None, timeout=None):
            client.calls.append(json)
            return _Response(_answers_for(json, choice="hidden", confidence=0.1))

        client.post = post
        result = prepare_compressible_window(
            turns,
            pins=SemanticPinsConfig(enabled=False),
            ladder=VisibilityLadderConfig(enabled=True),
            question="unrelated question",
            http_client=client,
        )
    assert result.fallback is True
    assert result.reason == "below_threshold"
    assert result.turns is turns
    assert is_jev_scorer_success(fallback=True, reason=result.reason) is True


def test_cache_reuse_decision_stays_false():
    from agent.context_compressor_jev import (
        CACHE_REUSE_DECISION,
        CACHE_REUSE_DECISION_REASON,
        parse_cache_reuse_decision,
    )

    assert CACHE_REUSE_DECISION is False
    assert parse_cache_reuse_decision(True) is False
    assert parse_cache_reuse_decision("yes") is False
    assert 'never mutate the cached prefix' in CACHE_REUSE_DECISION_REASON
    assert "sacred" in CACHE_REUSE_DECISION_REASON
    assert "99" in CACHE_REUSE_DECISION_REASON
    assert DEFAULT_CONFIG["compression"]["cache_reuse_decision"] is False
    compressor = _compressor(cache_reuse_decision=True)
    assert compressor.cache_reuse_decision is False


def test_new_keys_are_init_consumed_not_read_fresh():
    from agent.agent_init import _parse_compression_config
    from agent.semantic_pins import CONFIG_CLASSIFICATION, RELOAD_BOUNDARY
    from gateway.run import GatewayRunner

    cfg = {
        "semantic_pins": {"enabled": False, "classes": ["active_task"]},
        "visibility_ladder": {"enabled": False, "scoring": "slow_path_only"},
        "cache_reuse_decision": True,
    }
    agent = SimpleNamespace(model="test-model", provider="test")
    parsed = _parse_compression_config(agent, {"compression": cfg})
    assert parsed.semantic_pins.enabled is False
    assert parsed.visibility_ladder.enabled is False
    assert parsed.visibility_ladder.scoring == "slow_path_only"
    assert parsed.cache_reuse_decision is False

    cfg["semantic_pins"]["enabled"] = True
    cfg["visibility_ladder"]["enabled"] = True
    cfg["visibility_ladder"]["scoring"] = "async_precompute"
    assert parsed.semantic_pins.enabled is False
    assert parsed.visibility_ladder.scoring == "slow_path_only"

    fresh = _parse_compression_config(agent, {"compression": cfg})
    assert fresh.semantic_pins.enabled is True
    assert fresh.visibility_ladder.scoring == "async_precompute"
    assert fresh.cache_reuse_decision is False

    for key in (
        "compression.semantic_pins.enabled",
        "compression.semantic_pins.classes",
        "compression.visibility_ladder.enabled",
        "compression.visibility_ladder.scoring",
        "compression.cache_reuse_decision",
    ):
        assert CONFIG_CLASSIFICATION[key] == "init-consumed"
    assert "init-consumed" in RELOAD_BOUNDARY
    assert "compression.threshold" in RELOAD_BOUNDARY
    pairs = set(GatewayRunner._CACHE_BUSTING_CONFIG_KEYS)
    assert ("compression", "semantic_pins") not in pairs
    assert ("compression", "visibility_ladder") not in pairs
    assert ("compression", "cache_reuse_decision") not in pairs

    docs = open(
        "website/docs/developer-guide/context-compression-and-caching.md",
        encoding="utf-8",
    ).read()
    assert "init-consumed" in docs
    assert "compression.semantic_pins" in docs
    assert "does **not** resolve the existing conflict" in docs


def test_tui_live_apply_does_not_adopt_init_consumed_keys(monkeypatch):
    from tui_gateway import server

    compressor = _compressor()
    assert compressor.semantic_pins.enabled is False
    session = {
        "agent": SimpleNamespace(
            model="test/model",
            provider="test",
            context_compressor=compressor,
            compression_enabled=True,
            compression_idle_compact_after_seconds=0,
            codex_responses_native_compaction=False,
            codex_responses_compact_threshold=200_000,
        ),
        "session_key": "lab52",
    }
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"default": "test/model", "provider": "test"},
            "compression": {
                "semantic_pins": {"enabled": True},
                "visibility_ladder": {"enabled": True, "scoring": "async_precompute"},
                "cache_reuse_decision": True,
            },
        },
    )
    server._sync_agent_compression_with_config("lab52", session)
    assert compressor.semantic_pins.enabled is False
    assert compressor.visibility_ladder.enabled is False
    assert compressor.visibility_ladder.scoring == "slow_path_only"
    assert compressor.cache_reuse_decision is False


def test_default_off_does_not_call_jev(monkeypatch):
    posts = []
    monkeypatch.setattr(
        "agent.context_compressor_jev._post_systemone",
        lambda *a, **k: posts.append(1),
    )
    compressor = _compressor()
    assert compressor.semantic_pins.enabled is False
    assert compressor.visibility_ladder.enabled is False
    out = _compress(compressor, _middle_transcript())
    assert posts == []
    assert PIN_PHRASE not in _joined(out)
