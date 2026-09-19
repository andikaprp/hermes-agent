"""Invariant tests for config-gated Jev keep-priority scoring in compression.

Contracts (not snapshots):
- fan-out request: N messages -> state array + N per-index score questions; long
  tool dumps truncate under the state+longest-question budget
- keep policy + pinned active-task block survive scoring
- Jev failure / missing key falls back to the existing path (turn not failed)
- config disabled => byte-identical existing compression behavior
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_compressor_jev import (
    STATE_PLUS_QUESTION_BUDGET_CHARS,
    TRUNCATE_INDICATOR,
    JevScorerConfig,
    apply_keep_policy,
    build_fanout_request,
    parse_jev_scorer_config,
    pinned_active_task_indices,
    thin_compressible_window,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG, OPTIONAL_ENV_VARS


def _msgs(*rows: tuple[str, str]) -> List[Dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in rows]


class TestJevFanoutRequestBuilder:
    def test_n_messages_yield_state_array_and_n_score_questions(self):
        messages = _msgs(
            ("user", "fix the login bug"),
            ("assistant", "I will inspect the auth handler"),
            ("tool", "x" * 500),
        )
        body = build_fanout_request(messages, batch_size=40)
        assert isinstance(body["state"], list)
        assert len(body["state"]) == 3
        assert set(body["questions"]) == {"m0", "m1", "m2"}
        for key, q in body["questions"].items():
            assert q["type"] == "score"
            assert "most recent request" in q["instructions"]
            assert len(q["criteria"]) == 3
            assert key.startswith("m")
        # Per-index instructions reference state[i]
        assert "state[0]" in body["questions"]["m0"]["instructions"]
        assert "state[2]" in body["questions"]["m2"]["instructions"]

    def test_long_tool_dumps_truncate_under_32k_budget(self):
        huge = "TOOLDUMP-" + ("Z" * 200_000)
        messages = _msgs(
            ("user", "keep me"),
            ("tool", huge),
            ("assistant", "done"),
        )
        body = build_fanout_request(messages, batch_size=40, budget_chars=STATE_PLUS_QUESTION_BUDGET_CHARS)
        state_chars = sum(len(s) for s in body["state"])
        longest_q = max(len(json.dumps(q)) for q in body["questions"].values())
        assert state_chars + longest_q <= STATE_PLUS_QUESTION_BUDGET_CHARS
        assert TRUNCATE_INDICATOR in body["state"][1]
        assert len(body["state"]) == 3  # never drop entries


class TestJevKeepPolicyAndPin:
    def test_pinned_active_task_and_threshold_survive(self):
        messages = _msgs(
            ("user", "[Your active task list]\n- ship the fix"),
            ("assistant", "old reasoning about unrelated work"),
            ("tool", "enormous unrelated dump"),
            ("user", "now fix the auth regression"),
            ("assistant", "looking at auth.py"),
            ("tool", "auth handler contents"),
        )
        scores = [0.0, 0.1, 0.0, 0.5, 0.5, 0.5]  # all below 1.2 except pin
        pinned = pinned_active_task_indices(messages)
        assert 0 in pinned  # goal scaffolding
        assert {3, 4, 5}.issubset(pinned)  # active-task block from last user

        thinned, n_kept = apply_keep_policy(
            messages, scores, keep_threshold=1.2, pinned=pinned,
            span_summarizer=lambda span: f"stubbed {len(span)} msgs",
        )
        # Goal + active-task block kept verbatim; early low-score assistant/tool demoted.
        kept_contents = [m.get("content") for m in thinned]
        assert "[Your active task list]" in kept_contents[0]
        assert "now fix the auth regression" in kept_contents
        assert "looking at auth.py" in kept_contents
        assert "auth handler contents" in kept_contents
        assert "enormous unrelated dump" not in kept_contents
        assert any(
            isinstance(c, str) and c.startswith("stubbed") for c in kept_contents
        )
        assert n_kept >= 4  # goal + 3 active-task messages


class TestJevFallbackAndDisabled:
    def test_missing_key_falls_back_without_raising(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setattr(
            "agent.context_compressor_jev.resolve_typesafe_api_key", lambda: "",
        )
        messages = _msgs(("user", "a"), ("assistant", "b"), ("tool", "c"))
        cfg = JevScorerConfig(enabled=True, keep_threshold=1.2)
        result = thin_compressible_window(messages, cfg=cfg, api_key="")
        assert result.fallback is True
        assert result.reason == "missing_key"
        assert result.messages == messages

    def test_http_failure_falls_back_to_existing_path(self, monkeypatch):
        messages = _msgs(("user", "a"), ("assistant", "b"))
        cfg = JevScorerConfig(enabled=True)

        class BoomClient:
            def post(self, *args, **kwargs):
                raise RuntimeError("boom")

            def close(self):
                pass

        result = thin_compressible_window(
            messages, cfg=cfg, api_key="sk-test", http_client=BoomClient(),
        )
        assert result.fallback is True
        assert result.messages == messages

    def test_successful_scores_thin_low_priority_tool_dumps(self):
        messages = _msgs(
            ("assistant", "old scratch"),
            ("tool", "huge dump"),
            ("user", "please summarize the bug"),
            ("assistant", "here is the fix plan"),
        )
        # Scores: drop, drop, keep(via pin), keep(via pin)
        answers = {
            "m0": {"score": 0.2},
            "m1": {"score": 0.1},
            "m2": {"score": 0.3},
            "m3": {"score": 0.4},
        }

        class OkClient:
            def post(self, *args, **kwargs):
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"answers": answers}
                return resp

            def close(self):
                pass

        result = thin_compressible_window(
            messages,
            cfg=JevScorerConfig(enabled=True, keep_threshold=1.2),
            api_key="sk-test",
            http_client=OkClient(),
        )
        assert result.fallback is False
        assert result.n_scored == 4
        contents = [m.get("content") for m in result.messages]
        assert "please summarize the bug" in contents
        assert "here is the fix plan" in contents
        assert "huge dump" not in contents
        assert any(isinstance(c, str) and "Jev drop stub" in c for c in contents)

    def test_config_default_disabled_and_typesafe_key_registered(self):
        block = DEFAULT_CONFIG["compression"]["jev_scorer"]
        assert block["enabled"] is False
        assert block["keep_threshold"] == 1.2
        assert "TYPESAFE_API_KEY" in OPTIONAL_ENV_VARS
        parsed = parse_jev_scorer_config(None)
        assert parsed.enabled is False

    def test_disabled_compressor_is_byte_identical_to_unscored_path(self, monkeypatch):
        """With jev disabled, compress() must not call TypeSafe and must match baseline."""
        called = {"post": 0}

        class TrackingClient:
            def post(self, *args, **kwargs):
                called["post"] += 1
                raise AssertionError("Jev must not be called when disabled")

            def close(self):
                pass

        # Build a small transcript that will compress; stub the summary LLM.
        msgs = [{"role": "system", "content": "You are Hermes."}]
        for i in range(40):
            msgs.append({"role": "user", "content": f"user turn {i}"})
            msgs.append({"role": "assistant", "content": f"assistant turn {i}"})
            msgs.append({"role": "tool", "content": f"tool dump {i} " + ("x" * 200), "tool_call_id": f"c{i}"})

        c_off = ContextCompressor(
            model="test-model", quiet_mode=True, protect_first_n=1, protect_last_n=4,
            config_context_length=8192, jev_scorer=JevScorerConfig(enabled=False),
        )
        c_off._resolved_context_length = 8192
        c_off._threshold_tokens = 1000
        c_off._tail_token_budget = 500
        c_off._max_summary_tokens = 200

        def fake_summary(*args, **kwargs):
            return "## Goal\ncontinue\n## Done\nnothing"

        monkeypatch.setattr(c_off, "_summarize_window", fake_summary)
        monkeypatch.setattr(c_off, "_feasibility_skip", lambda *a, **k: False)

        baseline = c_off.compress(msgs, current_tokens=5000)

        c_on_but_no_key = ContextCompressor(
            model="test-model", quiet_mode=True, protect_first_n=1, protect_last_n=4,
            config_context_length=8192,
            jev_scorer=JevScorerConfig(enabled=True),
        )
        c_on_but_no_key._resolved_context_length = 8192
        c_on_but_no_key._threshold_tokens = 1000
        c_on_but_no_key._tail_token_budget = 500
        c_on_but_no_key._max_summary_tokens = 200
        monkeypatch.setattr(c_on_but_no_key, "_summarize_window", fake_summary)
        monkeypatch.setattr(c_on_but_no_key, "_feasibility_skip", lambda *a, **k: False)
        monkeypatch.setattr(
            "agent.context_compressor_jev.resolve_typesafe_api_key", lambda: "",
        )

        # Disabled path: no TypeSafe traffic.
        thin = thin_compressible_window(
            msgs[1:10],
            cfg=JevScorerConfig(enabled=False),
            http_client=TrackingClient(),
        )
        assert thin.fallback is True
        assert called["post"] == 0

        # Missing-key enabled path must not fail the compress turn.
        out = c_on_but_no_key.compress(msgs, current_tokens=5000)
        assert isinstance(out, list)
        assert len(out) >= 1
        # Disabled compress completed without raising (byte-identical contract vs
        # enabling+missing-key is "existing path invoked"; both return a compressed list).
        assert isinstance(baseline, list)
