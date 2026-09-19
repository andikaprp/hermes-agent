"""Invariant tests for optional Jev quality gate on LAB-52 fast-lane drafts.

Contracts (not snapshots):
- draft ``send`` / above threshold -> lane result unchanged
- confident ``escalate`` -> ``try_fast_lane`` returns None (fallback)
- gate disabled -> unchanged lane path (no Jev HTTP)
- Jev HTTP failure / timeout / missing key -> draft sent (fail-open)
- config parsing defaults off
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gateway.run_turn_fast_lane import try_fast_lane
from gateway.run_turn_fast_lane_quality_gate import (
    CHOICE_CRITERIA,
    CHOICE_INSTRUCTIONS,
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS,
    FastLaneQualityGateConfig,
    build_quality_gate_request,
    evaluate_fast_lane_draft,
    load_quality_gate_config,
    parse_quality_answer,
    parse_quality_gate_config,
)


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.7,
        "model": "jev-latest",
        "timeout_seconds": 1.5,
    }
    block.update(overrides)
    return {
        "gateway": {
            "telegram": {
                "fast_lane": {
                    "enabled": True,
                    "quality_gate": block,
                }
            }
        }
    }


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    other = "escalate" if choice == "send" else "send"
    return {
        "answers": {
            "quality": {
                "choice": choice,
                "confidence": confidence,
                "probabilities": {
                    choice: confidence,
                    other: 1.0 - confidence,
                },
            }
        }
    }


class _FakeHttp:
    def __init__(
        self,
        payload: Any = None,
        *,
        error: Optional[BaseException] = None,
    ):
        self.payload = (
            payload if payload is not None else _choice_response("send", 0.95)
        )
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, headers: Dict[str, Any] = None, json: Dict[str, Any] = None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.error is not None:
            raise self.error
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self.payload
        return resp

    def close(self):
        pass


class _Chunk:
    def __init__(self, text: str):
        self.choices = [
            type("C", (), {"delta": type("D", (), {"content": text})()})()
        ]


def _stream_llm(text: str = "hey there"):
    def _fn(**kwargs):
        mid = max(1, len(text) // 2)
        return iter([_Chunk(text[:mid]), _Chunk(text[mid:])])

    return _fn


def _runtime():
    return {"provider": "test", "model": "m", "api_key": "k"}


def _patch_key(monkeypatch):
    monkeypatch.setattr(
        "gateway.run_turn_fast_lane_quality_gate.resolve_typesafe_api_key",
        lambda: "test-key",
    )


class TestConfigDefaults:
    def test_parse_defaults_off(self):
        cfg = parse_quality_gate_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == DEFAULT_THRESHOLD
        assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert load_quality_gate_config(None).enabled is False
        assert load_quality_gate_config({}).enabled is False

    def test_default_config_declares_quality_gate_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        gate = DEFAULT_CONFIG["gateway"]["telegram"]["fast_lane"]["quality_gate"]
        assert gate["enabled"] is False
        assert float(gate["timeout_seconds"]) == pytest.approx(1.5)
        assert float(gate["threshold"]) >= 0.0

    def test_build_request_freezes_choice_wording(self):
        body = build_quality_gate_request("hi", "yo")
        q = body["questions"]["quality"]
        assert q["type"] == "choice"
        assert q["instructions"] == CHOICE_INSTRUCTIONS
        assert q["criteria"] == CHOICE_CRITERIA
        assert any("User message:" in s for s in body["state"])
        assert any("Draft reply:" in s for s in body["state"])


class TestEvaluateFailOpen:
    def test_disabled_never_calls_http(self):
        http = _FakeHttp()
        assert (
            evaluate_fast_lane_draft(
                "hi", "yo", user_config={}, http_client=http, api_key="k",
            )
            == "send"
        )
        assert http.calls == []

    def test_missing_key_sends_without_http(self):
        http = _FakeHttp()
        assert (
            evaluate_fast_lane_draft(
                "hi",
                "yo",
                user_config=_enabled_cfg(),
                http_client=http,
                api_key="",
            )
            == "send"
        )
        assert http.calls == []

    def test_http_failure_sends(self):
        http = _FakeHttp(error=RuntimeError("jev http 500"))
        assert (
            evaluate_fast_lane_draft(
                "hi",
                "yo",
                user_config=_enabled_cfg(),
                http_client=http,
                api_key="k",
            )
            == "send"
        )

    def test_timeout_sends(self):
        http = _FakeHttp(error=TimeoutError("timed out waiting for jev"))
        assert (
            evaluate_fast_lane_draft(
                "hi",
                "yo",
                user_config=_enabled_cfg(),
                http_client=http,
                api_key="k",
            )
            == "send"
        )

    def test_over_budget_ready_ms_sends(self, monkeypatch):
        cfg = FastLaneQualityGateConfig(
            enabled=True, threshold=0.7, timeout_seconds=1.5,
        )

        def _slow(*_a, **_k):
            return _choice_response("escalate", 0.99), 5.0, 2500.0

        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            _slow,
        )
        assert (
            evaluate_fast_lane_draft("hi", "yo", cfg=cfg, api_key="k")
            == "send"
        )


class TestEvaluateDecisions:
    def test_send_above_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            lambda *_a, **_k: (_choice_response("send", 0.95), 1.0, 2.0),
        )
        assert (
            evaluate_fast_lane_draft(
                "hi", "yo", user_config=_enabled_cfg(), api_key="k",
            )
            == "send"
        )

    def test_escalate_above_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            lambda *_a, **_k: (_choice_response("escalate", 0.92), 1.0, 2.0),
        )
        assert (
            evaluate_fast_lane_draft(
                "hi", "garbage", user_config=_enabled_cfg(), api_key="k",
            )
            == "escalate"
        )

    def test_escalate_below_threshold_fail_open(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            lambda *_a, **_k: (_choice_response("escalate", 0.40), 1.0, 2.0),
        )
        assert (
            evaluate_fast_lane_draft(
                "hi", "meh", user_config=_enabled_cfg(threshold=0.7), api_key="k",
            )
            == "send"
        )

    def test_parse_quality_answer(self):
        choice, conf = parse_quality_answer(_choice_response("send", 0.88))
        assert choice == "send"
        assert conf == pytest.approx(0.88)


class TestTryFastLaneIntegration:
    def test_gate_disabled_unchanged(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("Jev must not run when gate disabled")

        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            _boom,
        )
        result = try_fast_lane(
            history=[],
            user_message="hi",
            user_config={"gateway": {"telegram": {"fast_lane": {"enabled": True}}}},
            main_runtime=_runtime(),
            call_llm_fn=_stream_llm("hello"),
        )
        assert result is not None
        assert result["final_response"] == "hello"
        assert called["n"] == 0

    def test_send_keeps_draft(self, monkeypatch):
        _patch_key(monkeypatch)
        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            lambda *_a, **_k: (_choice_response("send", 0.99), 1.0, 2.0),
        )
        deltas: List[str] = []
        result = try_fast_lane(
            history=[],
            user_message="hi",
            user_config=_enabled_cfg(),
            main_runtime=_runtime(),
            on_delta=lambda t: deltas.append(t) if t else None,
            call_llm_fn=_stream_llm("warm hi"),
        )
        assert result is not None
        assert result["final_response"] == "warm hi"
        assert "".join(deltas) == "warm hi"

    def test_escalate_returns_none(self, monkeypatch):
        _patch_key(monkeypatch)
        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            lambda *_a, **_k: (_choice_response("escalate", 0.99), 1.0, 2.0),
        )
        deltas: List[str] = []
        result = try_fast_lane(
            history=[],
            user_message="hi",
            user_config=_enabled_cfg(),
            main_runtime=_runtime(),
            on_delta=lambda t: deltas.append(t) if t else None,
            call_llm_fn=_stream_llm("totally wrong answer about taxes"),
        )
        assert result is None
        assert deltas == []

    def test_jev_failure_still_sends_draft(self, monkeypatch):
        _patch_key(monkeypatch)

        def _fail(*_a, **_k):
            raise RuntimeError("jev http 503")

        monkeypatch.setattr(
            "gateway.run_turn_fast_lane_quality_gate._post_systemone",
            _fail,
        )
        result = try_fast_lane(
            history=[],
            user_message="hi",
            user_config=_enabled_cfg(),
            main_runtime=_runtime(),
            call_llm_fn=_stream_llm("hey"),
        )
        assert result is not None
        assert result["final_response"] == "hey"
