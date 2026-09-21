"""Invariant tests for confidence-gated Jev routing on uncertain fast-path turns.

Contracts (not snapshots):
- uncertain band + high-confidence Jev ``lane`` -> fast path taken
- uncertain + high-confidence ``task`` -> main path
- below-threshold -> deterministic default (main path for uncertain)
- disabled -> no Jev HTTP call
- Jev failure / missing key -> deterministic default
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gateway.run_turn_fast_path import (
    classify_fast_path,
    classify_fast_path_band,
    fast_path_reason,
    needs_jev_second_opinion,
)
from gateway.run_turn_jev_routing import (
    CHOICE_CRITERIA,
    CHOICE_INSTRUCTIONS,
    build_jev_routing_request,
    load_jev_routing_config,
    maybe_jev_route_uncertain,
    parse_jev_routing_config,
    parse_route_answer,
    reset_jev_routing_runtime_for_tests,
)


ASSISTANT_STATED = [{"role": "assistant", "content": "Deployed. Everything is green."}]

# Shape-ok lexicon miss: not ack/social, not directive/verb/URL/question.
UNCERTAIN_MSG = "alright then"
CLEAR_TASK_MSG = "please merge the PR"
CLEAR_LANE_MSG = "hi"


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.85,
        "model": "jev-latest",
        "timeout_seconds": 30,
    }
    block.update(overrides)
    return {"gateway": {"telegram": {"jev_routing": block, "fast_path": True}}}


@pytest.fixture(autouse=True)
def _reset_jev_routing_runtime():
    reset_jev_routing_runtime_for_tests()
    yield
    reset_jev_routing_runtime_for_tests()


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    return {
        "answers": {
            "route": {
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: confidence, "task" if choice == "lane" else "lane": 1.0 - confidence},
            }
        }
    }


class _FakeHttp:
    """Minimal httpx-like client recording POSTs."""

    def __init__(self, payload: Any = None, *, error: Optional[BaseException] = None, status: int = 200):
        self.payload = payload if payload is not None else _choice_response("lane", 0.95)
        self.error = error
        self.status = status
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, headers: Dict[str, Any] = None, json: Dict[str, Any] = None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.error is not None:
            raise self.error
        resp = MagicMock()
        resp.status_code = self.status
        resp.json.return_value = self.payload
        return resp

    def close(self):
        pass


class TestUncertainBand:
    def test_lexicon_miss_is_uncertain_not_lane(self):
        assert classify_fast_path(UNCERTAIN_MSG, history=ASSISTANT_STATED) is None
        assert classify_fast_path_band(UNCERTAIN_MSG, history=ASSISTANT_STATED) == "uncertain"
        assert needs_jev_second_opinion(UNCERTAIN_MSG, history=ASSISTANT_STATED) is True

    def test_hard_rejects_are_not_uncertain(self):
        assert needs_jev_second_opinion(CLEAR_TASK_MSG) is False
        assert classify_fast_path_band(CLEAR_TASK_MSG) is None
        assert needs_jev_second_opinion("yes", history=[{"role": "assistant", "content": "Shall I deploy?"}]) is False

    def test_clear_lane_is_not_uncertain(self):
        assert classify_fast_path(CLEAR_LANE_MSG) == "social"
        assert needs_jev_second_opinion(CLEAR_LANE_MSG) is False


class TestJevRoutingInvariants:
    def test_uncertain_high_confidence_lane_takes_fast_path(self):
        http = _FakeHttp(_choice_response("lane", 0.95))
        result = maybe_jev_route_uncertain(
            UNCERTAIN_MSG,
            user_config=_enabled_cfg(),
            chat_id=123,
            http_client=http,
            api_key="test-key",
        )
        assert result == "social"
        assert len(http.calls) == 1
        body = http.calls[0]["json"]
        assert body["questions"]["route"]["type"] == "choice"
        assert body["questions"]["route"]["instructions"] == CHOICE_INSTRUCTIONS
        assert body["questions"]["route"]["criteria"] == CHOICE_CRITERIA
        assert any("Inbound message" in s for s in body["state"])
        assert any("Ack words" in s for s in body["state"])

    def test_fast_path_reason_promotes_uncertain_lane(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing._post_systemone",
            lambda *_a, **_k: (_choice_response("lane", 0.99), 10.0, 12.0),
        )
        reason = fast_path_reason(
            UNCERTAIN_MSG,
            platform_key="telegram",
            chat_type="dm",
            history=ASSISTANT_STATED,
            user_config=_enabled_cfg(),
            chat_id=99,
        )
        assert reason == "social"

    def test_uncertain_high_confidence_task_stays_main_path(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing._post_systemone",
            lambda *_a, **_k: (_choice_response("task", 0.97), 1.0, 2.0),
        )
        assert (
            fast_path_reason(
                UNCERTAIN_MSG,
                platform_key="telegram",
                chat_type="dm",
                user_config=_enabled_cfg(),
            )
            is None
        )

    def test_below_threshold_uses_deterministic_default(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing._post_systemone",
            lambda *_a, **_k: (_choice_response("lane", 0.50), 1.0, 2.0),
        )
        assert (
            maybe_jev_route_uncertain(
                UNCERTAIN_MSG, user_config=_enabled_cfg(threshold=0.85), api_key="k",
            )
            is None
        )

    def test_disabled_makes_no_jev_call(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("Jev must not be called when disabled")

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _boom)
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        assert (
            fast_path_reason(
                UNCERTAIN_MSG,
                platform_key="telegram",
                chat_type="dm",
                user_config={"gateway": {"telegram": {"jev_routing": {"enabled": False}}}},
            )
            is None
        )
        assert called["n"] == 0

    def test_missing_key_falls_back_without_http(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("no HTTP without key")

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _boom)
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "",
        )
        assert (
            maybe_jev_route_uncertain(UNCERTAIN_MSG, user_config=_enabled_cfg()) is None
        )
        assert called["n"] == 0

    def test_http_failure_falls_back_to_deterministic(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )

        def _fail(*_a, **_k):
            raise RuntimeError("jev http 500")

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _fail)
        assert (
            maybe_jev_route_uncertain(
                UNCERTAIN_MSG, user_config=_enabled_cfg(), api_key="k",
            )
            is None
        )

    def test_clear_lane_never_calls_jev(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("clear lane must not call Jev")

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _boom)
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        assert (
            fast_path_reason(
                CLEAR_LANE_MSG,
                platform_key="telegram",
                chat_type="dm",
                user_config=_enabled_cfg(),
            )
            == "social"
        )
        assert called["n"] == 0


class TestJevRoutingHelpers:
    def test_parse_config_defaults_off(self):
        cfg = parse_jev_routing_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == 0.85
        assert load_jev_routing_config(None).enabled is False

    def test_build_request_includes_choice_wording(self):
        body = build_jev_routing_request("hmm")
        assert body["questions"]["route"]["instructions"] == CHOICE_INSTRUCTIONS
        assert set(body["questions"]["route"]["criteria"]) == set(CHOICE_CRITERIA)

    def test_parse_route_answer(self):
        choice, conf = parse_route_answer(_choice_response("task", 0.91))
        assert choice == "task"
        assert conf == pytest.approx(0.91)

    def test_build_request_includes_masked_prior_user_turns(self):
        history = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "token sk-live-abcdefghijklmnopqrstuvwxyz123456"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "follow up please"},
        ]
        body = build_jev_routing_request("alright then", history=history)
        prior_blob = next(s for s in body["state"] if s.startswith("Prior user turns"))
        assert "sk-live" not in prior_blob
        assert "[secret]" in prior_blob
        assert "follow up please" in prior_blob
        assert len(body["state"]) == 3

    def test_verdict_cache_skips_second_http(self):
        http = _FakeHttp(_choice_response("lane", 0.95))
        cfg = _enabled_cfg(verdict_cache_ttl_seconds=120)
        chat_id = 4242
        first = maybe_jev_route_uncertain(
            UNCERTAIN_MSG,
            user_config=cfg,
            chat_id=chat_id,
            http_client=http,
            api_key="test-key",
        )
        assert first == "social"
        assert len(http.calls) == 1
        second = maybe_jev_route_uncertain(
            UNCERTAIN_MSG,
            user_config=cfg,
            chat_id=chat_id,
            http_client=http,
            api_key="test-key",
        )
        assert second == "social"
        assert len(http.calls) == 1

    def test_breaker_opens_after_failures_and_probes_after_cooldown(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr(
            "gateway.run_turn_jev_routing.time.monotonic",
            lambda: clock["t"],
        )

        def _fail(*_a, **_k):
            raise RuntimeError("jev down")

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _fail)
        cfg = _enabled_cfg(breaker_trips=2, breaker_cooldown_seconds=30)
        for _ in range(2):
            assert (
                maybe_jev_route_uncertain(
                    UNCERTAIN_MSG, user_config=cfg, api_key="k", chat_id=1,
                )
                is None
            )
        blocked = maybe_jev_route_uncertain(
            UNCERTAIN_MSG, user_config=cfg, api_key="k", chat_id=1,
        )
        assert blocked is None

        clock["t"] = 100.0
        calls = {"n": 0}

        def _ok(*_a, **_k):
            calls["n"] += 1
            return (_choice_response("lane", 0.99), 1.0, 2.0)

        monkeypatch.setattr("gateway.run_turn_jev_routing._post_systemone", _ok)
        probed = maybe_jev_route_uncertain(
            UNCERTAIN_MSG, user_config=cfg, api_key="k", chat_id=1,
        )
        assert probed == "social"
        assert calls["n"] == 1
        after_reset = maybe_jev_route_uncertain(
            UNCERTAIN_MSG, user_config=cfg, api_key="k", chat_id=1,
        )
        assert after_reset == "social"
        assert calls["n"] == 2
