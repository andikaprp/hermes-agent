"""Invariant tests for optional Jev pre-spawn gate on delegate_task.

Contracts:
- disabled (default) -> spawn path, no System One call
- API failure / missing key -> spawn (fallback)
- high-confidence ``do_directly`` -> no spawn (override)
- high-confidence ``delegate`` -> spawn (matches default)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.delegate_tool_jev import (
    JevCheckConfig,
    build_delegate_choice_request,
    evaluate_jev_delegate_gate,
    parse_choice_answer,
    parse_jev_check_config,
)


def _tasks(goal: str = "research the auth bug") -> list:
    return [{"goal": goal}]


def _choice_payload(choice: str, confidence: float) -> dict:
    return {
        "answers": {
            "route": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: confidence, "do_directly" if choice == "delegate" else "delegate": 1.0 - confidence},
            }
        }
    }


class TestJevCheckConfigDefaults:
    def test_defaults_off(self):
        block = DEFAULT_CONFIG["delegation"]["jev_check"]
        assert block["enabled"] is False
        assert block["threshold"] == 0.8
        parsed = parse_jev_check_config(None)
        assert parsed.enabled is False
        assert parsed.threshold == 0.8

    def test_choice_request_shape(self):
        body = build_delegate_choice_request(_tasks(), context="parent notes")
        assert "delegate" in body["questions"]["route"]["criteria"]
        assert "do_directly" in body["questions"]["route"]["criteria"]
        assert body["questions"]["route"]["type"] == "choice"
        assert "auth bug" in body["state"]
        assert "parent notes" in body["state"]
        choice, conf = parse_choice_answer(_choice_payload("do_directly", 0.91))
        assert choice == "do_directly"
        assert conf == pytest.approx(0.91)


class TestJevDelegateGateInvariants:
    def test_disabled_spawns_without_api_call(self):
        called = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("System One must not be called when disabled")

        with patch("tools.delegate_tool_jev._call_systemone", side_effect=boom):
            decision = evaluate_jev_delegate_gate(
                _tasks(),
                cfg=JevCheckConfig(enabled=False),
                api_key="sk-test",
            )
        assert decision.spawn is True
        assert decision.override is False
        assert decision.fallback is False
        assert decision.reason == "disabled"
        assert called == []

    def test_failure_falls_back_to_spawn(self):
        with patch(
            "tools.delegate_tool_jev._call_systemone",
            side_effect=RuntimeError("jev http 500"),
        ):
            decision = evaluate_jev_delegate_gate(
                _tasks(),
                cfg=JevCheckConfig(enabled=True, threshold=0.8),
                api_key="sk-test",
            )
        assert decision.spawn is True
        assert decision.fallback is True
        assert decision.override is False
        assert decision.reason == "RuntimeError"

    def test_high_confidence_do_directly_skips_spawn(self):
        with patch(
            "tools.delegate_tool_jev._call_systemone",
            return_value=_choice_payload("do_directly", 0.95),
        ):
            decision = evaluate_jev_delegate_gate(
                _tasks(),
                cfg=JevCheckConfig(enabled=True, threshold=0.8),
                api_key="sk-test",
            )
        assert decision.spawn is False
        assert decision.override is True
        assert decision.choice == "do_directly"
        assert decision.confidence == pytest.approx(0.95)

    def test_high_confidence_delegate_spawns(self):
        with patch(
            "tools.delegate_tool_jev._call_systemone",
            return_value=_choice_payload("delegate", 0.92),
        ):
            decision = evaluate_jev_delegate_gate(
                _tasks(),
                cfg=JevCheckConfig(enabled=True, threshold=0.8),
                api_key="sk-test",
            )
        assert decision.spawn is True
        assert decision.override is False
        assert decision.fallback is False
        assert decision.choice == "delegate"


class TestDelegateTaskJevSeam:
    """E2E through ``delegate_task``: override skips ``_build_children``."""

    def test_do_directly_never_builds_children(self, monkeypatch):
        from tools import delegate_tool as dt

        parent = MagicMock()
        parent._delegate_depth = 0
        builds = []

        monkeypatch.setattr(dt, "is_spawn_paused", lambda: False)
        monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 3)
        monkeypatch.setattr(dt, "_load_config", lambda: {
            "max_iterations": 10,
            "jev_check": {"enabled": True, "threshold": 0.8, "model": "jev-latest", "timeout_seconds": 30},
        })
        monkeypatch.setattr(
            dt, "_resolve_delegation_credentials",
            lambda *_a, **_k: {"model": "m", "provider": "p"},
        )
        monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 3)
        monkeypatch.setattr(
            dt, "_normalize_task_list",
            lambda goal, context, tasks, output_schema, top_role, max_children: (
                [{"goal": goal or "g"}], None,
            ),
        )
        monkeypatch.setattr(dt, "_coerce_task_schemas", lambda *a, **k: (None, None))
        monkeypatch.setattr(dt, "_coerce_task_images", lambda *a, **k: (None, None))

        def _no_build(*_a, **_k):
            builds.append(1)
            raise AssertionError("_build_children must not run on do_directly override")

        monkeypatch.setattr(dt, "_build_children", _no_build)
        monkeypatch.setattr(
            "tools.delegate_tool_jev._call_systemone",
            lambda *_a, **_k: _choice_payload("do_directly", 0.99),
        )
        monkeypatch.setattr(
            "tools.delegate_tool_jev.resolve_typesafe_api_key",
            lambda: "sk-test",
        )

        out = json.loads(dt.delegate_task(goal="simple lookup", parent_agent=parent))
        assert out["status"] == "skipped"
        assert out["reason"] == "jev_do_directly"
        assert builds == []
