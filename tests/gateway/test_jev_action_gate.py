"""Invariant tests for the LAB-60 Jev computer/browser action gate.

Contracts (not snapshots):
- disabled (default) -> reobserve, no System One call
- sensitive goal/label refused before send -> reobserve
- withheld payload keys refused -> reobserve
- API failure / timeout / missing key -> reobserve (fail open)
- below threshold -> reobserve
- high-confidence choice must be an id from the approved table
- invented ids from Jev are rejected -> reobserve
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from gateway.jev_action_gate import (
    NONE_ID,
    REOBSERVE_ID,
    JevActionGateConfig,
    assert_no_withheld_payload,
    build_action_choice_request,
    choose_next_action,
    decision_payload,
    looks_sensitive,
    parse_action_answer,
    parse_jev_action_gate_config,
)


def _candidates(*extra: Dict[str, str]) -> List[Dict[str, str]]:
    base = [
        {"id": "click-appearance", "description": "Click the Appearance row"},
        {"id": REOBSERVE_ID, "description": "Take a fresh observation"},
        {"id": NONE_ID, "description": "Stop and ask for help"},
    ]
    return base + list(extra)


def _choice_payload(choice: str, confidence: float) -> Dict[str, Any]:
    return {
        "answers": {
            "next_action": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
            }
        }
    }


class TestConfigDefaults:
    def test_defaults_off_on_both_surfaces(self):
        for surface in ("computer_use", "browser"):
            block = DEFAULT_CONFIG[surface]["jev_action_gate"]
            assert block["enabled"] is False
            assert 0.0 < float(block["threshold"]) <= 1.0
            assert float(block["timeout_seconds"]) >= 1.0
        parsed = parse_jev_action_gate_config(None)
        assert parsed.enabled is False


class TestWithheldContext:
    def test_sensitive_goal_and_label_detected(self):
        assert looks_sensitive("type the password into the field")
        assert looks_sensitive("sk-abcdefghijklmnopqrstuvwxyz")
        assert looks_sensitive("user@example.com")
        assert not looks_sensitive("Open Appearance settings")

    def test_forbidden_payload_keys_refused(self):
        with pytest.raises(ValueError, match="withheld"):
            assert_no_withheld_payload({"goal": "x", "screenshot": "base64..."})

    def test_request_state_has_no_page_text_or_field_values(self):
        body = build_action_choice_request(
            goal="Open Appearance",
            candidates=_candidates(),
            regions=[{"id": "r1", "role": "button", "label": "Appearance", "interactive": True}],
        )
        blob = json.dumps(body).lower()
        assert "page text" not in blob or "never receive" in blob
        assert "field_value" not in blob
        assert "base64" not in blob
        assert set(body["questions"]["next_action"]["criteria"]) >= {
            "click-appearance", REOBSERVE_ID, NONE_ID,
        }


class TestChooseInvariants:
    def test_disabled_reobserves_without_api_call(self):
        called = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("System One must not be called when disabled")

        with patch("gateway.jev_action_gate._post_systemone", side_effect=boom):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=False),
                api_key="sk-test",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.took_jev is False
        assert decision.fallback is False
        assert decision.reason == "disabled"
        assert called == []

    def test_sensitive_goal_reobserves_without_api_call(self):
        called = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("must not call Jev with sensitive goal")

        with patch("gateway.jev_action_gate._post_systemone", side_effect=boom):
            decision = choose_next_action(
                goal="enter the password hunter2",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True),
                api_key="sk-test",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.fallback is True
        assert "sensitive" in decision.reason
        assert called == []

    def test_missing_key_fails_open(self):
        with patch("gateway.jev_action_gate.resolve_typesafe_api_key", return_value=""):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True),
                api_key="",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.fallback is True
        assert decision.reason == "missing_key"

    def test_api_error_fails_open(self):
        with patch(
            "gateway.jev_action_gate._post_systemone",
            side_effect=RuntimeError("jev http 503"),
        ):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True),
                api_key="sk-test",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.fallback is True

    def test_below_threshold_reobserves(self):
        with patch(
            "gateway.jev_action_gate._post_systemone",
            return_value=(_choice_payload("click-appearance", 0.4), 1.0, 2.0),
        ):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True, threshold=0.65),
                api_key="sk-test",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.reason == "below_threshold"
        assert decision.took_jev is True
        assert decision.fallback is True

    def test_high_confidence_returns_table_id_only(self):
        with patch(
            "gateway.jev_action_gate._post_systemone",
            return_value=(_choice_payload("click-appearance", 0.91), 1.0, 2.0),
        ):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True, threshold=0.65),
                api_key="sk-test",
            )
        assert decision.action_id == "click-appearance"
        assert decision.took_jev is True
        assert decision.fallback is False
        assert decision.action_id in {c["id"] for c in _candidates()}

    def test_invented_id_rejected(self):
        with patch(
            "gateway.jev_action_gate._post_systemone",
            return_value=(_choice_payload("hack-the-planet", 0.99), 1.0, 2.0),
        ):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True),
                api_key="sk-test",
            )
        assert decision.action_id == REOBSERVE_ID
        assert decision.fallback is True

    def test_parse_rejects_unknown_choice(self):
        with pytest.raises(ValueError, match="approved table"):
            parse_action_answer(_choice_payload("not-in-table", 0.9), {"a": "x"})

    def test_decision_payload_flags(self):
        with patch(
            "gateway.jev_action_gate._post_systemone",
            return_value=(_choice_payload(NONE_ID, 0.95), 1.0, 2.0),
        ):
            decision = choose_next_action(
                goal="Open Appearance",
                candidates=_candidates(),
                cfg=JevActionGateConfig(enabled=True),
                api_key="sk-test",
            )
        payload = decision_payload(decision)
        assert payload["none"] is True
        assert payload["reobserve"] is False
        assert payload["action_id"] == NONE_ID


class TestToolSurface:
    def test_handler_returns_json_and_respects_table(self):
        from tools.jev_action_gate_tool import handle_jev_choose_action

        with patch(
            "tools.jev_action_gate_tool.load_jev_action_gate_config",
            return_value=JevActionGateConfig(enabled=True, threshold=0.65),
        ), patch(
            "gateway.jev_action_gate._post_systemone",
            return_value=(_choice_payload("click-appearance", 0.9), 1.0, 2.0),
        ), patch(
            "gateway.jev_action_gate.resolve_typesafe_api_key",
            return_value="sk-test",
        ):
            raw = handle_jev_choose_action({
                "goal": "Open Appearance",
                "candidates": _candidates(),
                "surface": "computer_use",
            })
        data = json.loads(raw)
        assert data["action_id"] == "click-appearance"

    def test_check_fn_false_when_disabled(self):
        from tools.jev_action_gate_tool import check_jev_choose_action_available

        with patch(
            "tools.jev_action_gate_tool.load_jev_action_gate_config",
            return_value=JevActionGateConfig(enabled=False),
        ):
            assert check_jev_choose_action_available() is False
