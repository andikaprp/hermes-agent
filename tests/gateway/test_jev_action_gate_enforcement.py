"""Execute-time enforcement for the LAB-60 Jev action gate.

The chooser is not approval. When ``computer_use.jev_action_gate`` or
``browser.jev_action_gate`` is enabled, a mutating computer-use / browser
tool call must not run unless Jev returns the caller-supplied action id
from the approved table. Error, timeout, low confidence, and a missing
key (all of which the chooser turns into ``reobserve``) mean the tool
call does not execute.

When the gate is disabled (the default), dispatch does not call Jev and
the handler sees the original args object.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

import tools.browser_tool  # noqa: F401 — registers browser_* handlers
import tools.browser_use_cli  # noqa: F401 — registers browser_exec
import tools.computer_use_tool  # noqa: F401 — registers computer_use
from gateway.jev_action_gate import (
    NONE_ID,
    REOBSERVE_ID,
    JevActionGateConfig,
)
from tools.registry import registry


_SECRET = "hunter2-field-value-must-not-reach-jev"
_SCREENSHOT = "base64-screenshot-must-not-reach-jev"


def _candidates(action_id: str = "click-appearance") -> List[Dict[str, str]]:
    return [
        {"id": action_id, "description": "Click the Appearance row"},
        {"id": REOBSERVE_ID, "description": "Take a fresh observation"},
        {"id": NONE_ID, "description": "Stop and ask for help"},
    ]


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


# (tool name, mutating args, config surface)
_GATED = (
    ("computer_use", {"action": "click", "element": 1, "text": _SECRET}, "computer_use"),
    ("browser_click", {"ref": "e12", "page_text": _SECRET}, "browser"),
    (
        "browser_exec",
        {"code": f"print({_SECRET!r})", "screenshot": _SCREENSHOT},
        "browser",
    ),
)


@pytest.fixture
def swap_handler():
    """Replace a registered handler with a recorder. Restores on teardown."""
    saved: List[Tuple[Any, Any]] = []

    def _swap(name: str) -> list:
        entry = registry.get_entry(name)
        assert entry is not None, name
        calls: list = []

        def handler(args, **kwargs):
            calls.append((args, kwargs))
            return json.dumps({"ok": True, "tool": name})

        saved.append((entry, entry.handler))
        entry.handler = handler
        return calls

    yield _swap
    for entry, handler in saved:
        entry.handler = handler


def _patch_config(monkeypatch, *, computer_use: bool = False, browser: bool = False):
    def load(user_config=None, surface=None):
        if surface == "computer_use":
            return JevActionGateConfig(enabled=computer_use)
        if surface == "browser":
            return JevActionGateConfig(enabled=browser)
        return JevActionGateConfig(enabled=computer_use or browser)

    monkeypatch.setattr(
        "gateway.jev_action_gate.load_jev_action_gate_config", load,
    )


def _forbid_jev(monkeypatch) -> list:
    called: list = []

    def boom(*_args, **_kwargs):
        called.append(1)
        raise AssertionError("Jev must not be called")

    monkeypatch.setattr("gateway.jev_action_gate._post_systemone", boom)
    monkeypatch.setattr("gateway.jev_action_gate.choose_next_action", boom)
    return called


def _dispatch(name: str, args: Dict[str, Any]):
    return registry.dispatch(name, args)


def _parsed(raw: Any) -> Dict[str, Any]:
    assert isinstance(raw, str), raw
    return json.loads(raw)


class TestEnabledBlocksWithoutApproval:
    @pytest.mark.parametrize("name,args,surface", _GATED)
    def test_blocked_without_approved_id(self, monkeypatch, swap_handler, name, args, surface):
        calls = swap_handler(name)
        _patch_config(
            monkeypatch,
            computer_use=surface == "computer_use",
            browser=surface == "browser",
        )
        jev = _forbid_jev(monkeypatch)

        raw = _dispatch(name, dict(args))

        assert calls == []
        assert jev == []
        body = _parsed(raw)
        assert body["executed"] is False
        assert body["blocked"] is True
        assert body["reason"] == "missing_approved_action_id"
        assert _SECRET not in raw
        assert _SCREENSHOT not in raw

    @pytest.mark.parametrize("name,args,surface", _GATED)
    def test_blocked_when_id_not_in_approved_table(
        self, monkeypatch, swap_handler, name, args, surface,
    ):
        calls = swap_handler(name)
        _patch_config(
            monkeypatch,
            computer_use=surface == "computer_use",
            browser=surface == "browser",
        )
        jev = _forbid_jev(monkeypatch)
        payload = dict(args)
        payload["jev_action_id"] = "not-in-table"
        payload["jev_goal"] = "Open Appearance"
        payload["jev_candidates"] = _candidates()

        raw = _dispatch(name, payload)

        assert calls == []
        assert jev == []
        body = _parsed(raw)
        assert body["executed"] is False
        assert body["reason"] == "action_not_in_approved_table"


class TestEnabledAllowsApprovedId:
    @pytest.mark.parametrize("name,args,surface", _GATED)
    def test_executes_only_when_jev_returns_the_id(
        self, monkeypatch, swap_handler, name, args, surface,
    ):
        calls = swap_handler(name)
        _patch_config(
            monkeypatch,
            computer_use=surface == "computer_use",
            browser=surface == "browser",
        )
        seen: list = []

        def fake_post(body, **_kwargs):
            seen.append(body)
            return (_choice_payload("click-appearance", 0.93), 1.0, 2.0)

        monkeypatch.setattr("gateway.jev_action_gate._post_systemone", fake_post)
        monkeypatch.setattr(
            "gateway.jev_action_gate.resolve_typesafe_api_key", lambda: "sk-test",
        )
        payload = dict(args)
        payload.update({
            "jev_action_id": "click-appearance",
            "jev_goal": "Open Appearance",
            "jev_candidates": _candidates(),
            "screenshot": _SCREENSHOT,
            "page_text": _SECRET,
            "field_value": _SECRET,
        })

        raw = _dispatch(name, payload)

        assert len(calls) == 1
        handler_args = calls[0][0]
        assert handler_args.get("action", handler_args.get("ref", handler_args.get("code")))
        assert "jev_action_id" not in handler_args
        assert seen, "execution must consult Jev before acting"
        blob = json.dumps(seen[0])
        assert _SECRET not in blob
        assert _SCREENSHOT not in blob
        assert "field_value" not in blob
        assert "screenshot" not in blob
        assert _parsed(raw)["ok"] is True


class TestEnabledErrorDoesNotExecute:
    @pytest.mark.parametrize("name,args,surface", _GATED)
    def test_jev_error_does_not_execute(self, monkeypatch, swap_handler, name, args, surface):
        calls = swap_handler(name)
        _patch_config(
            monkeypatch,
            computer_use=surface == "computer_use",
            browser=surface == "browser",
        )

        def boom(*_a, **_k):
            raise RuntimeError("jev http 503")

        monkeypatch.setattr("gateway.jev_action_gate._post_systemone", boom)
        monkeypatch.setattr(
            "gateway.jev_action_gate.resolve_typesafe_api_key", lambda: "sk-test",
        )
        payload = dict(args)
        payload.update({
            "jev_action_id": "click-appearance",
            "jev_goal": "Open Appearance",
            "jev_candidates": _candidates(),
        })

        raw = _dispatch(name, payload)

        assert calls == []
        body = _parsed(raw)
        assert body["executed"] is False
        assert body["blocked"] is True
        assert body["reason"] in {"RuntimeError", "error", "timeout", "missing_key"}
        assert _SECRET not in raw

    @pytest.mark.parametrize(
        "reason_setup",
        ["missing_key", "below_threshold", "timeout"],
    )
    def test_fail_open_reobserve_does_not_execute(self, monkeypatch, swap_handler, reason_setup):
        calls = swap_handler("computer_use")
        _patch_config(monkeypatch, computer_use=True)
        if reason_setup == "missing_key":
            monkeypatch.setattr("gateway.jev_action_gate.resolve_typesafe_api_key", lambda: "")
        elif reason_setup == "below_threshold":
            monkeypatch.setattr(
                "gateway.jev_action_gate._post_systemone",
                lambda *_a, **_k: (_choice_payload("click-appearance", 0.2), 1.0, 2.0),
            )
            monkeypatch.setattr(
                "gateway.jev_action_gate.resolve_typesafe_api_key", lambda: "sk-test",
            )
        else:
            from concurrent.futures import TimeoutError as FuturesTimeoutError

            def slow(*_a, **_k):
                raise FuturesTimeoutError()

            monkeypatch.setattr("gateway.jev_action_gate._post_systemone", slow)
            monkeypatch.setattr(
                "gateway.jev_action_gate.resolve_typesafe_api_key", lambda: "sk-test",
            )

        raw = _dispatch("computer_use", {
            "action": "click",
            "element": 3,
            "text": _SECRET,
            "jev_action_id": "click-appearance",
            "jev_goal": "Open Appearance",
            "jev_candidates": _candidates(),
        })

        assert calls == []
        body = _parsed(raw)
        assert body["executed"] is False
        assert body["reason"] == {
            "missing_key": "missing_key",
            "below_threshold": "below_threshold",
            "timeout": "timeout",
        }[reason_setup]


class TestDisabledUnchanged:
    @pytest.mark.parametrize("name,args,surface", _GATED)
    def test_disabled_skips_jev_and_keeps_args(
        self, monkeypatch, swap_handler, name, args, surface,
    ):
        calls = swap_handler(name)
        _patch_config(monkeypatch, computer_use=False, browser=False)
        _forbid_jev(monkeypatch)
        payload = dict(args)

        raw = _dispatch(name, payload)

        assert len(calls) == 1
        assert calls[0][0] is payload
        assert _parsed(raw)["ok"] is True

    def test_other_surface_stays_off(self, monkeypatch, swap_handler):
        calls = swap_handler("browser_click")
        _patch_config(monkeypatch, computer_use=True, browser=False)
        _forbid_jev(monkeypatch)
        payload = {"ref": "e1"}

        raw = _dispatch("browser_click", payload)

        assert len(calls) == 1
        assert calls[0][0] is payload
        assert _parsed(raw)["ok"] is True


class TestObservationNotGated:
    @pytest.mark.parametrize(
        "name,args",
        [
            ("computer_use", {"action": "capture", "mode": "ax"}),
            ("browser_snapshot", {"full": False}),
        ],
    )
    def test_enabled_observation_does_not_call_jev(self, monkeypatch, swap_handler, name, args):
        calls = swap_handler(name)
        _patch_config(monkeypatch, computer_use=True, browser=True)
        _forbid_jev(monkeypatch)
        payload = dict(args)

        raw = _dispatch(name, payload)

        assert len(calls) == 1
        assert calls[0][0] is payload
        assert _parsed(raw)["ok"] is True


class TestSchemaOnlyWhenEnabled:
    def test_computer_use_schema_gains_action_id_only_when_enabled(self, monkeypatch):
        entry = registry.get_entry("computer_use")
        assert entry is not None
        assert "jev_action_id" not in entry.schema["parameters"]["properties"]
        assert entry.dynamic_schema_overrides is not None

        _patch_config(monkeypatch, computer_use=False)
        assert not entry.dynamic_schema_overrides()

        _patch_config(monkeypatch, computer_use=True)
        overrides = entry.dynamic_schema_overrides()
        assert "jev_action_id" in overrides["parameters"]["properties"]
        assert "jev_action_id" not in entry.schema["parameters"]["properties"]
