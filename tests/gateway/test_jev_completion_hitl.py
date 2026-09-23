"""LAB-61 HITL consumer: a partial/low-confidence verdict is an ask, not a notice.

The verifier already sets ``hitl_escalation``. These tests require a consumer
that reads that flag and ends the turn in a real question (the existing
``tools.clarify_gateway`` clarification seam) without auto-claiming done.
Disabled and confident-done paths must stay unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gateway.jev_completion_hitl import (
    CLARIFY_SEAM,
    consume_hitl_escalation,
    get_pending_ask,
    intercept_completion_reply,
)
from gateway.run import _should_clear_resume_pending_after_turn
from gateway.run_turn_jev_completion import (
    CHOICE_CRITERIA,
    maybe_verify_turn_completion,
)
from tools.clarify_gateway import get_pending_for_session

ASK_PHRASE = "Did this turn finish what you asked?"
NOTICE_PREFIX = "⚠️ Completion check needs a human look"


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.75,
        "model": "jev-latest",
        "timeout_seconds": 30,
    }
    block.update(overrides)
    return {"gateway": {"jev_completion": block}}


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    others = [c for c in CHOICE_CRITERIA if c != choice]
    probs = {choice: confidence}
    rem = (1.0 - confidence) / max(1, len(others))
    for other in others:
        probs[other] = rem
    return {
        "answers": {
            "completion": {
                "choice": choice,
                "confidence": confidence,
                "probabilities": probs,
            }
        }
    }


class _FakeHttp:
    def __init__(self, payload: Any = None, *, error: Optional[BaseException] = None):
        self.payload = payload if payload is not None else _choice_response("done", 0.95)
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


def _result(**overrides: Any) -> Dict[str, Any]:
    base = {
        "final_response": "shipped the fix",
        "completed": True,
        "failed": False,
        "partial": False,
        "interrupted": False,
        "api_calls": 3,
        "error": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def session_key():
    key = "sess-hitl"
    yield key
    from gateway.jev_completion_hitl import clear_pending_ask

    clear_pending_ask(key)


def _verify(result, http, monkeypatch, tmp_path, *, cfg=None, evidence=None):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "gateway.run_turn_jev_completion._evidence_snapshot",
        lambda **_k: evidence if evidence is not None else {"status": "unverified"},
    )
    return maybe_verify_turn_completion(
        result,
        user_config=cfg if cfg is not None else _enabled_cfg(),
        http_client=http,
        api_key="k",
        session_id="s1",
        cwd=tmp_path,
    )


def _assert_explicit_ask(out: dict, ask: str, session_key: str) -> None:
    assert ask
    assert ask == out["final_response"]
    assert ASK_PHRASE in ask
    assert "?" in ask
    assert NOTICE_PREFIX not in ask
    assert out["completed"] is False
    assert out["partial"] is True
    assert out["approval_pending"] is True
    assert out["hitl_escalation"] is True
    assert out["hitl_ask"]["seam"] == CLARIFY_SEAM
    assert out["hitl_ask"]["seam"] == "tools.clarify_gateway"
    assert "Confirm done" in out["hitl_ask"]["choices"]
    assert _should_clear_resume_pending_after_turn(out) is False
    pending = get_pending_ask(session_key)
    assert pending is not None
    assert pending["question"] == ask
    clarify = get_pending_for_session(session_key, include_choice_prompts=True)
    assert clarify is not None
    assert clarify.clarify_id == pending["clarify_id"]
    assert list(clarify.choices or []) == ["Confirm done", "Not done"]


class TestHitlConsumer:
    def test_partial_ends_in_explicit_ask_and_does_not_claim_done(
        self, monkeypatch, tmp_path, session_key
    ):
        out = _verify(
            _result(already_sent=True),
            _FakeHttp(_choice_response("partial", 0.9)),
            monkeypatch,
            tmp_path,
        )
        ask = consume_hitl_escalation(out, session_key=session_key)
        _assert_explicit_ask(out, ask, session_key)
        assert out["already_sent"] is False
        assert "Draft reply (not a completion)" in ask
        assert "shipped the fix" in ask

    def test_low_confidence_asks_instead_of_claiming_done(
        self, monkeypatch, tmp_path, session_key
    ):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("done", 0.4)),
            monkeypatch,
            tmp_path,
            evidence={"status": "passed"},
        )
        ask = consume_hitl_escalation(out, session_key=session_key)
        _assert_explicit_ask(out, ask, session_key)
        assert "verdict=done" in ask

    def test_contradiction_asks_instead_of_claiming_done(
        self, monkeypatch, tmp_path, session_key
    ):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("done", 0.99)),
            monkeypatch,
            tmp_path,
            evidence={"status": "failed", "kind": "test", "exit_code": 1},
        )
        ask = consume_hitl_escalation(out, session_key=session_key)
        _assert_explicit_ask(out, ask, session_key)

    def test_failed_verdict_asks(self, monkeypatch, tmp_path, session_key):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("failed", 0.88)),
            monkeypatch,
            tmp_path,
        )
        ask = consume_hitl_escalation(out, session_key=session_key)
        _assert_explicit_ask(out, ask, session_key)
        assert "verdict=failed" in ask

    def test_disabled_gate_leaves_turn_unchanged(self, monkeypatch, tmp_path, session_key):
        result = _result()
        out = _verify(result, _FakeHttp(), monkeypatch, tmp_path, cfg={})
        ask = consume_hitl_escalation(out, session_key=session_key)
        assert ask is None
        assert out["final_response"] == "shipped the fix"
        assert out["completed"] is True
        assert out.get("hitl_escalation") is not True
        assert out.get("approval_pending") is not True
        assert "hitl_ask" not in out
        assert get_pending_ask(session_key) is None
        assert get_pending_for_session(session_key, include_choice_prompts=True) is None

    def test_confident_done_is_normal_done_path(self, monkeypatch, tmp_path, session_key):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("done", 0.95)),
            monkeypatch,
            tmp_path,
            evidence={"status": "passed", "kind": "test", "scope": "full", "exit_code": 0},
        )
        ask = consume_hitl_escalation(out, session_key=session_key)
        assert ask is None
        assert out["completed"] is True
        assert out["final_response"] == "shipped the fix"
        assert out.get("hitl_escalation") is not True
        assert out.get("approval_pending") is not True
        assert out["jev_completion"]["claim_done"] is True
        assert _should_clear_resume_pending_after_turn(out) is True
        assert get_pending_ask(session_key) is None

    def test_partial_without_flag_is_not_an_ask(self, session_key):
        result = _result(completed=False, partial=True, final_response="still working")
        ask = consume_hitl_escalation(result, session_key=session_key)
        assert ask is None
        assert result["final_response"] == "still working"
        assert "approval_pending" not in result
        assert get_pending_ask(session_key) is None

    def test_confirm_reply_acks_without_flipping_completed(
        self, monkeypatch, tmp_path, session_key
    ):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("partial", 0.9)),
            monkeypatch,
            tmp_path,
        )
        consume_hitl_escalation(out, session_key=session_key)
        outcome = intercept_completion_reply(session_key, "1")
        assert outcome is not None
        assert outcome.action == "ack"
        assert "not marked" in outcome.text.lower()
        assert out["completed"] is False
        assert get_pending_ask(session_key) is None
        assert get_pending_for_session(session_key, include_choice_prompts=True) is None

    def test_correction_releases_ask_for_followup(self, monkeypatch, tmp_path, session_key):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("partial", 0.9)),
            monkeypatch,
            tmp_path,
        )
        consume_hitl_escalation(out, session_key=session_key)
        outcome = intercept_completion_reply(session_key, "the tests are still red")
        assert outcome is not None
        assert outcome.action == "release"
        assert get_pending_ask(session_key) is None
        assert get_pending_for_session(session_key, include_choice_prompts=True) is None
        assert out["completed"] is False

    def test_slash_command_leaves_ask_pending(self, monkeypatch, tmp_path, session_key):
        out = _verify(
            _result(),
            _FakeHttp(_choice_response("partial", 0.9)),
            monkeypatch,
            tmp_path,
        )
        consume_hitl_escalation(out, session_key=session_key)
        assert intercept_completion_reply(session_key, "/new") is None
        assert get_pending_ask(session_key) is not None
