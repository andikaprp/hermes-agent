"""Invariant tests for optional Jev completion verification (LAB-61).

Contracts (not snapshots):
- disabled / missing key / HTTP failure / timeout -> passthrough (claim done)
- confident ``done`` + no evidence contradiction -> claim done, record ledger
- confident ``partial``/``failed`` -> HITL escalate (never auto-claim done)
- below-threshold confidence -> HITL escalate
- ``done`` contradicted by failed/stale evidence -> HITL escalate
- config defaults off; request state is hashes/metadata only (no raw prompts)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gateway.run_turn_jev_completion import (
    CHOICE_CRITERIA,
    CHOICE_INSTRUCTIONS,
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS,
    apply_completion_verdict,
    build_completion_request,
    evaluate_turn_completion,
    evidence_contradicts_done,
    load_jev_completion_config,
    maybe_verify_turn_completion,
    parse_completion_answer,
    parse_jev_completion_config,
    result_payload_hash,
)


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
    for o in others:
        probs[o] = rem
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
    def __init__(
        self,
        payload: Any = None,
        *,
        error: Optional[BaseException] = None,
    ):
        self.payload = (
            payload if payload is not None else _choice_response("done", 0.95)
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


class TestConfigDefaults:
    def test_parse_defaults_off(self):
        cfg = parse_jev_completion_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == DEFAULT_THRESHOLD
        assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert load_jev_completion_config(None).enabled is False
        assert load_jev_completion_config({}).enabled is False

    def test_default_config_declares_jev_completion_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        block = DEFAULT_CONFIG["gateway"]["jev_completion"]
        assert block["enabled"] is False
        assert float(block["threshold"]) >= 0.0
        assert float(block["timeout_seconds"]) >= 1.0

    def test_build_request_is_hashes_only(self):
        result = _result(final_response="SECRET PROMPT TEXT should not appear")
        body = build_completion_request(
            result_hash="abc123",
            agent_result=result,
            evidence={"status": "passed", "kind": "test", "scope": "full", "exit_code": 0},
        )
        q = body["questions"]["completion"]
        assert q["type"] == "choice"
        assert q["instructions"] == CHOICE_INSTRUCTIONS
        assert q["criteria"] == CHOICE_CRITERIA
        blob = json.dumps(body)
        assert "SECRET PROMPT" not in blob
        assert "result_hash=abc123" in blob
        assert "evidence_status=passed" in blob


class TestEvaluateFailOpen:
    def test_disabled_never_calls_http(self):
        http = _FakeHttp()
        v = evaluate_turn_completion(
            _result(), user_config={}, http_client=http, api_key="k",
        )
        assert v.claim_done is True
        assert v.escalate is False
        assert v.fallback is True
        assert http.calls == []

    def test_missing_key_passthrough(self):
        http = _FakeHttp()
        v = evaluate_turn_completion(
            _result(),
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="",
        )
        assert v.claim_done is True
        assert v.escalate is False
        assert http.calls == []

    def test_http_failure_passthrough(self):
        http = _FakeHttp(error=RuntimeError("jev http 500"))
        v = evaluate_turn_completion(
            _result(),
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
        )
        assert v.claim_done is True
        assert v.escalate is False
        assert v.fallback is True

    def test_timeout_passthrough(self):
        http = _FakeHttp(error=TimeoutError("timed out waiting for jev"))
        v = evaluate_turn_completion(
            _result(),
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
        )
        assert v.claim_done is True
        assert v.escalate is False


class TestVerdictsAndHitl:
    def test_confident_done_claims(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        http = _FakeHttp(_choice_response("done", 0.95))
        monkeypatch.setattr(
            "gateway.run_turn_jev_completion._evidence_snapshot",
            lambda **_k: {"status": "passed", "kind": "test", "scope": "full", "exit_code": 0},
        )
        result = _result()
        v = evaluate_turn_completion(
            result,
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
            session_id="s1",
            cwd=tmp_path,
        )
        assert v.claim_done is True
        assert v.escalate is False
        assert v.verdict == "done"
        out = apply_completion_verdict(result, v)
        assert out.get("completed") is True
        assert out.get("hitl_escalation") is not True
        assert "shipped the fix" in (out.get("final_response") or "")

    def test_confident_partial_escalates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        http = _FakeHttp(_choice_response("partial", 0.9))
        monkeypatch.setattr(
            "gateway.run_turn_jev_completion._evidence_snapshot",
            lambda **_k: {"status": "unverified"},
        )
        result = _result()
        v = evaluate_turn_completion(
            result,
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
            session_id="s1",
            cwd=tmp_path,
        )
        assert v.escalate is True
        assert v.claim_done is False
        out = apply_completion_verdict(result, v)
        assert out.get("completed") is False
        assert out.get("partial") is True
        assert out.get("hitl_escalation") is True
        assert "verdict=partial" in (out.get("final_response") or "")

    def test_below_threshold_escalates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        http = _FakeHttp(_choice_response("done", 0.4))
        monkeypatch.setattr(
            "gateway.run_turn_jev_completion._evidence_snapshot",
            lambda **_k: {"status": "passed"},
        )
        result = _result()
        v = evaluate_turn_completion(
            result,
            user_config=_enabled_cfg(threshold=0.75),
            http_client=http,
            api_key="k",
            session_id="s1",
            cwd=tmp_path,
        )
        assert v.escalate is True
        assert v.claim_done is False
        assert v.reason == "below_threshold"

    def test_done_with_failed_evidence_contradicts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        assert evidence_contradicts_done("failed") is True
        assert evidence_contradicts_done("stale") is True
        assert evidence_contradicts_done("passed") is False
        http = _FakeHttp(_choice_response("done", 0.99))
        monkeypatch.setattr(
            "gateway.run_turn_jev_completion._evidence_snapshot",
            lambda **_k: {"status": "failed", "kind": "test", "exit_code": 1},
        )
        result = _result()
        v = evaluate_turn_completion(
            result,
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
            session_id="s1",
            cwd=tmp_path,
        )
        assert v.escalate is True
        assert v.contradicted is True
        assert v.claim_done is False

    def test_maybe_verify_writes_ledger(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        http = _FakeHttp(_choice_response("failed", 0.88))
        monkeypatch.setattr(
            "gateway.run_turn_jev_completion._evidence_snapshot",
            lambda **_k: {"status": "unverified"},
        )
        out = maybe_verify_turn_completion(
            _result(),
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="k",
            session_id="sess-lab61",
            cwd=tmp_path,
        )
        assert out.get("hitl_escalation") is True
        db = home / "verification_evidence.db"
        assert db.is_file()
        conn = sqlite3.connect(str(db))
        row = conn.execute(
            "SELECT kind, status, output_summary FROM verification_events "
            "WHERE kind = 'jev_completion' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "jev_completion"
        assert row[1] == "failed"
        summary = row[2]
        assert "SECRET" not in summary
        assert "result_hash" in summary or "failed" in summary

    def test_result_hash_stable_and_ignores_text_body(self):
        a = result_payload_hash(_result(final_response="aaa"))
        b = result_payload_hash(_result(final_response="bbb"))
        # Same length → same hash (chars counted, body not hashed).
        assert a == b
        c = result_payload_hash(_result(final_response="aaaa"))
        assert a != c


class TestParseAnswer:
    def test_parse_ok(self):
        choice, conf = parse_completion_answer(_choice_response("done", 0.81))
        assert choice == "done"
        assert conf == pytest.approx(0.81)

    def test_parse_rejects_unknown(self):
        with pytest.raises(ValueError):
            parse_completion_answer({"answers": {"completion": {"choice": "maybe", "confidence": 1}}})
