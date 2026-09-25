"""Invariant tests for optional Jev PR review triage (LAB-62).

Contracts (not snapshots):
- disabled -> no Jev HTTP call; should_post False
- low-confidence approve -> needs-human (never auto-approve)
- high-confidence approve / request-changes preserved
- failure / missing key / empty payload -> needs-human
- state is hashes/metadata + bounded diff (not unbounded file dumps)
- at most one verdict comment per PR (marker dedupe)
- github event is always COMMENT (human review gate preserved)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agent.jev_review_triage import (
    CHOICE_CRITERIA,
    CHOICE_INSTRUCTIONS,
    VERDICT_COMMENT_MARKER,
    VERDICTS,
    FileDiffMeta,
    ReviewComment,
    ReviewTriageInput,
    apply_confidence_gate,
    build_jev_review_triage_request,
    build_review_state,
    comment_already_posted,
    file_meta_from_patch,
    format_verdict_comment,
    github_review_event_for,
    load_jev_review_triage_config,
    parse_jev_review_triage_config,
    parse_verdict_answer,
    patch_sha256,
    post_verdict_comment_once,
    triage_pr_review,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.85,
        "model": "jev-latest",
        "timeout_seconds": 30,
        "max_diff_chars": 12000,
    }
    block.update(overrides)
    return {"code_review": {"jev_triage": block}}


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    return {
        "answers": {
            "verdict": {
                "choice": choice,
                "confidence": confidence,
            }
        }
    }


def _sample_payload(**overrides: Any) -> ReviewTriageInput:
    base = ReviewTriageInput(
        pr_number=62,
        title="Add Jev review triage",
        head_sha="abc123",
        base_ref="main",
        files=[
            FileDiffMeta(
                path="agent/jev_review_triage.py",
                patch_sha256="deadbeef",
                additions=40,
                deletions=2,
            )
        ],
        comments=[
            ReviewComment(
                body="Looks focused; add tests for low-confidence gate.",
                path="tests/agent/test_jev_review_triage.py",
                line=1,
                severity="suggestion",
            )
        ],
        diff_excerpt="diff --git a/agent/jev_review_triage.py b/agent/jev_review_triage.py\n+def triage_pr_review():\n",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class _FakeHttp:
    def __init__(self, payload: Any = None, *, error: Optional[BaseException] = None, status: int = 200):
        self.payload = payload if payload is not None else _choice_response("approve", 0.95)
        self.error = error
        self.status = status
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, headers: Dict[str, Any] = None, json: Dict[str, Any] = None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.error is not None:
            raise self.error
        resp = MagicMock()
        resp.status_code = self.status
        if self.status >= 400:
            resp.text = f"http {self.status}"
            resp.raise_for_status = MagicMock(side_effect=RuntimeError(f"http {self.status}"))
        else:
            resp.json = MagicMock(return_value=self.payload)
            resp.raise_for_status = MagicMock()
            resp.text = ""
        return resp


class TestJevReviewTriageHelpers:
    def test_parse_config_defaults_off(self):
        cfg = parse_jev_review_triage_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == 0.85
        assert load_jev_review_triage_config(None).enabled is False
        assert DEFAULT_CONFIG["code_review"]["jev_triage"]["enabled"] is False

    def test_build_request_is_choice_over_metadata(self):
        payload = _sample_payload()
        body = build_jev_review_triage_request(payload)
        q = body["questions"]["verdict"]
        assert q["type"] == "choice"
        assert q["instructions"] == CHOICE_INSTRUCTIONS
        assert q["criteria"] == CHOICE_CRITERIA
        state_blob = "\n".join(body["state"])
        assert "patch_sha256=deadbeef" in state_blob
        assert "Review comments:" in state_blob
        assert "Diff excerpt" in state_blob
        # Hashes/metadata pattern: path listed, not an unbounded raw file dump key.
        assert "agent/jev_review_triage.py" in state_blob

    def test_patch_hash_stable_and_file_meta(self):
        patch = "+line\n"
        assert patch_sha256(patch) == patch_sha256(patch)
        meta = file_meta_from_patch("a.py", patch, additions=1, deletions=0)
        assert meta.patch_sha256 == patch_sha256(patch)
        assert meta.path == "a.py"

    def test_diff_excerpt_bounded(self):
        huge = "x" * 50_000
        state = build_review_state(
            _sample_payload(diff_excerpt=huge),
            max_diff_chars=2_000,
        )
        joined = "\n".join(state)
        assert len(joined) < 10_000
        assert "truncated for Jev" in joined or "Diff excerpt" in joined

    def test_parse_verdict_and_aliases(self):
        assert parse_verdict_answer(_choice_response("approve", 0.9)) == ("approve", pytest.approx(0.9))
        assert parse_verdict_answer(_choice_response("request_changes", 0.8))[0] == "request-changes"
        assert parse_verdict_answer(_choice_response("needs_human", 0.5))[0] == "needs-human"

    def test_confidence_gate_never_keeps_low_approve(self):
        verdict, reason = apply_confidence_gate("approve", 0.5, 0.85)
        assert verdict == "needs-human"
        assert reason == "below_threshold"
        verdict, reason = apply_confidence_gate("request-changes", 0.9, 0.85)
        assert verdict == "request-changes"

    def test_github_event_never_approve(self):
        for v in VERDICTS:
            assert github_review_event_for(v) == "COMMENT"

    def test_comment_dedupe_and_format(self):
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config({"enabled": False}),
        )
        body = format_verdict_comment(result, pr_number=62)
        assert VERDICT_COMMENT_MARKER in body
        assert comment_already_posted([body]) is True
        posted: List[str] = []
        first = post_verdict_comment_once(
            existing_bodies=[],
            body=body,
            poster=posted.append,
        )
        second = post_verdict_comment_once(
            existing_bodies=posted,
            body=body,
            poster=posted.append,
        )
        assert first["posted"] is True
        assert second["posted"] is False
        assert second["reason"] == "already_posted"
        assert len(posted) == 1


class TestJevReviewTriageInvariants:
    def test_disabled_makes_no_jev_call(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("Jev must not be called when disabled")

        monkeypatch.setattr("agent.jev_review_triage._post_systemone", _boom)
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config({"enabled": False}),
        )
        assert result.should_post is False
        assert result.reason == "disabled"
        assert result.verdict == "needs-human"
        assert called["n"] == 0

    def test_low_confidence_approve_becomes_needs_human(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        http = _FakeHttp(_choice_response("approve", 0.4))
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            http_client=http,
            api_key="test-key",
        )
        assert result.verdict == "needs-human"
        assert result.raw_choice == "approve"
        assert result.reason == "below_threshold"
        assert result.is_approve is False
        assert len(http.calls) == 1

    def test_high_confidence_approve_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        http = _FakeHttp(_choice_response("approve", 0.95))
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            http_client=http,
            api_key="test-key",
        )
        assert result.verdict == "approve"
        assert result.confidence == pytest.approx(0.95)
        # Still never maps to GitHub APPROVE.
        assert github_review_event_for(result.verdict) == "COMMENT"

    def test_high_confidence_request_changes(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        http = _FakeHttp(_choice_response("request-changes", 0.91))
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            http_client=http,
            api_key="test-key",
        )
        assert result.verdict == "request-changes"

    def test_failure_falls_to_needs_human(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "test-key",
        )

        def _fail(*_a, **_k):
            raise RuntimeError("jev http 500")

        monkeypatch.setattr("agent.jev_review_triage._post_systemone", _fail)
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            api_key="test-key",
        )
        assert result.verdict == "needs-human"
        assert result.fallback is True
        assert result.is_approve is False

    def test_missing_key_needs_human(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "",
        )
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("no call without key")

        monkeypatch.setattr("agent.jev_review_triage._post_systemone", _boom)
        result = triage_pr_review(
            _sample_payload(),
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            api_key="",
        )
        assert result.verdict == "needs-human"
        assert result.reason == "missing_key"
        assert called["n"] == 0

    def test_empty_payload_needs_human(self, monkeypatch):
        monkeypatch.setattr(
            "agent.jev_review_triage.resolve_typesafe_api_key",
            lambda: "test-key",
        )
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("empty payload must not call Jev")

        monkeypatch.setattr("agent.jev_review_triage._post_systemone", _boom)
        empty = ReviewTriageInput(pr_number=1, title="empty")
        result = triage_pr_review(
            empty,
            cfg=parse_jev_review_triage_config(_enabled_cfg()["code_review"]["jev_triage"]),
            api_key="test-key",
        )
        assert result.verdict == "needs-human"
        assert result.reason == "empty_payload"
        assert called["n"] == 0
