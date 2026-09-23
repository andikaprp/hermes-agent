"""LAB-59: every Jev request builder sends hashes/metadata only.

Red-on-base: the six builders still embed redacted prompt/tool text, the
decision jsonl has no max-bytes rollover, and the store contract is not
documented as the decision log (routes are not a second telemetry sink).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.context_compressor_jev import build_fanout_request, message_to_state_text
from agent.jev_payload_hygiene import content_hash
from agent.skill_routing_jev import SkillCandidate, build_jev_skill_routing_request
from gateway.jev_observability import (
    read_recent_decisions,
    record_jev_decision,
    reset_observability_for_tests,
)
from gateway.run_turn_fast_lane_quality_gate import build_quality_gate_request
from gateway.run_turn_jev_routing import build_jev_routing_request
from tools.delegate_tool_jev import build_delegate_choice_request
from tools.memory_jev_triage import build_jev_memory_triage_request

# Unique sentinels — must not be substrings of frozen instructions or lexicon.
INBOUND = "LAB59_INBOUND_PROMPT_zqx9_do_not_leak"
PRIOR = "LAB59_PRIOR_USER_TURN_zqx9_do_not_leak"
ENTRY = "LAB59_MEMORY_ENTRY_zqx9_do_not_leak durable fact about the user"
GOAL = "LAB59_DELEGATE_GOAL_zqx9_do_not_leak"
CONTEXT = "LAB59_DELEGATE_CONTEXT_zqx9_do_not_leak"
TASK = "LAB59_SKILL_TASK_zqx9_do_not_leak"
USER_MSG = "LAB59_QUALITY_USER_zqx9_do_not_leak"
DRAFT = "LAB59_QUALITY_DRAFT_zqx9_do_not_leak"
USER_TURN = "LAB59_COMPRESS_USER_zqx9_do_not_leak"
ASSISTANT = "LAB59_COMPRESS_ASSISTANT_zqx9_do_not_leak"
TOOL_OUT = "LAB59_TOOL_RESULT_zqx9_do_not_leak"
TOOL_ARGS = "LAB59_TOOL_ARGS_zqx9_do_not_leak"

_PROMPT_KEYS = frozenset({
    "prompt",
    "text",
    "raw",
    "raw_text",
    "content",
    "message",
    "goal",
    "arguments",
    "draft",
    "draft_reply",
    "user_message",
    "entry",
    "context",
    "tool_result",
})


def _obs_cfg(mode: str = "on") -> dict:
    return {"gateway": {"jev_observability": {"mode": mode, "limit": 50}}}


def assert_hashes_only(body: Any, *secrets: str) -> None:
    """Fail if raw prompt/tool text or a prompt-shaped key reaches the builder."""
    blob = json.dumps(body)
    _reject_prompt_keys(body)
    for secret in secrets:
        if secret and secret in blob:
            raise AssertionError(f"raw text reached Jev request: {secret[:40]!r}")
    for secret in secrets:
        if not secret:
            continue
        digest = content_hash(secret)
        if f"hash={digest}" not in blob:
            raise AssertionError(f"missing content hash for {secret[:40]!r}")
    if "chars=" not in blob:
        raise AssertionError("metadata missing chars=")
    if "words=" not in blob and "flags=" not in blob:
        raise AssertionError("metadata missing counts/flags")


def _reject_prompt_keys(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _PROMPT_KEYS:
                raise AssertionError(f"prompt key {path}.{key} reached Jev request")
            _reject_prompt_keys(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _reject_prompt_keys(value, f"{path}[{i}]")


class TestLeakAssertionIsNotVacuous:
    def test_hand_built_prompt_key_is_rejected(self):
        secret = INBOUND
        bad = {"prompt": secret, "state": [f"hash={content_hash(secret)} chars=1 words=1"]}
        with pytest.raises(AssertionError, match="prompt key"):
            assert_hashes_only(bad, secret)

    def test_hand_built_raw_state_is_rejected(self):
        secret = INBOUND
        bad = {"state": [secret], "questions": {}}
        with pytest.raises(AssertionError, match="raw text"):
            assert_hashes_only(bad, secret)


class TestCallSitePayloadHygiene:
    def test_routing_request_is_hash_metadata_only(self):
        body = build_jev_routing_request(
            INBOUND,
            history=[{"role": "user", "content": PRIOR}],
        )
        assert_hashes_only(body, INBOUND, PRIOR)
        assert "prompt" not in body

    def test_memory_triage_request_is_hash_metadata_only(self):
        body = build_jev_memory_triage_request(ENTRY, target="memory")
        assert_hashes_only(body, ENTRY)

    def test_delegate_request_is_hash_metadata_only(self):
        body = build_delegate_choice_request(
            [{"goal": GOAL, "context": "task-local " + CONTEXT}],
            context=CONTEXT,
        )
        assert_hashes_only(body, GOAL, CONTEXT)

    def test_skill_routing_request_is_hash_metadata_only(self):
        candidates = [SkillCandidate("alpha", "Alpha skill.", "alpha work")]
        body = build_jev_skill_routing_request(TASK, candidates)
        assert_hashes_only(body, TASK)
        # Skill ids stay as choice criteria; the task text does not.
        assert "alpha" in body["questions"]["skill"]["criteria"]

    def test_quality_gate_request_is_hash_metadata_only(self):
        body = build_quality_gate_request(USER_MSG, DRAFT)
        assert_hashes_only(body, USER_MSG, DRAFT)

    def test_compressor_state_is_hash_metadata_only(self):
        args_blob = json.dumps({"command": TOOL_ARGS})
        messages = [
            {"role": "user", "content": USER_TURN},
            {
                "role": "assistant",
                "content": ASSISTANT,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": args_blob,
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": TOOL_OUT,
            },
        ]
        flat_user = message_to_state_text(messages[0])
        flat_assistant = message_to_state_text(messages[1])
        flat_tool = message_to_state_text(messages[2])
        assert_hashes_only({"state": [flat_user]}, USER_TURN)
        assert_hashes_only({"state": [flat_assistant]}, ASSISTANT, args_blob)
        assert TOOL_ARGS not in flat_assistant
        assert_hashes_only({"state": [flat_tool]}, TOOL_OUT)
        body = build_fanout_request(messages, batch_size=40)
        assert_hashes_only(body, USER_TURN, ASSISTANT, args_blob, TOOL_OUT)
        assert TOOL_ARGS not in json.dumps(body)
        assert len(body["state"]) == 3


class TestDecisionLogRotation:
    @pytest.fixture(autouse=True)
    def _clean(self):
        reset_observability_for_tests()
        yield
        reset_observability_for_tests()

    def test_jsonl_rolls_over_to_dot_1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import gateway.jev_observability as obs

        assert hasattr(obs, "DECISIONS_MAX_BYTES"), "stdlib rotation max-bytes missing"
        monkeypatch.setattr(obs, "DECISIONS_MAX_BYTES", 64)
        hashes = []
        for i in range(4):
            digest = f"h{i:02d}" + ("a" * 12)
            hashes.append(digest)
            row = record_jev_decision(
                kind="routing",
                tier="lane",
                model="jev-latest",
                confidence=0.9,
                latency_ms=1.0,
                reason="ok",
                content_hash_value=digest,
                user_config=_obs_cfg("on"),
            )
            assert row is not None
        current = tmp_path / "logs" / "jev-decisions.jsonl"
        rolled = Path(str(current) + ".1")
        assert rolled.exists(), "expected .1 rollover"
        assert not Path(str(current) + ".2").exists(), "backupCount must be 1"
        current_text = current.read_text(encoding="utf-8")
        rolled_text = rolled.read_text(encoding="utf-8")
        assert hashes[-1] in current_text
        assert hashes[-2] in rolled_text
        assert hashes[-1] not in rolled_text
        # Newest row still readable from the decision store after rollover.
        feed = read_recent_decisions(limit=10, user_config=_obs_cfg("on"))
        assert any(d.get("content_hash") == hashes[-1] for d in feed["decisions"])


class TestDecisionStoreContract:
    def test_jsonl_is_the_decision_store_routes_are_local_views(self):
        import gateway.jev_observability as obs

        doc = obs.__doc__ or ""
        assert "THE decision store" in doc
        assert "read-only local view" in doc
        assert "nothing is sent" in doc.lower() or "Nothing is sent" in doc
        readme = Path(__file__).resolve().parents[2] / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "jev-decisions.jsonl is the decision store" in text
        assert "read-only local view" in text
