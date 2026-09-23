"""LAB-59: native Jev routing observability (metadata-only local feed).

Contracts:
- mode off (default) records nothing
- shadow/on record tier/model/confidence/latency/reason + content_hash only
- recorded rows never contain raw prompt / tool text
- routing log path feeds the same store used by /api/jev/decisions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gateway.jev_observability import (
    load_jev_observability_config,
    parse_jev_observability_config,
    read_recent_decisions,
    record_jev_decision,
    reset_observability_for_tests,
)
from gateway.run_turn_jev_routing import (
    build_jev_routing_request,
    maybe_jev_route_uncertain,
)
from agent.jev_payload_hygiene import content_hash, tool_result_metadata
from agent.context_compressor_jev import message_to_state_text


def _obs_cfg(mode: str = "on", **overrides: Any) -> Dict[str, Any]:
    block = {"mode": mode, "limit": 50}
    block.update(overrides)
    return {"gateway": {"jev_observability": block, "telegram": {"jev_routing": {
        "enabled": True,
        "threshold": 0.85,
        "model": "jev-latest",
        "timeout_seconds": 30,
    }, "fast_path": True}}}


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    return {
        "answers": {
            "route": {
                "choice": choice,
                "confidence": confidence,
                "probabilities": {
                    choice: confidence,
                    "task" if choice == "lane" else "lane": 1.0 - confidence,
                },
            }
        }
    }


class _FakeHttp:
    def __init__(self, payload: Any = None):
        self.payload = payload if payload is not None else _choice_response("lane", 0.95)
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, headers: Dict[str, Any] = None, json: Dict[str, Any] = None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self.payload
        return resp

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean_ring():
    reset_observability_for_tests()
    yield
    reset_observability_for_tests()


class TestJevObservabilityConfig:
    def test_default_off(self):
        cfg = parse_jev_observability_config(None)
        assert cfg.mode == "off"
        loaded = load_jev_observability_config({})
        assert loaded.mode == "off"

    def test_modes(self):
        assert parse_jev_observability_config({"mode": "shadow"}).mode == "shadow"
        assert parse_jev_observability_config({"mode": "on"}).mode == "on"
        assert parse_jev_observability_config({"mode": "nope"}).mode == "off"


class TestJevObservabilityRecording:
    def test_off_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record_jev_decision(
            kind="routing",
            tier="lane",
            model="jev-latest",
            confidence=0.9,
            latency_ms=12.5,
            reason="ok",
            content_hash_value="abc",
            user_config=_obs_cfg("off"),
        )
        feed = read_recent_decisions(user_config=_obs_cfg("off"))
        assert feed["enabled"] is False
        assert feed["decisions"] == []
        assert not (tmp_path / "logs" / "jev-decisions.jsonl").exists()

    def test_shadow_records_metadata_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        secret = "please merge the PR with sk-abcdefghijklmnopqrstuvwxyz"
        record_jev_decision(
            kind="routing",
            tier="lane",
            model="jev-latest",
            confidence=0.91,
            latency_ms=42.0,
            reason="ok",
            content_hash_value=content_hash(secret),
            user_config=_obs_cfg("shadow"),
        )
        feed = read_recent_decisions(user_config=_obs_cfg("shadow"))
        assert feed["enabled"] is True
        assert len(feed["decisions"]) == 1
        row = feed["decisions"][0]
        assert row["tier"] == "lane"
        assert row["model"] == "jev-latest"
        assert row["confidence"] == pytest.approx(0.91)
        assert row["latency_ms"] == pytest.approx(42.0)
        assert row["content_hash"] == content_hash(secret)
        blob = json.dumps(row)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in blob
        assert "please merge" not in blob
        path = Path(feed["path"])
        assert path.exists()
        assert "sk-" not in path.read_text(encoding="utf-8")

    def test_routing_hook_feeds_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "gateway.jev_observability.load_jev_observability_config",
            lambda user_config=None: parse_jev_observability_config({"mode": "on", "limit": 50}),
        )
        http = _FakeHttp(_choice_response("lane", 0.95))
        result = maybe_jev_route_uncertain(
            "alright then",
            user_config=_obs_cfg("on"),
            chat_id=123,
            http_client=http,
            api_key="test-key",
        )
        assert result == "social"
        feed = read_recent_decisions(limit=10, user_config=_obs_cfg("on"))
        assert feed["enabled"] is True
        assert any(d.get("kind") == "routing" and d.get("tier") == "lane" for d in feed["decisions"])
        for d in feed["decisions"]:
            assert "alright then" not in json.dumps(d)
        assert any(d.get("latency_ms") is not None for d in feed["decisions"])
        assert any(d.get("content_hash") for d in feed["decisions"])


class TestJevPayloadHygiene:
    def test_routing_request_is_hash_metadata_only(self):
        secret = "hi «redacted:sk-…» " + ("x" * 500)
        body = build_jev_routing_request(secret)
        state = "\n".join(body["state"])
        assert "«redacted:sk-…" not in state
        assert "[secret]" not in state
        assert "x" * 40 not in state
        assert f"hash={content_hash(secret)}" in state
        assert "chars=" in state
        assert len(state) < 900

    def test_tool_results_are_hash_metadata_only(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "SECRET_TOOL_OUTPUT with password=hunter2 and dump",
        }
        flat = message_to_state_text(msg)
        assert "SECRET_TOOL_OUTPUT" not in flat
        assert "hunter2" not in flat
        assert "hash=" in flat
        assert tool_result_metadata("abc", tool_call_id="t").startswith("[tool t]")
