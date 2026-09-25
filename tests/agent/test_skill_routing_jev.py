"""Invariant tests for optional Jev skill selection at session setup.

Contracts (not snapshots):
- config off -> no Jev HTTP call; default auto_load unchanged
- failure / missing key / below threshold -> current loader behavior
- high-confidence pick that differs from default -> Jev pick used
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agent.skill_routing_jev import (
    CHOICE_INSTRUCTIONS,
    NONE_CHOICE,
    SkillCandidate,
    build_jev_skill_routing_request,
    load_jev_skill_routing_config,
    maybe_override_auto_load_skills,
    parse_jev_skill_routing_config,
    parse_skill_answer,
)


CANDIDATES = [
    SkillCandidate("alpha", "Alpha skill.", "Working on alpha tasks"),
    SkillCandidate("beta", "Beta skill.", "Working on beta tasks"),
]


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.85,
        "model": "jev-latest",
        "timeout_seconds": 30,
    }
    block.update(overrides)
    return {"agent": {"skill_routing": block}, "skills": {"auto_load": ["alpha"]}}


def _choice_response(choice: str, confidence: float) -> Dict[str, Any]:
    return {
        "answers": {
            "skill": {
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: confidence},
            }
        }
    }


class _FakeHttp:
    def __init__(self, payload: Any = None, *, error: Optional[BaseException] = None, status: int = 200):
        self.payload = payload if payload is not None else _choice_response("beta", 0.95)
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


class TestConfigParsing:
    def test_defaults_disabled(self):
        cfg = parse_jev_skill_routing_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == 0.85
        assert load_jev_skill_routing_config({}).enabled is False


class TestJevSkillRoutingInvariants:
    def test_disabled_makes_no_jev_call(self):
        http = _FakeHttp()
        result = maybe_override_auto_load_skills(
            "please use beta for this",
            default_names=["alpha"],
            user_config={"agent": {"skill_routing": {"enabled": False}}},
            http_client=http,
            api_key="test-key",
            candidates=CANDIDATES,
        )
        assert result is None
        assert http.calls == []

    def test_failure_keeps_default_loader(self):
        http = _FakeHttp(error=RuntimeError("jev http 500"))
        result = maybe_override_auto_load_skills(
            "please use beta for this",
            default_names=["alpha"],
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="test-key",
            candidates=CANDIDATES,
        )
        assert result is None
        assert len(http.calls) == 1

    def test_high_confidence_differing_pick_overrides_default(self):
        http = _FakeHttp(_choice_response("beta", 0.95))
        result = maybe_override_auto_load_skills(
            "please use beta for this",
            default_names=["alpha"],
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="test-key",
            candidates=CANDIDATES,
        )
        assert result == ["beta"]
        assert len(http.calls) == 1
        body = http.calls[0]["json"]
        assert body["questions"]["skill"]["type"] == "choice"
        assert body["questions"]["skill"]["instructions"] == CHOICE_INSTRUCTIONS
        assert NONE_CHOICE in body["questions"]["skill"]["criteria"]
        assert "alpha" in body["questions"]["skill"]["criteria"]
        assert any("Inbound task" in s for s in body["state"])

    def test_low_confidence_keeps_default(self):
        http = _FakeHttp(_choice_response("beta", 0.40))
        result = maybe_override_auto_load_skills(
            "ambiguous task",
            default_names=["alpha"],
            user_config=_enabled_cfg(threshold=0.85),
            http_client=http,
            api_key="test-key",
            candidates=CANDIDATES,
        )
        assert result is None

    def test_high_confidence_none_clears_auto_load(self):
        http = _FakeHttp(_choice_response(NONE_CHOICE, 0.99))
        result = maybe_override_auto_load_skills(
            "just say hi",
            default_names=["alpha"],
            user_config=_enabled_cfg(),
            http_client=http,
            api_key="test-key",
            candidates=CANDIDATES,
        )
        assert result == []

    def test_request_shape_and_parse(self):
        body = build_jev_skill_routing_request("do the thing", CANDIDATES)
        assert body["questions"]["skill"]["type"] == "choice"
        choice, conf = parse_skill_answer(
            _choice_response("alpha", 0.9),
            valid_ids={"alpha", "beta", NONE_CHOICE},
        )
        assert choice == "alpha" and conf == pytest.approx(0.9)
