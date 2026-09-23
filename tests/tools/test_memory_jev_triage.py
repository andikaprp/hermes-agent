"""Invariant tests for optional Jev noul triage before durable memory adds.

Contracts (not snapshots):
- disabled -> no Jev HTTP call; write happens
- Jev failure / missing key -> write happens (fail-open)
- below threshold -> skipped + logged; no durable write
- above threshold -> written
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.memory_jev_triage import (
    NOUL_CRITERIA,
    NOUL_INSTRUCTIONS,
    build_jev_memory_triage_request,
    load_jev_memory_triage_config,
    parse_jev_memory_triage_config,
    parse_noul_answer,
    triage_memory_add,
)
from tools.memory_tool import MemoryStore, memory_tool


DURABLE_ENTRY = (
    "User prefers Python 3.12 for all new projects and reviews PRs in Cursor."
)
assert len(DURABLE_ENTRY) >= 40


def _enabled_cfg(**overrides: Any) -> Dict[str, Any]:
    block = {
        "enabled": True,
        "threshold": 0.75,
        "model": "jev-latest",
        "timeout_seconds": 30,
        "min_chars": 40,
    }
    block.update(overrides)
    return {"memory": {"jev_triage": block}}


def _noul_response(noul: float) -> Dict[str, Any]:
    return {"answers": {"durable": {"noul": noul}}}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=2000, user_char_limit=1000)
    s.load_from_disk()
    return s


class TestJevMemoryTriageHelpers:
    def test_parse_config_defaults_off(self):
        cfg = parse_jev_memory_triage_config(None)
        assert cfg.enabled is False
        assert cfg.threshold == 0.75
        assert cfg.min_chars == 40
        assert load_jev_memory_triage_config(None).enabled is False
        assert DEFAULT_CONFIG["memory"]["jev_triage"]["enabled"] is False

    def test_build_request_is_noul(self):
        body = build_jev_memory_triage_request(DURABLE_ENTRY)
        q = body["questions"]["durable"]
        assert q["type"] == "noul"
        assert q["instructions"] == NOUL_INSTRUCTIONS
        assert q["criteria"] == NOUL_CRITERIA
        assert DURABLE_ENTRY not in body["state"][0]
        assert "hash=" in body["state"][0]
        assert "chars=" in body["state"][0]

    def test_parse_noul_answer(self):
        assert parse_noul_answer(_noul_response(0.81)) == pytest.approx(0.81)


class TestJevMemoryTriageInvariants:
    def test_disabled_makes_no_jev_call_and_writes(self, store, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("Jev must not be called when disabled")

        monkeypatch.setattr("tools.memory_jev_triage._post_systemone", _boom)
        monkeypatch.setattr(
            "tools.memory_jev_triage.load_jev_memory_triage_config",
            lambda *_a, **_k: parse_jev_memory_triage_config({"enabled": False}),
        )
        result = json.loads(memory_tool(action="add", content=DURABLE_ENTRY, store=store))
        assert result["success"] is True
        assert result.get("skipped") is not True
        assert DURABLE_ENTRY in store._entries_for("memory")
        assert called["n"] == 0

    def test_failure_falls_open_and_writes(self, store, monkeypatch):
        monkeypatch.setattr(
            "tools.memory_jev_triage.load_jev_memory_triage_config",
            lambda *_a, **_k: parse_jev_memory_triage_config(_enabled_cfg()["memory"]["jev_triage"]),
        )
        monkeypatch.setattr("tools.memory_jev_triage.resolve_typesafe_api_key", lambda: "test-key")

        def _fail(*_a, **_k):
            raise RuntimeError("jev http 500")

        monkeypatch.setattr("tools.memory_jev_triage._post_systemone", _fail)
        result = json.loads(memory_tool(action="add", content=DURABLE_ENTRY, store=store))
        assert result["success"] is True
        assert result.get("skipped") is not True
        assert DURABLE_ENTRY in store._entries_for("memory")

    def test_below_threshold_skips_write_and_logs(self, store, monkeypatch, caplog):
        monkeypatch.setattr(
            "tools.memory_jev_triage.load_jev_memory_triage_config",
            lambda *_a, **_k: parse_jev_memory_triage_config(_enabled_cfg()["memory"]["jev_triage"]),
        )
        monkeypatch.setattr("tools.memory_jev_triage.resolve_typesafe_api_key", lambda: "test-key")
        monkeypatch.setattr(
            "tools.memory_jev_triage._post_systemone",
            lambda *_a, **_k: (_noul_response(0.40), 1.0, 2.0),
        )
        with caplog.at_level(logging.INFO, logger="tools.memory_jev_triage"):
            result = json.loads(memory_tool(action="add", content=DURABLE_ENTRY, store=store))
        assert result["success"] is True
        assert result["skipped"] is True
        assert result["noul"] == pytest.approx(0.40)
        assert result["reason"] == "below_threshold"
        assert DURABLE_ENTRY not in store._entries_for("memory")
        assert any("jev_memory_triage" in r.getMessage() and "skipped=true" in r.getMessage() for r in caplog.records)

    def test_above_threshold_writes(self, store, monkeypatch):
        monkeypatch.setattr(
            "tools.memory_jev_triage.load_jev_memory_triage_config",
            lambda *_a, **_k: parse_jev_memory_triage_config(_enabled_cfg()["memory"]["jev_triage"]),
        )
        monkeypatch.setattr("tools.memory_jev_triage.resolve_typesafe_api_key", lambda: "test-key")
        monkeypatch.setattr(
            "tools.memory_jev_triage._post_systemone",
            lambda *_a, **_k: (_noul_response(0.92), 5.0, 6.0),
        )
        result = json.loads(memory_tool(action="add", content=DURABLE_ENTRY, store=store))
        assert result["success"] is True
        assert result.get("skipped") is not True
        assert DURABLE_ENTRY in store._entries_for("memory")

    def test_short_entry_prefilter_skips_jev(self, monkeypatch):
        called = {"n": 0}

        def _boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("short entries must not call Jev")

        monkeypatch.setattr("tools.memory_jev_triage._post_systemone", _boom)
        out = triage_memory_add(
            "ok",
            user_config=_enabled_cfg(),
            api_key="k",
        )
        assert out.allow_write is True
        assert out.reason == "prefilter_short"
        assert called["n"] == 0
