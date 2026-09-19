"""Regression: telegram_latency_samples parses current-day session= ready lines.

Live gateway.log lines look like
``response ready: platform=telegram chat=… session=… time=…s api_calls=N response=N chars``.
The sampler originally required chat= immediately followed by time=, so a head file that
only held the post-#6214769 format (and --date today) reported zero turns even when the
markers were present. Fixtures are redacted shapes copied from /workspace/hermes/logs.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "telegram_latency_samples.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "telegram_latency"


def _load_sampler():
    spec = importlib.util.spec_from_file_location("telegram_latency_samples", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sampler():
    return _load_sampler()


def test_re_ready_matches_session_field_and_legacy(sampler):
    modern = (
        "response ready: platform=telegram chat=redactedchat "
        "session=agent:main:telegram:dm:redactedchat time=0.9s api_calls=1 response=8 chars"
    )
    legacy = (
        "response ready: platform=telegram chat=redactedchat "
        "time=2.4s api_calls=1 response=10 chars"
    )
    m = sampler.RE_READY.search(modern)
    assert m is not None
    assert m.group(1) == "telegram" and m.group(2) == "redactedchat"
    assert m.group(3) == "0.9" and m.group(4) == "1" and m.group(5) == "8"
    m2 = sampler.RE_READY.search(legacy)
    assert m2 is not None and m2.group(3) == "2.4"


def test_date_2026_09_19_from_multi_day_head_file(sampler, tmp_path):
    """Head file spans Sep 18 evening + Sep 19; --date 2026-09-19 must find turns."""
    gw = FIXTURES / "gateway.log"
    ag = FIXTURES / "agent.log"
    cfg = FIXTURES / "config.yaml"
    scan = sampler.scan_logs([str(gw)], [str(ag)])
    turns = sampler.build_turns(scan)
    assert turns, "session= ready lines must parse"
    measured = [
        sampler.measure_turn(t, 2.0, 5.0) for t in turns if True
    ]
    day19 = [s for s in measured if s["date"] == "2026-09-19"]
    assert len(day19) >= 2
    assert any(s["quiet_window_clean_ms"] is not None for s in day19)


def test_expand_log_paths_pulls_rotated_siblings(sampler):
    paths = sampler.expand_log_paths([str(FIXTURES / "gateway.log")])
    names = {Path(p).name for p in paths}
    assert "gateway.log" in names
    assert "gateway.log.1" in names


def test_cli_date_2026_09_19_prints_per_class_summary():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--date", "2026-09-19",
            "--gateway-log", str(FIXTURES / "gateway.log"),
            "--agent-log", str(FIXTURES / "agent.log"),
            "--config", str(FIXTURES / "config.yaml"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "no telegram turns" not in proc.stderr
    assert "turns=" in proc.stdout
    assert "[single_batch]" in proc.stdout or "[all]" in proc.stdout


def test_cli_all_dates_includes_legacy_rotated_sibling():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--all-dates",
            "--gateway-log", str(FIXTURES / "gateway.log"),
            "--agent-log", str(FIXTURES / "agent.log"),
            "--config", str(FIXTURES / "config.yaml"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    # 1 legacy (log.1) + 1 evening Sep 18 + 2 Sep 19 = 4
    assert "turns=4" in proc.stdout
