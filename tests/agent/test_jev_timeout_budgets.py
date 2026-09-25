"""Per-surface Jev timeout budgets. These defaults stall a turn at 30s today."""

from __future__ import annotations


def test_routing_budget_is_4s():
    from gateway.run_turn_jev_routing import (
        DEFAULT_TIMEOUT_SECONDS,
        parse_jev_routing_config,
    )

    assert DEFAULT_TIMEOUT_SECONDS == 4.0
    assert parse_jev_routing_config(None).timeout_seconds == 4.0


def test_quality_gate_stays_tighter_than_4s():
    """Fast-lane gate is already bounded; do not raise it to the 4s suggestion."""
    from gateway.run_turn_fast_lane_quality_gate import (
        DEFAULT_TIMEOUT_SECONDS,
        parse_quality_gate_config,
    )

    assert DEFAULT_TIMEOUT_SECONDS == 1.5
    assert parse_quality_gate_config(None).timeout_seconds == 1.5
    assert DEFAULT_TIMEOUT_SECONDS <= 4.0


def test_memory_skill_delegate_budgets_are_5s():
    from agent.skill_routing_jev import (
        DEFAULT_TIMEOUT_SECONDS as skill_timeout,
        parse_jev_skill_routing_config,
    )
    from tools.delegate_tool_jev import (
        DEFAULT_TIMEOUT_SECONDS as delegate_timeout,
        parse_jev_check_config,
    )
    from tools.memory_jev_triage import (
        DEFAULT_TIMEOUT_SECONDS as memory_timeout,
        parse_jev_memory_triage_config,
    )

    assert memory_timeout == 5.0
    assert skill_timeout == 5.0
    assert delegate_timeout == 5.0
    assert parse_jev_memory_triage_config(None).timeout_seconds == 5.0
    assert parse_jev_skill_routing_config(None).timeout_seconds == 5.0
    assert parse_jev_check_config(None).timeout_seconds == 5.0


def test_completion_budget_is_10s():
    from gateway.run_turn_jev_completion import (
        DEFAULT_TIMEOUT_SECONDS,
        parse_jev_completion_config,
    )

    assert DEFAULT_TIMEOUT_SECONDS == 10.0
    assert parse_jev_completion_config(None).timeout_seconds == 10.0


def test_review_triage_budget_is_15s():
    from agent.jev_review_triage import (
        DEFAULT_TIMEOUT_SECONDS,
        parse_jev_review_triage_config,
    )

    assert DEFAULT_TIMEOUT_SECONDS == 15.0
    assert parse_jev_review_triage_config(None).timeout_seconds == 15.0


def test_action_gate_stays_at_3s():
    from gateway.jev_action_gate import DEFAULT_TIMEOUT_SECONDS

    assert DEFAULT_TIMEOUT_SECONDS == 3.0


def test_config_defaults_match_budgets_and_cache():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["telegram"]["jev_routing"]["timeout_seconds"] == 4
    assert (
        DEFAULT_CONFIG["gateway"]["telegram"]["fast_lane"]["quality_gate"][
            "timeout_seconds"
        ]
        == 1.5
    )
    assert DEFAULT_CONFIG["memory"]["jev_triage"]["timeout_seconds"] == 5
    assert DEFAULT_CONFIG["agent"]["skill_routing"]["timeout_seconds"] == 5
    assert DEFAULT_CONFIG["delegation"]["jev_check"]["timeout_seconds"] == 5
    assert DEFAULT_CONFIG["gateway"]["jev_completion"]["timeout_seconds"] == 10
    assert DEFAULT_CONFIG["code_review"]["jev_triage"]["timeout_seconds"] == 15
    cache = DEFAULT_CONFIG["jev"]["decision_cache"]
    assert cache["enabled"] is True
    assert cache["ttl_seconds"] == 3600
    assert cache["max_entries"] == 512
