"""Telegram root DMs stream by default, and the turn records time-to-first-content.

LAB-3: with no stream consumer the first outbound of a turn IS the turn-final send, so a DM
user sees nothing until the whole turn completes (measured p50 16.3 s / p90 50.6 s to first
send; 8 of 20 sampled turns sent exactly one message). These tests pin the decision — who
streams by default and who does not — and the two latency markers that make the win visible
in the logs at all.
"""

import logging

import pytest

from gateway.display_config import resolve_session_streaming
from gateway.stream_consumer_latency import (
    FIRST_DELTA_MARKER,
    FIRST_VISIBLE_MARKER,
    StreamLatencyMarks,
)


class _Clock:
    """Monotonic stand-in; tests advance it explicitly rather than sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _marks(clock: _Clock, logger: logging.Logger) -> StreamLatencyMarks:
    return StreamLatencyMarks(turn_id="t-1", chat_id="9876", clock=clock, marker_logger=logger)


def _marker_records(caplog) -> "list[dict]":
    return [r.stream_latency for r in caplog.records if hasattr(r, "stream_latency")]


# ── the default: who streams ──────────────────────────────────────────────


def test_telegram_root_dm_streams_with_no_configuration():
    """The regression this change exists for: the shipped default left DMs silent.

    ``streaming.enabled`` defaults to False, so before this change an operator who had
    configured nothing got no stream consumer and therefore no outbound until the turn ended.
    """
    assert resolve_session_streaming({}, "telegram", "dm", master_enabled=False) is True
    assert resolve_session_streaming({}, "telegram", "private", master_enabled=False) is True


def test_the_gateway_default_agrees_with_the_cli_default_it_never_merges():
    """The two default tables must not disagree about whether a Telegram DM streams.

    ``hermes_cli/config_defaults.py`` declares it on; the gateway resolves through this module
    with no DEFAULT_CONFIG merge, so a config.yaml that predates that declaration reaches the
    turn with nothing set. This asserts the relationship between the two, not either value.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    declared = (DEFAULT_CONFIG["display"]["platforms"].get("telegram") or {}).get("streaming")
    resolved = resolve_session_streaming({}, "telegram", "dm", master_enabled=False)
    assert resolved is bool(declared)


def test_telegram_group_is_not_swept_into_the_dm_default():
    """Progressive edits in a shared channel are noise and burn the same flood budget."""
    assert resolve_session_streaming({}, "telegram", "group", master_enabled=False) is False
    assert resolve_session_streaming({}, "telegram", "supergroup", master_enabled=False) is False


def test_master_switch_still_reaches_every_chat_type():
    """The DM default adds a floor; it must not become a ceiling for the global switch."""
    assert resolve_session_streaming({}, "telegram", "group", master_enabled=True) is True


@pytest.mark.parametrize("chat_type", ["dm", "group"])
def test_explicit_per_platform_opt_out_wins_over_the_dm_default(chat_type):
    config = {"display": {"platforms": {"telegram": {"streaming": False}}}}
    assert resolve_session_streaming(config, "telegram", chat_type, master_enabled=True) is False


def test_transport_off_suppresses_the_dm_default():
    """``transport: off`` is an operator saying no; only an explicit opt-in may override it."""
    assert resolve_session_streaming(
        {}, "telegram", "dm", master_enabled=False, transport="off") is False
    assert resolve_session_streaming(
        {"display": {"platforms": {"telegram": {"streaming": True}}}},
        "telegram", "dm", master_enabled=False, transport="off") is True


def test_other_platforms_keep_their_existing_resolution():
    """Only Telegram DMs gained a default; tier values and the master switch are untouched."""
    for platform in ("slack", "discord", "matrix"):
        assert resolve_session_streaming({}, platform, "dm", master_enabled=False) is False
        assert resolve_session_streaming({}, platform, "dm", master_enabled=True) is True
    # A tier that pins the value keeps pinning it, in both directions.
    assert resolve_session_streaming({}, "signal", "dm", master_enabled=True) is False
    assert resolve_session_streaming({}, "wecom", "group", master_enabled=False) is True


# ── the markers: making the win measurable ────────────────────────────────


def test_first_delta_and_first_visible_are_emitted_with_the_gap_between_them(caplog):
    clock = _Clock()
    logger = logging.getLogger("test.stream_latency.pair")
    with caplog.at_level(logging.INFO, logger=logger.name):
        marks = _marks(clock, logger)
        clock.now += 6.6  # provider think time
        marks.note_delta()
        clock.now += 0.2  # consumer debounce + transport
        marks.note_visible()

    delta, visible = _marker_records(caplog)
    assert delta["marker"] == FIRST_DELTA_MARKER
    assert delta["since_open_ms"] == 6600.0
    assert visible["marker"] == FIRST_VISIBLE_MARKER
    assert visible["since_open_ms"] == 6800.0
    # The consumer's own contribution, separable from the provider's.
    assert visible["since_first_delta_ms"] == 200.0


def test_each_marker_emits_at_most_once_per_turn(caplog):
    clock = _Clock()
    logger = logging.getLogger("test.stream_latency.once")
    with caplog.at_level(logging.INFO, logger=logger.name):
        marks = _marks(clock, logger)
        for _ in range(3):
            clock.now += 1.0
            marks.note_delta()
            marks.note_visible()

    markers = [r["marker"] for r in _marker_records(caplog)]
    assert markers == [FIRST_DELTA_MARKER, FIRST_VISIBLE_MARKER]


def test_markers_never_publish_a_raw_chat_id(caplog):
    logger = logging.getLogger("test.stream_latency.redaction")
    with caplog.at_level(logging.INFO, logger=logger.name):
        marks = _marks(_Clock(), logger)
        marks.note_delta()
        marks.note_visible()

    for record in _marker_records(caplog):
        assert "9876" not in record["chat"]
    assert "9876" not in caplog.text


def test_a_visible_send_with_no_preceding_delta_omits_the_gap(caplog):
    """Interim commentary can reach the platform before any model token does."""
    logger = logging.getLogger("test.stream_latency.no_delta")
    with caplog.at_level(logging.INFO, logger=logger.name):
        marks = _marks(_Clock(), logger)
        marks.note_visible()

    (visible,) = _marker_records(caplog)
    assert "since_first_delta_ms" not in visible


def test_a_logging_failure_cannot_break_a_live_turn():
    class _Exploding(logging.Logger):
        def info(self, *args, **kwargs):
            raise RuntimeError("log sink down")

    marks = StreamLatencyMarks(turn_id="t", chat_id="1", marker_logger=_Exploding("boom"))
    marks.note_delta()
    marks.note_visible()
    assert marks.first_delta_seen and marks.first_visible_seen
