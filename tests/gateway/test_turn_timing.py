import logging

from gateway.turn_timing import TurnTiming


def test_timing_record_is_ordered_complete_and_redacted(caplog):
    ticks = iter((10.0, 10.1, 10.2, 10.4, 10.8, 11.0, 11.5, 11.7, 12.0, 12.2, 12.6, 13.0, 13.4, 14.0, 14.5, 15.0, 15.5))
    timing = TurnTiming(clock=lambda: next(ticks))
    for phase in timing.required_phases:
        timing.mark(phase)

    with caplog.at_level(logging.DEBUG, logger="gateway.timing"):
        timing.log_terminal()

    record = next(record for record in caplog.records if record.name == "gateway.timing")
    payload = record.turn_timing
    assert tuple(payload) == tuple(f"{phase}_ms" for phase in timing.required_phases)
    assert all(value >= 0 for value in payload.values())
    assert "telegram_quiet_window_ms" in record.getMessage()
    assert "final_delivery_ms" in record.getMessage()
    assert "secret-token" not in record.getMessage()
    assert "12345" not in record.getMessage()
    assert "https://private.example/path" not in record.getMessage()


def test_timing_carrier_clamps_backward_clock_and_exposes_no_metadata():
    ticks = iter((5.0, 4.0, 6.0, 7.0))
    timing = TurnTiming(clock=lambda: next(ticks))
    timing.mark("telegram_quiet_window")
    timing.mark("adapter_wait")
    timing.mark("queue_wait")

    assert timing.snapshot() == {
        "telegram_quiet_window_ms": 0,
        "adapter_wait_ms": 1000,
        "queue_wait_ms": 1000,
        "gateway_prep_ms": None,
        "cached_agent_model_resolution_ms": None,
        "context_memory_pre_llm_ms": None,
        "api_start_ms": None,
        "api_first_chunk_ms": None,
        "api_end_ms": None,
        "tool_time_ms": None,
        "final_delivery_ms": None,
    }
    assert not hasattr(timing, "message")
    assert not hasattr(timing, "session_id")


def test_finish_delivery_marks_and_logs_once(caplog):
    ticks = iter((1.0, 1.2, 1.5, 1.5))
    timing = TurnTiming(clock=lambda: next(ticks))
    timing.mark("telegram_quiet_window")
    with caplog.at_level(logging.DEBUG, logger="gateway.timing"):
        timing.finish_delivery()
        timing.finish_delivery()
        timing.log_terminal()
    records = [record for record in caplog.records if record.name == "gateway.timing"]
    assert len(records) == 1
    payload = records[0].turn_timing
    assert payload["telegram_quiet_window_ms"] == 200
    assert payload["final_delivery_ms"] == 300
    assert payload["adapter_wait_ms"] is None
