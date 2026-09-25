"""Keep-alive System One client, exact-repeat cache, and fail-safe contract.

No network. A stub connection stands in for http.client.HTTPSConnection.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, List, Optional

import pytest

from agent.jev_payload_hygiene import text_metadata
from agent.jev_transport import (
    configure_decision_cache,
    decision_cache_key,
    get_persistent_client,
    post_systemone,
    reset_jev_transport_for_tests,
    set_connection_factory_for_tests,
)
from agent.context_compressor_jev import _post_systemone


def _body(text: str = "hello", *, question: str = "route") -> Dict[str, Any]:
    return {
        "model": "jev-latest",
        "state": [text_metadata(text, label="Inbound message")],
        "questions": {
            question: {
                "type": "choice",
                "instructions": "pick one",
                "criteria": {"lane": "fast", "task": "full"},
            }
        },
    }


class _Resp:
    def __init__(self, status: int, payload: Any, *, connection: str = "keep-alive"):
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")
        self._connection = connection

    def read(self) -> bytes:
        return self._raw

    def getheader(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if name.lower() == "connection":
            return self._connection
        return default


class _Conn:
    def __init__(self, timeout: float = 0.0, *, script: Optional[List[Any]] = None):
        self.timeout = timeout
        self.script = list(script or [])
        self.requests: List[Dict[str, Any]] = []
        self.closed = False
        self._pending: Any = None

    def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> None:
        self.requests.append(
            {"method": method, "path": path, "body": body, "headers": headers, "timeout": self.timeout}
        )
        if not self.script:
            self._pending = _Resp(200, {"answers": {"route": {"choice": "lane"}}})
            return
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        self._pending = step

    def getresponse(self) -> _Resp:
        if isinstance(self._pending, BaseException):
            raise self._pending
        return self._pending

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_transport():
    reset_jev_transport_for_tests()
    configure_decision_cache(enabled=True, ttl_seconds=3600, max_entries=512)
    yield
    reset_jev_transport_for_tests()


def _install(script_per_conn: Optional[List[List[Any]]] = None) -> List[_Conn]:
    made: List[_Conn] = []
    plans = list(script_per_conn or [])

    def factory(timeout: float) -> _Conn:
        plan = plans.pop(0) if plans else []
        conn = _Conn(timeout, script=plan)
        made.append(conn)
        return conn

    set_connection_factory_for_tests(factory)
    return made


def test_connection_reused_across_distinct_payloads():
    made = _install()
    first, _, _ = post_systemone(_body("one"), api_key="k", timeout_seconds=4.0)
    second, _, _ = post_systemone(_body("two"), api_key="k", timeout_seconds=4.0)
    assert first["answers"]["route"]["choice"] == "lane"
    assert second["answers"]["route"]["choice"] == "lane"
    assert len(made) == 1
    assert len(made[0].requests) == 2
    assert get_persistent_client().opens == 1
    assert made[0].closed is False


def test_stale_connection_retries_once_on_fresh_socket():
    made = _install(
        [
            [ConnectionResetError("stale")],
            [],
        ]
    )
    data, _, _ = post_systemone(_body("retry"), api_key="k", timeout_seconds=4.0)
    assert data["answers"]["route"]["choice"] == "lane"
    assert len(made) == 2
    assert made[0].closed is True
    assert get_persistent_client().opens == 2


def test_second_connection_error_is_not_retried():
    made = _install(
        [
            [ConnectionResetError("first")],
            [ConnectionResetError("second")],
        ]
    )
    with pytest.raises(ConnectionResetError):
        post_systemone(_body("fail"), api_key="k", timeout_seconds=4.0)
    assert len(made) == 2
    assert get_persistent_client().opens == 2


def test_timeout_is_applied_and_not_retried():
    made = _install([[socket.timeout("timed out")]])
    t0 = time.perf_counter()
    with pytest.raises(socket.timeout):
        post_systemone(_body("slow"), api_key="k", timeout_seconds=4.0)
    assert time.perf_counter() - t0 < 1.0
    assert len(made) == 1
    assert made[0].requests[0]["timeout"] == 4.0


def test_slow_stub_is_cut_by_call_budget():
    def factory(timeout: float) -> _Conn:
        conn = _Conn(timeout)

        def request(method: str, path: str, body: Any = None, headers: Any = None) -> None:
            conn.requests.append({"timeout": conn.timeout})
            time.sleep(5)

        conn.request = request  # type: ignore[method-assign]
        return conn

    set_connection_factory_for_tests(factory)
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        post_systemone(_body("budget"), api_key="k", timeout_seconds=0.2)
    assert time.perf_counter() - t0 < 1.0


def test_http_error_is_not_retried():
    made = _install([[_Resp(429, {"error": "slow"})]])
    with pytest.raises(RuntimeError, match="429"):
        post_systemone(_body("rate"), api_key="k", timeout_seconds=4.0)
    assert len(made) == 1


def test_cache_hit_skips_transport():
    made = _install()
    body = _body("same")
    first, _, _ = post_systemone(body, api_key="k", timeout_seconds=4.0)
    second, ttft, ready = post_systemone(dict(body), api_key="k", timeout_seconds=4.0)
    assert second == first
    assert len(made) == 1
    assert len(made[0].requests) == 1
    assert ready < 50


def test_cache_key_is_content_hash_plus_question_spec():
    body = _body("keyed")
    key = decision_cache_key(body)
    digest = body["state"][0].split("hash=")[1].split()[0]
    assert digest in key
    assert "pick one" in key
    other = _body("keyed", question="other")
    assert decision_cache_key(other) != key


def test_different_question_misses_cache():
    made = _install()
    post_systemone(_body("q", question="route"), api_key="k", timeout_seconds=4.0)
    post_systemone(_body("q", question="other"), api_key="k", timeout_seconds=4.0)
    assert len(made[0].requests) == 2


def test_ttl_expiry_calls_transport_again(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("agent.jev_transport.time.monotonic", lambda: clock["t"])
    configure_decision_cache(enabled=True, ttl_seconds=10, max_entries=512)
    made = _install()
    body = _body("ttl")
    post_systemone(body, api_key="k", timeout_seconds=4.0)
    clock["t"] = 1005.0
    post_systemone(body, api_key="k", timeout_seconds=4.0)
    assert len(made[0].requests) == 1
    clock["t"] = 1011.0
    post_systemone(body, api_key="k", timeout_seconds=4.0)
    assert len(made[0].requests) == 2


def test_cache_can_be_disabled():
    configure_decision_cache(enabled=False, ttl_seconds=3600, max_entries=512)
    made = _install()
    body = _body("off")
    post_systemone(body, api_key="k", timeout_seconds=4.0)
    post_systemone(body, api_key="k", timeout_seconds=4.0)
    assert len(made[0].requests) == 2


def test_cache_is_lru_bounded():
    configure_decision_cache(enabled=True, ttl_seconds=3600, max_entries=2)
    made = _install()
    post_systemone(_body("a"), api_key="k", timeout_seconds=4.0)
    post_systemone(_body("b"), api_key="k", timeout_seconds=4.0)
    post_systemone(_body("c"), api_key="k", timeout_seconds=4.0)
    post_systemone(_body("a"), api_key="k", timeout_seconds=4.0)
    assert len(made[0].requests) == 4


def test_post_systemone_uses_persistent_client_when_no_http_client():
    made = _install()
    data, _, _ = _post_systemone(_body("shared"), api_key="k", timeout_seconds=4.0)
    assert data["answers"]["route"]["choice"] == "lane"
    assert len(made) == 1
    _post_systemone(_body("shared-2"), api_key="k", timeout_seconds=4.0)
    assert len(made) == 1
    assert len(made[0].requests) == 2


def test_injected_http_client_is_unchanged():
    class _Http:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers=None, json=None):
            self.calls += 1
            resp = type("R", (), {})()
            resp.status_code = 200
            resp.json = lambda: {"answers": {"route": {"choice": "task"}}}
            return resp

        def close(self):
            raise AssertionError("injected client must not be closed")

    http = _Http()
    data, _, _ = _post_systemone(
        _body("injected"), api_key="k", timeout_seconds=4.0, http_client=http,
    )
    assert data["answers"]["route"]["choice"] == "task"
    assert http.calls == 1
    assert get_persistent_client().opens == 0


def test_routing_falls_back_to_normal_lane_on_transport_failure():
    from gateway.run_turn_jev_routing import maybe_jev_route_uncertain

    _install([[ConnectionResetError("down")], [ConnectionResetError("still down")]])
    verdict = maybe_jev_route_uncertain(
        "alright then",
        user_config={
            "gateway": {
                "telegram": {
                    "jev_routing": {"enabled": True, "threshold": 0.85, "timeout_seconds": 4}
                }
            }
        },
        api_key="k",
    )
    assert verdict is None


def test_quality_gate_fail_opens_on_transport_failure():
    from gateway.run_turn_fast_lane_quality_gate import evaluate_fast_lane_draft

    _install([[ConnectionResetError("down")], [ConnectionResetError("still down")]])
    decision = evaluate_fast_lane_draft(
        "hi",
        "hello",
        user_config={
            "gateway": {
                "telegram": {
                    "fast_lane": {
                        "quality_gate": {
                            "enabled": True,
                            "threshold": 0.7,
                            "timeout_seconds": 1.5,
                        }
                    }
                }
            }
        },
        api_key="k",
    )
    assert decision == "send"
