"""LAB-61 wiring: the turn path must read hitl_escalation outside the writer.

Red on base: the flag is written in run_turn_jev_completion and nothing in the
turn handler or inbound intercept consumes it.
"""

from __future__ import annotations

import inspect


def test_turn_handler_consumes_hitl_escalation():
    from gateway.run_turn import GatewayTurnMixin

    src = inspect.getsource(GatewayTurnMixin._handle_message_with_agent)
    assert "consume_hitl_escalation" in src


def test_inbound_intercepts_completion_hitl_reply():
    from gateway.run_inbound import GatewayInboundMixin

    src = inspect.getsource(GatewayInboundMixin._hm_pending_reply_intercepts)
    assert "_hm_completion_hitl_reply" in src


def test_session_boundary_clears_completion_hitl():
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    src = inspect.getsource(GatewayAgentCacheMixin._clear_session_boundary_security_state)
    assert "jev_completion_hitl" in src
