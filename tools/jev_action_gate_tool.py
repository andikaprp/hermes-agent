"""Service-gated ``jev_choose_action`` tool (LAB-60).

Appears when ``computer_use.jev_action_gate.enabled`` or
``browser.jev_action_gate.enabled`` is true. Returns an action id from the
caller-supplied pre-approved table (or ``reobserve`` / ``none``). Does not
execute the action itself. When the gate is enabled, the computer_use /
browser tool call must pass that id as ``jev_action_id`` or it will not run.
Existing approval and risk gates still apply after that check.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from gateway.jev_action_gate import (
    choose_next_action,
    decision_payload,
    load_jev_action_gate_config,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

JEV_CHOOSE_ACTION_SCHEMA = {
    "name": "jev_choose_action",
    "description": (
        "Ask Jev which pre-approved computer/browser action to take next. "
        "Pass only a goal, short element labels, and a table of safe actions "
        "(must include reobserve and none). Never pass screenshots, page text, "
        "or field values — those are refused. Returns one action id from the "
        "table, or reobserve on error/timeout/low confidence. Pass that id as "
        "jev_action_id on the computer_use or browser call, with the same "
        "candidate table: while the gate is enabled the tool does not execute "
        "unless Jev approves it. Approval gates still apply after that."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Short non-sensitive goal for this step.",
            },
            "candidates": {
                "type": "array",
                "description": (
                    "Pre-approved actions. Each item needs id + description. "
                    "Must include reobserve and none."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["id", "description"],
                },
            },
            "regions": {
                "type": "array",
                "description": "Optional short on-screen element labels (id, role, label).",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "role": {"type": "string"},
                        "label": {"type": "string"},
                        "interactive": {"type": "boolean"},
                    },
                    "required": ["id", "label"],
                },
            },
            "history": {
                "type": "array",
                "description": "Optional recent selected_id + outcome pairs.",
                "items": {
                    "type": "object",
                    "properties": {
                        "selected_id": {"type": "string"},
                        "outcome": {"type": "string"},
                    },
                },
            },
            "observation_id": {
                "type": "string",
                "description": "Optional opaque observation id for this step.",
            },
            "surface": {
                "type": "string",
                "enum": ["computer_use", "browser"],
                "description": "Which config surface to read (default: either enabled).",
            },
        },
        "required": ["goal", "candidates"],
    },
}


def check_jev_choose_action_available() -> bool:
    """True when either computer_use or browser Jev action gate is enabled."""
    cfg = load_jev_action_gate_config()
    return bool(cfg.enabled)


def handle_jev_choose_action(args: Dict[str, Any], **_kwargs: Any) -> str:
    """Run the withheld-context action gate; always returns JSON."""
    surface = args.get("surface")
    if surface not in (None, "", "computer_use", "browser"):
        surface = None
    surface_name = surface or None
    cfg = load_jev_action_gate_config(surface=surface_name)
    decision = choose_next_action(
        goal=str(args.get("goal") or ""),
        candidates=list(args.get("candidates") or []),
        regions=args.get("regions"),
        history=args.get("history"),
        observation_id=str(args.get("observation_id") or ""),
        cfg=cfg,
        surface=surface_name,
        request_payload=args,
    )
    return json.dumps(decision_payload(decision), ensure_ascii=False)


registry.register(
    name="jev_choose_action",
    toolset="computer_use",
    schema=JEV_CHOOSE_ACTION_SCHEMA,
    handler=lambda args, **kw: handle_jev_choose_action(args, **kw),
    check_fn=check_jev_choose_action_available,
    requires_env=[],
    description=(
        "Jev picks the next pre-approved computer/browser action from a table "
        "(withheld context). Config-gated; fails open to reobserve."
    ),
    emoji="🎯",
)
