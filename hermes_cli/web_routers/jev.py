"""Dashboard routes for native Jev decision observability (LAB-59).

Read-only local view of THE decision store (``logs/jev-decisions.jsonl``
and the in-process ring). Does not tail gateway.log and sends nothing
off-box. Config gate: ``gateway.jev_observability.mode`` (off|shadow|on).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from hermes_cli.web_deps import late

router = APIRouter()

_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")


@router.get("/api/jev/decisions")
async def get_jev_decisions(
    limit: int = Query(200, ge=1, le=5000),
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Last N Jev decisions (tier, model, confidence, latency, reason)."""

    def _run() -> Dict[str, Any]:
        from gateway.jev_observability import read_recent_decisions

        with _profile_scope(profile):
            return read_recent_decisions(limit=limit)

    return await asyncio.to_thread(_run)


@router.get("/api/jev/observability")
async def get_jev_observability_status(profile: Optional[str] = None) -> Dict[str, Any]:
    """Current observability mode + path (for dashboard header)."""

    def _run() -> Dict[str, Any]:
        from gateway.jev_observability import (
            decisions_path,
            load_jev_observability_config,
            read_recent_decisions,
        )

        with _profile_scope(profile):
            cfg = load_jev_observability_config()
            feed = read_recent_decisions(limit=min(cfg.limit, 50))
            return {
                "mode": cfg.mode,
                "limit": cfg.limit,
                "path": str(decisions_path()),
                "count": len(feed.get("decisions") or []),
                "ui": cfg.mode == "on",
            }

    return await asyncio.to_thread(_run)
