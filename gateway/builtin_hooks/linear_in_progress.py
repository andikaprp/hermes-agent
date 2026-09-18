"""Linear in-progress turn note (builtin gateway hook).

Always-registered ``agent:start`` hook: surfaces the Labs team's in-progress
Linear issues (LAB-*) as a compact note at the start of a real turn, enforcing
the standing convention "check in-progress Linear issues at the start of any
real task". The note rides the user message via the api_content sidecar seam
(never the system prompt, so prompt caching stays intact).

Fail-closed by design:
  * reuses the Linear MCP integration's cached OAuth access token
    (``HERMES_HOME/mcp-tokens/linear.json``, the file
    ``tools.mcp_oauth.HermesTokenStorage`` owns) so no new secret plumbing;
  * never refreshes tokens (a refresh POST could burn the MCP runtime's
    single-use refresh token); expired/missing token means NO note;
  * any failure (no token, network error, timeout, malformed payload, 401)
    degrades to no note and never blocks, delays or breaks a turn;
  * network work is bounded and the result is TTL-cached per profile home so
    consecutive turns cost nothing.
"""

import asyncio
import json
import logging
import re
import time
import urllib.error  # noqa: F401 -- imported for the documented error surface
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("gateway.hooks")

_GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"
_TEAM_KEY = "LAB"          # Labs team; issue identifiers are LAB-*
_ISSUE_PREFIX = "LAB-"    # defensive identifier filter
_STATE_TYPE_FILTER = "started"  # Linear's built-in "In Progress" state type
_MAX_ISSUES = 5
_MAX_TITLE_CHARS = 100
_TTL_SECONDS = 60.0
_FETCH_TIMEOUT_SECONDS = 2.0
_TOTAL_BUDGET_SECONDS = 2.5

_QUERY = """
query InProgressIssues($teamKey: String!) {
  team(key: $teamKey) {
    issues(filter: { state: { type: { eq: "started" } } }, first: 6, orderBy: updatedAt) {
      nodes { identifier title }
    }
  }
}
"""

#: TTL cache per profile home: ``str(home) -> (monotonic stamp, note_or_None)``.
_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}


def _clean_title(title: str) -> str:
    """One line, control chars removed, capped; the note is context, not prose."""
    cleaned = re.sub(r"\s+", " ", title).strip()
    if len(cleaned) > _MAX_TITLE_CHARS:
        cleaned = cleaned[:_MAX_TITLE_CHARS - 1].rstrip() + "\u2026"
    return cleaned


def parse_in_progress(payload: Optional[Dict[str, Any]], *, team_key: str = _TEAM_KEY) -> List[Tuple[str, str]]:
    """Extract ``(identifier, title)`` rows from a Linear GraphQL response; [] when malformed."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return []
    team = payload["data"].get("team")
    if not isinstance(team, dict):
        return []
    issues = team.get("issues")
    nodes = issues.get("nodes") if isinstance(issues, dict) else None
    if not isinstance(nodes, list):
        return []
    prefix = f"{team_key}-"
    rows: List[Tuple[str, str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        identifier = node.get("identifier")
        title = node.get("title")
        if not isinstance(identifier, str) or not isinstance(title, str):
            continue
        if not identifier.startswith(prefix):
            continue
        rows.append((identifier, _clean_title(title)))
    return rows


def build_note(rows: Sequence[Tuple[str, str]], *, team_key: str = _TEAM_KEY) -> Optional[str]:
    """Compact one-line note for up to ``_MAX_ISSUES`` rows; None when empty."""
    if not rows:
        return None
    shown = list(rows)[:_MAX_ISSUES]
    parts = " \u00b7 ".join(f"{identifier} {title}" for identifier, title in shown)
    if len(rows) > _MAX_ISSUES:
        parts += f" (+{len(rows) - _MAX_ISSUES} more)"
    return (
        f"[Linear in progress ({team_key})] {parts} "
        "- check for a matching issue before starting the task"
    )


def _request_graphql(access_token: str, query: str, variables: Dict[str, Any], *, timeout: float) -> Optional[Dict[str, Any]]:
    """POST one GraphQL query; None on any HTTP/network/parse failure (caller catches)."""
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        _GRAPHQL_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_in_progress(
    access_token: str,
    *,
    team_key: str = _TEAM_KEY,
    timeout: float = _FETCH_TIMEOUT_SECONDS,
) -> List[Tuple[str, str]]:
    """In-progress Linear issues for the team; [] on any failure (never raises)."""
    try:
        payload = _request_graphql(access_token, _QUERY, {"teamKey": team_key}, timeout=timeout)
    except Exception:
        logger.debug("linear-in-progress: fetch failed (%s)", type(__import__("sys").exc_info()[1]).__name__)
        return []
    return parse_in_progress(payload, team_key=team_key)


def _read_cached_access_token(home: Path) -> Optional[str]:
    """Read the Linear MCP integration's cached OAuth access token if unexpired.

    Layout owned by ``tools.mcp_oauth.HermesTokenStorage``. The hook never
    refreshes, so an expired/missing token is the same thing: no note.
    """
    try:
        from tools.mcp_oauth import _get_token_dir, _safe_filename
        path = _get_token_dir(home) / f"{_safe_filename('linear')}.json"
    except Exception:
        path = home / "mcp-tokens" / "linear.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        return None
    expires_at = data.get("expires_at")
    if isinstance(expires_at, (int, float)) and expires_at <= time.time():
        return None
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in <= 0:
        return None
    return token


def _note_for_home(home: Path) -> Optional[str]:
    """One decided note for a profile home; None on every failed branch (fail closed)."""
    key = str(home)
    cached = _CACHE.get(key)
    if cached is not None and (time.monotonic() - cached[0]) < _TTL_SECONDS:
        return cached[1]
    note: Optional[str] = None
    try:
        token = _read_cached_access_token(home)
        if token:
            note = build_note(fetch_in_progress(token))
    except Exception as exc:  # noqa: BLE001 -- a hook must never raise
        logger.debug("linear-in-progress: skipped (%s)", type(exc).__name__)
        note = None
    _CACHE[key] = (time.monotonic(), note)
    return note


def _clear_cache() -> None:
    """Test helper: drop the per-home TTL cache."""
    _CACHE.clear()


async def handle(event_type: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Hook handler for ``agent:start``: the in-progress Linear note, or None (fail closed)."""
    if event_type != "agent:start":
        return None
    from hermes_cli.config import get_hermes_home

    home = get_hermes_home()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_note_for_home, home),
            timeout=_TOTAL_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug("linear-in-progress: timed out (>%.1fs); skipping note", _TOTAL_BUDGET_SECONDS)
        return None
    except Exception as exc:  # noqa: BLE001 -- a hook must never raise
        logger.debug("linear-in-progress: skipped (%s)", type(exc).__name__)
        return None