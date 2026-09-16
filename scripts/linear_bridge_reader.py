#!/usr/bin/env python3
"""Read-only, project-scoped Linear bridge over Hermes' canonical MCP client."""
from __future__ import annotations
import argparse, json, os, random, sys, time, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.redact import redact_sensitive_text

PROJECT_ID = "a181fe78-5b41-41ea-8e59-4be3766ca9b6"
TITLE_PREFIX = "[INSTINCT-BRIDGE]"
MCP_SERVER = "linear"
# Linear's MCP read tool is intentionally resolved through Hermes discovery/dispatch.
# No OAuth material or HTTP transport is handled here.
ISSUE_READ_TOOL = "list_issues"

class Busy(RuntimeError): pass
class MCPCapabilityError(RuntimeError): pass

class FileLock:
    def __init__(self, path: Path): self.path, self.fd = path, None
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True); self.fd = self.path.open("a+")
        try: fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: self.fd.close(); raise Busy("bridge run already in progress")
        return self
    def __exit__(self, *_):
        import fcntl
        fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close()

def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def parse_states(value: str | None) -> tuple[str, ...]:
    values = tuple(x.strip() for x in (value or "Todo,In Progress").split(",") if x.strip())
    if not values: raise ValueError("status allowlist cannot be empty")
    return values

def load_state(path: Path) -> dict[str, Any]:
    if not path.exists(): return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict): raise ValueError("invalid bridge state")
    return data

def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")), encoding="utf-8"); os.replace(tmp, path)

class LinearMCPAdapter:
    """Thin adapter over Hermes' already-connected MCP server.

    ``call`` is injectable for tests; production dispatch is the same native handler used by
    registered MCP tools, including account selection, OAuth refresh, trust gates and backoff.
    """
    def __init__(self, call: Callable[[str, dict[str, Any]], Any] | None = None):
        dispatch = call
        if dispatch is None:
            # Use Hermes' canonical synchronous discovery path first.  A normal Hermes
            # context may have configured servers but no live registry entry yet.
            from tools.mcp_tool_discovery import discover_mcp_tools
            discover_mcp_tools(allowed_mcp_names=[MCP_SERVER])
            from tools.mcp_tool_handlers import _make_tool_handler
            # Resolve and invoke the native synchronous registry handler here, from the
            # caller's ordinary thread.  Do not create/run an asyncio loop in this adapter.
            handler = _make_tool_handler(MCP_SERVER, ISSUE_READ_TOOL, 30)
            def dispatch(server_tool: str, arguments: dict[str, Any]):
                return handler(arguments)
        self._call = dispatch

    def list_issues(self, *, project_id: str, title_prefix: str, states: tuple[str, ...], after: str | None) -> dict[str, Any]:
        # These are MCP arguments, not a GraphQL query. Keep the boundary explicit so a server
        # cannot broaden this bridge's scope silently.
        args = {
            "project": project_id,
            "team": None,
            "status": list(states),
            "filter": {"title": {"startsWith": title_prefix}},
            "pagination": {"after": after},
        }
        raw = self._call(ISSUE_READ_TOOL, args)
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except json.JSONDecodeError as exc: raise MCPCapabilityError("Linear MCP list_issues returned non-JSON data") from exc
        if not isinstance(raw, dict): raise MCPCapabilityError("Linear MCP list_issues returned an unsupported result")
        # Canonical Linear MCP returns a JSON envelope with a flat issues array and
        # cursor metadata.  Normalize it to the bridge's internal page shape.
        issues = raw.get("issues")
        if isinstance(issues, list):
            return {"nodes": issues, "pageInfo": {
                "hasNextPage": bool(raw.get("hasNextPage", False)),
                "endCursor": raw.get("cursor"),
            }}
        # Keep compatibility with structured MCP renderers used by older installations.
        issues = raw.get("data", raw)
        if isinstance(issues, dict) and "issues" in issues: issues = issues["issues"]
        if isinstance(issues, dict) and isinstance(issues.get("nodes"), list):
            return issues
        raise MCPCapabilityError("Linear MCP must expose list_issues with issues and pagination metadata")

_CREDENTIAL_ASSIGNMENT = re.compile(r"(?i)(\b(?:api[_ .-]?key|token|secret|password|passwd|pass|pw|credential|authorization|auth)\b\s*[:=]\s*)([^\s,;&]+)")
_BEARER = re.compile(r"(?i)(\bbearer\s+)([^\s,;&]+)")
def _safe_external_text(value: Any) -> Any:
    if not isinstance(value, str): return value
    value = redact_sensitive_text(value, force=True, redact_url_credentials=True)
    return _BEARER.sub(r"\1[REDACTED]", _CREDENTIAL_ASSIGNMENT.sub(r"\1[REDACTED]", value))
def _safe_external(value: Any) -> Any:
    if isinstance(value, str): return _safe_external_text(value)
    if isinstance(value, list): return [_safe_external(x) for x in value]
    if isinstance(value, dict): return {k: _safe_external(v) for k,v in value.items()}
    return value

def safe_item(issue: dict[str, Any]) -> dict[str, Any]:
    item = {"source":"linear", "trust":"untrusted_external_input", "project_id":PROJECT_ID,
      "issue_id":issue.get("id"), "identifier":issue.get("identifier"), "title":issue.get("title"),
      "description":(issue.get("description") or "")[:20000], "url":issue.get("url"),
      "created_at":issue.get("createdAt"), "updated_at":issue.get("updatedAt"), "state":issue.get("state") or {},
      "labels":[x.get("name", "") for x in (issue.get("labels") or {}).get("nodes", [])],
      "comments":[{k:c.get(k) for k in ("id","body","createdAt","updatedAt","user")} for c in (issue.get("comments") or {}).get("nodes", [])]}
    for key in ("requested_outcome", "requestedOutcome", "evidence", "error", "link", "links"):
        if key in issue: item[key] = issue[key]
    return _safe_external(item)

def read_once(client: LinearMCPAdapter, state_path: Path, output, statuses: tuple[str, ...]) -> int:
    with FileLock(state_path.with_suffix(state_path.suffix + ".lock")):
        state = load_state(state_path)
        if "watermark" not in state:
            save_state(state_path, {"watermark": now(), "seen": []}); return 0
        watermark, seen, after, found = state["watermark"], set(state.get("seen", [])), None, []
        while True:
            page = client.list_issues(project_id=PROJECT_ID, title_prefix=TITLE_PREFIX, states=statuses, after=after)
            for issue in page["nodes"]:
                # Treat MCP-side filters as advisory: enforce the bridge scope again at the
                # trust boundary so a permissive/misconfigured server cannot broaden reads.
                project = issue.get("project") or {}
                project_id = project.get("id") if isinstance(project, dict) else project
                state = issue.get("state") or {}
                state_name = state.get("name") if isinstance(state, dict) else state
                if (
                    project_id == PROJECT_ID
                    and issue.get("title", "").startswith(TITLE_PREFIX)
                    and state_name in statuses
                    and issue.get("updatedAt", "") > watermark
                ):
                    key = f'{issue.get("id")}:{issue.get("updatedAt")}'
                    if key not in seen: found.append((key, issue))
            if not page.get("pageInfo", {}).get("hasNextPage"): break
            after = page["pageInfo"].get("endCursor")
        for key, issue in sorted(found, key=lambda pair: pair[1].get("updatedAt", "")):
            output.write(json.dumps(safe_item(issue), ensure_ascii=False, sort_keys=True) + "\n"); seen.add(key)
        save_state(state_path, {"watermark": max([watermark] + [i.get("updatedAt", watermark) for _, i in found]), "seen": sorted(seen)[-10000:]})
        return len(found)

def main(argv=None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--state", type=Path, default=Path(os.getenv("LINEAR_BRIDGE_STATE_FILE", "linear-bridge-state.json"))); p.add_argument("--output", type=Path); p.add_argument("--statuses", default=os.getenv("LINEAR_BRIDGE_STATUSES", "Todo,In Progress")); a=p.parse_args(argv)
    out = a.output.open("a", encoding="utf-8") if a.output else sys.stdout
    try: read_once(LinearMCPAdapter(), a.state, out, parse_states(a.statuses))
    except Busy: return 0
    except MCPCapabilityError as exc: print(f"Linear MCP capability missing: {exc}", file=sys.stderr); return 2
    finally:
        if a.output: out.close()
    return 0
if __name__ == "__main__": raise SystemExit(main())
