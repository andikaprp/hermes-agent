#!/usr/bin/env python3
"""Live Telegram turn-latency sampler (stdlib only, read-only).

Closes LAB-3's measurement gap with REAL log-derived numbers instead of fixtures.
It never sends a Telegram message, never touches a service, and only reads log files.

What it measures, per real Telegram turn
----------------------------------------
* ``quiet_window_ms``   — the intentional coalescing window: gap from the arriving
                          user bubble's platform timestamp to the adapter's batch
                          dispatch (the ``Flushing text batch`` line, which is emitted
                          *after* the quiet period elapses). Model-free by construction.
* ``ttfs_ms``           — time to first outbound send: first user bubble -> first
                          ``telegram_delivery_receipt`` send of that turn.
                          Only available for turns carrying delivery receipts.
* ``total_ms``          — last inbound bubble -> last outbound send / turn end.
* ``model_calls``       — API calls in the turn (declared by the gateway, cross-checked
                          against ``agent.conversation_loop: API call #N`` lines).
* ``tool_calls``        — ``agent.tool_executor: tool X completed`` lines in the turn.
* median / p90 (nearest-rank) / worst, per message class and overall.

Log markers used (all VERIFIED present in /workspace/hermes/logs, see report)
----------------------------------------------------------------------------
gateway.log
  ``gateway.run: inbound message: platform=telegram ...``          turn's inbound (post-flush, so it EXCLUDES the quiet window)
  ``hermes_plugins.telegram_platform.adapter: [Telegram] Flushing text batch <key> (<N> chars)``
                                                                  batch dispatch = END of the coalescing window
  ``gateway.telegram_delivery_receipt: telegram_delivery_receipt mono=... mid=... outcome=...``
                                                                  one per NEW message handed to the Bot API (edits/control sends emit none)
  ``gateway.run: response ready: platform=telegram chat=... time=Xs api_calls=N response=N chars``
                                                                  turn end; X is measured from the inbound log line, i.e. from AFTER the quiet window
agent.log
  ``agent.turn_context: conversation turn: ... platform=telegram ... msg='[Www YYYY-MM-DD HH:MM:SS TZ] ...'``
                                                                  embedded bracket = the user bubble's PLATFORM timestamp
  ``agent.conversation_loop: API call #N: ... latency=<f>s``
  ``agent.tool_executor: tool <name> completed (<f>s, <N> chars)``

Turn model
----------
A turn is anchored on a ``response ready`` line (the gateway's own turn boundary).
Its start is ``ready_ts - time``. Every flush / receipt / turn-context line is assigned to
the turn with the latest start that still precedes it — so a mid-turn ("steered") bubble
lands in the turn it interrupted, matching how the gateway actually served it.

Usage
-----
  python3 telegram_latency_samples.py                                  # today, default logs
  python3 telegram_latency_samples.py --date 2026-09-17 --json-out /tmp/x.json
  python3 telegram_latency_samples.py --all-dates --md-out /tmp/tables.md
  python3 telegram_latency_samples.py --gateway-log /workspace/hermes/logs/gateway.log \\
                                     --agent-log  /workspace/hermes/logs/agent.log

Re-run after more live Telegram traffic (no restart needed; logs are appended live):
  python3 /home/rancana/.hermes/hermes-agent/scripts/telegram_latency_samples.py \\
      --date $(date +%F) --json-out /workspace/projects/telegram-delivery-verification/lab3-live-latency.json

Stdlib only. Exit code 0 on success, 2 on unusable input.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_LOG_DIR = os.environ.get("HERMES_LOG_DIR", "/workspace/hermes/logs")


def _discover(name: str) -> List[str]:
    """``gateway.log`` plus rotated ``gateway.log.N`` siblings, newest first.

    Rotation is independent per file (gateway.log.N can cover a different day than
    agent.log.N), so a day's quiet window is only measurable when BOTH logs still
    contain that day. Per-day coverage in the JSON makes that visible instead of
    silently averaging over days that lack one side of the pair.
    """
    base = os.path.join(DEFAULT_LOG_DIR, name)
    out = [base] if os.path.exists(base) else []
    for suffix in sorted(os.listdir(DEFAULT_LOG_DIR) if os.path.isdir(DEFAULT_LOG_DIR) else [],
                         key=lambda s: (len(s), s)):
        if suffix.startswith(name + ".") and suffix[len(name) + 1:].isdigit():
            out.append(os.path.join(DEFAULT_LOG_DIR, suffix))
    return out


DEFAULT_GATEWAY = _discover("gateway.log")
DEFAULT_AGENT = _discover("agent.log")
DEFAULT_CONFIG = os.environ.get("HERMES_CONFIG", "/workspace/hermes/config.yaml")

LINE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
EMBEDDED_TS = re.compile(r"\[([A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([A-Za-z]{2,5})\]")

RE_FLUSH = re.compile(r"Flushing text batch (\S+) \((\d+) chars\)")
RE_INBOUND = re.compile(r"inbound message: platform=(\S+) user=(\S+) chat=(\S+)")
RE_RECEIPT = re.compile(
    r"mono=([\d.]+) chat=(\S+) attempt=(\d+) mid=(\S+) anchor=(\S+) thread=(\S+) outcome=(\S+)")
RE_READY = re.compile(r"response ready: platform=(\S+) chat=(\S+) time=([\d.]+)s api_calls=(\d+) response=(\d+)")
RE_WATCH = re.compile(r"Watch pattern notification . injecting for (\S+) chat=(\S+)")
RE_SEND_RESPONSE = re.compile(r"\[(Telegram|Slack|Inline)\] Sending response \((\d+) chars\) to (\S+)")
RE_SUPPRESS = re.compile(r"Suppressing normal final send for session (\S+)")
RE_TCTX = re.compile(
    r"session=(\S+) model=(\S+) provider=(\S+) platform=(\S+) history=(\d+) msg=")
RE_API = re.compile(
    r"API call #(\d+): model=(\S+) provider=(\S+) in=(\d+) out=(\d+) total=(\d+) latency=([\d.]+)s")
RE_TOOL = re.compile(r"tool (\S+) completed \(([\d.]+)s, (\d+) chars\)")
# Emitted by gateway/stream_consumer_latency.py. Absent from any log written before that
# landed, so ``stream_ttft_ms`` / ``stream_transport_ms`` are simply unmeasurable on older
# logs rather than zero — ``None`` is preserved all the way into the JSON.
RE_FIRST_DELTA = re.compile(r"stream_first_delta turn=(\S+) chat=(\S+) since_open_ms=([\d.]+)")
RE_FIRST_VISIBLE = re.compile(
    r"stream_first_visible turn=(\S+) chat=(\S+) since_open_ms=([\d.]+)(?: since_first_delta_ms=([\d.]+))?")
RE_FAST_PATH = re.compile(r"fast_path_turn chat=(\S+) reason=(\S+)")
RE_FAST_PATH_OUTCOME = re.compile(
    r"fast_path_outcome chat=(\S+) reason=(\S+) api_calls=(\d+) tool_calls=(\d+)")
RE_SESSION_TAG = re.compile(r"\[(\d{8}_\d{6}_[0-9a-f]+)\]")

# A gap this far above the configured quiet window cannot be a clean coalescing wait:
# the batcher clamps the intentional window at text_batch_max_wait_seconds (config.yaml).
QUIET_SLACK_S = 0.5


def parse_line_ts(text: str) -> Optional[float]:
    """Epoch seconds for a leading ``YYYY-mm-dd HH:MM:SS,mmm`` log stamp (log-local time)."""
    m = LINE_TS.match(text)
    if not m:
        return None
    base = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return base.timestamp() + int(m.group(2)) / 1000.0


def parse_embedded_ts(text: str) -> Optional[float]:
    """Epoch seconds for a bracketed ``[Www YYYY-mm-dd HH:MM:SS TZ]`` platform stamp.

    Naive-local by design: the gateway renders these in the host's local zone
    (Asia/Jakarta here) and the log stamps are local too, so both share a clock.
    """
    m = EMBEDDED_TS.search(text)
    if not m:
        return None
    stamp = m.group(1).split(" ", 1)[1]
    return _dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()


def ceil_rank(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile: sorted[ceil(pct * n) - 1]."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(pct * len(ordered)))
    return ordered[k - 1]


def stats_block(values: List[float]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "min": None, "median": None, "p90": None, "worst": None}
    return {
        "n": len(clean),
        "min": round(min(clean), 1),
        "median": round(statistics.median(clean), 1),
        "p90": round(ceil_rank(clean, 0.90) or 0.0, 1),
        "worst": round(max(clean), 1),
    }


def configured_quiet_window(config_path: str) -> Dict[str, Any]:
    """Read the intentional quiet window from config.yaml without a YAML dependency."""
    out: Dict[str, Any] = {"path": config_path, "quiet_seconds": None, "max_wait_seconds": None,
                           "conversational_dm_batching": None, "source": "unreadable"}
    try:
        with open(config_path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    for yaml_key, short_key in (("text_batch_quiet_seconds", "quiet_seconds"),
                                ("text_batch_max_wait_seconds", "max_wait_seconds")):
        m = re.search(rf"^\s*{yaml_key}:\s*([0-9.]+)\s*$", text, re.M)
        if m:
            out[short_key] = float(m.group(1))
    m = re.search(r"^\s*conversational_dm_batching:\s*(\w+)\s*$", text, re.M)
    if m:
        out["conversational_dm_batching"] = m.group(1).lower() == "true"
    out["source"] = "config.yaml"
    return out


class LogScan:
    """One pass over the gateway/agent logs -> timestamped marker lists."""

    def __init__(self) -> None:
        self.flush: List[Dict[str, Any]] = []
        self.inbound: List[Dict[str, Any]] = []
        self.receipt: List[Dict[str, Any]] = []
        self.ready: List[Dict[str, Any]] = []
        self.watch: List[Dict[str, Any]] = []
        self.suppress: List[Dict[str, Any]] = []
        self.final_send: List[Dict[str, Any]] = []
        self.turnctx: List[Dict[str, Any]] = []
        self.api: List[Dict[str, Any]] = []
        self.tool: List[Dict[str, Any]] = []
        self.first_delta: List[Dict[str, Any]] = []
        self.first_visible: List[Dict[str, Any]] = []
        self.fast_path: List[Dict[str, Any]] = []
        self.fast_path_outcome: List[Dict[str, Any]] = []


def scan_logs(gateway_paths: Iterable[str], agent_paths: Iterable[str]) -> LogScan:
    scan = LogScan()
    for path in gateway_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                ts = parse_line_ts(line)
                if ts is None:
                    continue
                if (m := RE_FLUSH.search(line)):
                    scan.flush.append({"ts": ts, "key": m.group(1), "chars": int(m.group(2))})
                elif (m := RE_RECEIPT.search(line)):
                    scan.receipt.append({"ts": ts, "mono": float(m.group(1)), "chat": m.group(2),
                                         "attempt": int(m.group(3)), "mid": m.group(4),
                                         "outcome": m.group(7)})
                elif (m := RE_READY.search(line)):
                    scan.ready.append({"ts": ts, "platform": m.group(1), "chat": m.group(2),
                                       "time_s": float(m.group(3)), "api_calls": int(m.group(4)),
                                       "response_chars": int(m.group(5))})
                elif (m := RE_INBOUND.search(line)):
                    scan.inbound.append({"ts": ts, "platform": m.group(1), "chat": m.group(3)})
                elif (m := RE_WATCH.search(line)):
                    scan.watch.append({"ts": ts, "chat": m.group(2)})
                elif (m := RE_SEND_RESPONSE.search(line)):
                    if m.group(1) == "Telegram":
                        scan.final_send.append({"ts": ts, "chat": m.group(3), "chars": int(m.group(2))})
                elif (m := RE_FIRST_DELTA.search(line)):
                    scan.first_delta.append({"ts": ts, "turn": m.group(1), "chat": m.group(2),
                                             "since_open_ms": float(m.group(3))})
                elif (m := RE_FIRST_VISIBLE.search(line)):
                    scan.first_visible.append({
                        "ts": ts, "turn": m.group(1), "chat": m.group(2),
                        "since_open_ms": float(m.group(3)),
                        "since_first_delta_ms": float(m.group(4)) if m.group(4) else None})
                elif (m := RE_FAST_PATH_OUTCOME.search(line)):
                    scan.fast_path_outcome.append({
                        "ts": ts, "chat": m.group(1), "reason": m.group(2),
                        "api_calls": int(m.group(3)), "tool_calls": int(m.group(4))})
                elif (m := RE_FAST_PATH.search(line)):
                    scan.fast_path.append({"ts": ts, "chat": m.group(1), "reason": m.group(2)})
                elif RE_SUPPRESS.search(line):
                    scan.suppress.append({"ts": ts})
        # WARNING: two different log names appear for the same adapter (hermes_plugins.*
        # and plugins.platforms.*); both carry the same "Flushing text batch" body, so the
        # match above is name-agnostic on purpose.
    for path in agent_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                ts = parse_line_ts(line)
                if ts is None:
                    continue
                tag = RE_SESSION_TAG.search(line)
                session = tag.group(1) if tag else None
                if "agent.turn_context: conversation turn:" in line:
                    m = RE_TCTX.search(line)
                    if m and m.group(4) == "telegram":
                        scan.turnctx.append({"ts": ts, "session": m.group(1),
                                             "model": m.group(2), "history": int(m.group(5)),
                                             "platform_ts": parse_embedded_ts(line)})
                elif "agent.conversation_loop: API call #" in line:
                    m = RE_API.search(line)
                    if m:
                        scan.api.append({"ts": ts, "session": session, "n": int(m.group(1)),
                                         "latency_s": float(m.group(7)), "provider": m.group(3)})
                elif "agent.tool_executor: tool" in line and "completed" in line:
                    m = RE_TOOL.search(line)
                    if m:
                        scan.tool.append({"ts": ts, "session": session, "name": m.group(1),
                                          "duration_s": float(m.group(2))})
    return scan


def build_turns(scan: LogScan, platform: str = "telegram") -> List[Dict[str, Any]]:
    """Anchor every turn on a ``response ready`` line, then attribute markers to it."""
    turns: List[Dict[str, Any]] = []
    for r in scan.ready:
        if r["platform"] != platform:
            continue
        turns.append({
            "chat": r["chat"], "ready_ts": r["ts"], "turn_start": r["ts"] - r["time_s"],
            "declared_time_s": r["time_s"], "declared_api_calls": r["api_calls"],
            "response_chars": r["response_chars"],
            "flush": [], "receipt": [], "inbound": [], "turnctx": [], "watch": [], "suppress": [],
            "final_send": [], "first_delta": [], "first_visible": [],
            "fast_path": [], "fast_path_outcome": [],
        })
    turns.sort(key=lambda t: t["ready_ts"])

    def owner(ts: float) -> Optional[Dict[str, Any]]:
        best = None
        for t in turns:
            if t["turn_start"] - 0.5 <= ts <= t["ready_ts"] + 0.5:
                if best is None or t["turn_start"] > best["turn_start"]:
                    best = t
        return best

    for marker_list, bucket in ((scan.flush, "flush"), (scan.receipt, "receipt"),
                                (scan.inbound, "inbound"), (scan.turnctx, "turnctx"),
                                (scan.watch, "watch"), (scan.suppress, "suppress"),
                                (scan.final_send, "final_send"),
                                (scan.first_delta, "first_delta"),
                                (scan.first_visible, "first_visible"),
                                (scan.fast_path, "fast_path"),
                                (scan.fast_path_outcome, "fast_path_outcome")):
        for item in marker_list:
            t = owner(item["ts"])
            if t is not None:
                t[bucket].append(item)
    for bucket in ("flush", "receipt", "inbound", "turnctx", "watch", "suppress", "final_send",
                   "first_delta", "first_visible", "fast_path", "fast_path_outcome"):
        for t in turns:
            t[bucket].sort(key=lambda i: i["ts"])

    telegram_sessions = {c["session"] for c in scan.turnctx}
    for t in turns:
        lo, hi = t["turn_start"] - 0.5, t["ready_ts"] + 0.5
        window_sessions = {c["session"] for c in t["turnctx"]} or telegram_sessions
        t["api_all"] = [a for a in scan.api if lo <= a["ts"] <= hi and a["session"] in window_sessions]
        # The API call counter is per-session and keeps counting across turns (a turn can start at
        # #9), so numbering cannot identify a turn boundary. Concurrent turns can also share the
        # session id (a long turn still running while the next one starts), which makes a bare time
        # window over-count. The gateway's own api_calls=N on the response-ready line is
        # authoritative for the turn, so take the last N calls in the window: a turn's calls are
        # contiguous and end at its own response.
        declared = t["declared_api_calls"]
        if declared is not None and declared >= 0 and len(t["api_all"]) > declared:
            t["api"] = t["api_all"][-declared:] if declared else []
            t["api_attribution"] = "declared_count_tail(concurrent_overlap_dropped)"
        else:
            t["api"] = list(t["api_all"])
            t["api_attribution"] = "full_window_match"
        tool_lo = t["api"][0]["ts"] if t["api"] else lo
        t["tool"] = [x for x in scan.tool if tool_lo - 0.5 <= x["ts"] <= hi and x["session"] in window_sessions]
    return turns


def measure_turn(turn: Dict[str, Any], quiet_seconds: Optional[float],
                 max_wait_seconds: Optional[float]) -> Dict[str, Any]:
    """All per-turn metrics, with an explicit ``basis`` for every timestamp choice."""
    flush = turn["flush"]
    receipts = turn["receipt"]
    tctx = [c for c in turn["turnctx"] if c["platform_ts"] is not None]

    first_flush = flush[0]["ts"] if flush else None
    last_flush = flush[-1]["ts"] if flush else None
    platform_ts = tctx[0]["platform_ts"] if tctx else None
    first_receipt = receipts[0]["ts"] if receipts else None
    final_send = turn["final_send"]
    if receipts:
        last_send, last_send_basis = receipts[-1]["ts"], "last_receipt"
    elif final_send:
        last_send, last_send_basis = final_send[-1]["ts"], "Sending_response(non-streamed_final_send)"
    else:
        last_send, last_send_basis = turn["ready_ts"], "response_ready_ts(proxy,no_send_marker)"

    # Non-streamed turns deliver their one and only response message at the "Sending response"
    # line, so it is both the first and last send. Turns that streamed (they carry a
    # "Suppressing normal final send" line) delivered earlier and are excluded from this cohort.
    assumed_first_send, assumed_first_basis = None, None
    if first_receipt is None and final_send and not turn["suppress"]:
        assumed_first_send = final_send[0]["ts"]
        assumed_first_basis = "Sending_response(assumed_first:non-streamed,turn_has_no_suppress_line)"

    # quiet window: bubble platform stamp -> batch dispatch (model-free).
    quiet_ms = None
    quiet_basis = None
    if first_flush is not None and platform_ts is not None:
        quiet_ms = round((first_flush - platform_ts) * 1000.0, 1)
        quiet_basis = "flush_minus_platform_ts"
    elif first_flush is not None:
        quiet_basis = "unavailable_no_platform_ts"

    clean_quiet = None
    if quiet_ms is not None:
        cap_ms = None
        if quiet_seconds is not None and max_wait_seconds is not None:
            cap_ms = (max_wait_seconds + QUIET_SLACK_S) * 1000.0
        clean_quiet = quiet_ms if (cap_ms is None or quiet_ms <= cap_ms) else None

    ttfs_ms = None
    if first_receipt is not None and platform_ts is not None:
        ttfs_ms = round((first_receipt - platform_ts) * 1000.0, 1)
    ttfs_assumed_ms = None
    if assumed_first_send is not None and platform_ts is not None:
        ttfs_assumed_ms = round((assumed_first_send - platform_ts) * 1000.0, 1)
    ttfs_flush_basis_ms = None
    if first_receipt is not None and first_flush is not None:
        ttfs_flush_basis_ms = round((first_receipt - first_flush) * 1000.0, 1)

    agent_before_first_send_ms = None
    if first_receipt is not None and first_flush is not None:
        # dispatch of the batch that opened the turn -> first outbound send.
        # Loudest honest decomposition: quiet_window + this == ttfs.
        agent_before_first_send_ms = round((first_receipt - first_flush) * 1000.0, 1)
    decomposition_sum_ms = None
    if quiet_ms is not None and agent_before_first_send_ms is not None:
        decomposition_sum_ms = round(quiet_ms + agent_before_first_send_ms, 1)

    total_from_platform_ms = round((last_send - platform_ts) * 1000.0, 1) if platform_ts else None
    total_from_last_flush_ms = round((last_send - last_flush) * 1000.0, 1) if last_flush else None
    total_ready_basis_ms = round((last_send - turn["turn_start"]) * 1000.0, 1)

    if total_from_platform_ms is not None:
        total_ms, total_basis = total_from_platform_ms, "last_send_minus_platform_ts"
    elif total_from_last_flush_ms is not None:
        total_ms, total_basis = total_from_last_flush_ms, "last_send_minus_last_inbound_batch(flush_proxy)"
    else:
        total_ms, total_basis = total_ready_basis_ms, "last_send_minus_turn_start(no_bubble_marker)"

    gaps = [round((b["ts"] - a["ts"]) * 1000.0, 1) for a, b in zip(flush, flush[1:])]
    if not flush:
        message_class = "injected_no_user_bubble"
    elif len(flush) == 1:
        message_class = "single_batch"
    elif quiet_seconds is not None and any(g > quiet_seconds * 1000.0 for g in gaps):
        message_class = "multi_batch_steered"
    else:
        message_class = "multi_batch_rapid"

    # Streamed time-to-first-token, from the markers gateway/stream_consumer_latency.py emits.
    # ``None`` on any log written before those landed (and on turns that never streamed):
    # unmeasurable, which the aggregate must keep distinct from "measured as zero".
    first_delta = turn["first_delta"]
    first_visible = turn["first_visible"]
    stream_ttft_ms = None
    if first_delta and platform_ts is not None:
        stream_ttft_ms = round((first_delta[0]["ts"] - platform_ts) * 1000.0, 1)
    stream_transport_ms = first_visible[0]["since_first_delta_ms"] if first_visible else None
    stream_first_visible_ms = None
    if first_visible and platform_ts is not None:
        stream_first_visible_ms = round((first_visible[0]["ts"] - platform_ts) * 1000.0, 1)

    outcome = turn["fast_path_outcome"]
    routed = turn["fast_path"]
    fast_path_taken = bool(outcome or routed)
    fast_path_reason = (outcome[0]["reason"] if outcome else (routed[0]["reason"] if routed else None))
    fast_path_api_calls = outcome[0]["api_calls"] if outcome else None
    fast_path_tool_calls = outcome[0]["tool_calls"] if outcome else None
    # Compliance with the one-hop contract: measurable only when the outcome marker is present.
    fast_path_one_hop = None
    if outcome:
        fast_path_one_hop = outcome[0]["api_calls"] == 1 and outcome[0]["tool_calls"] == 0

    api_sessions = {a["session"] for a in turn["api"]}
    return {
        "stream_ttft_ms": stream_ttft_ms,
        "stream_first_visible_ms": stream_first_visible_ms,
        "stream_transport_ms": stream_transport_ms,
        "streamed": bool(first_delta or first_visible),
        "fast_path_taken": fast_path_taken,
        "fast_path_reason": fast_path_reason,
        "fast_path_api_calls": fast_path_api_calls,
        "fast_path_tool_calls": fast_path_tool_calls,
        "fast_path_one_hop": fast_path_one_hop,
        "ready_ts": turn["ready_ts"],
        "ready_at": _dt.datetime.fromtimestamp(turn["ready_ts"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "date": _dt.datetime.fromtimestamp(turn["ready_ts"]).strftime("%Y-%m-%d"),
        "turn_start": turn["turn_start"],
        "chat": turn["chat"],
        "declared_time_s": turn["declared_time_s"],
        "response_chars": turn["response_chars"],
        "inbound_batches": len(flush),
        "inbound_messages_logged": len(turn["inbound"]),
        "interbatch_gaps_ms": gaps,
        "message_class": message_class,
        "watch_injections": len(turn["watch"]),
        "quiet_window_ms": quiet_ms,
        "quiet_window_clean_ms": clean_quiet,
        "quiet_window_basis": quiet_basis,
        "time_to_first_send_ms": ttfs_ms,
        "time_to_first_send_flush_basis_ms": ttfs_flush_basis_ms,
        "time_to_first_send_assumed_nonstreamed_ms": ttfs_assumed_ms,
        "first_send_basis": ("first_receipt" if first_receipt is not None else assumed_first_basis),
        "last_send_basis": last_send_basis,
        "has_suppress_line": bool(turn["suppress"]),
        "agent_before_first_send_ms": agent_before_first_send_ms,
        "decomposition_sum_ms": decomposition_sum_ms,
        "total_from_platform_ms": total_from_platform_ms,
        "total_from_last_flush_ms": total_from_last_flush_ms,
        "total_ready_basis_ms": total_ready_basis_ms,
        "total_ms": total_ms,
        "total_basis": total_basis,
        "outbound_sends_receipted": len(receipts),
        "receipt_mids": [r["mid"] for r in receipts],
        "model_calls_declared": turn["declared_api_calls"],
        "model_calls_agent_log": len(turn["api"]),
        "api_calls_in_window_raw": len(turn["api_all"]),
        "api_attribution": turn["api_attribution"],
        "concurrent_run_overlap": len(turn["api_all"]) != len(turn["api"]),
        "tool_calls_agent_log": len(turn["tool"]),
        "model_latency_sum_s": round(sum(a["latency_s"] for a in turn["api"]), 2) if turn["api"] else None,
        "tool_time_sum_s": round(sum(x["duration_s"] for x in turn["tool"]), 2) if turn["tool"] else None,
        "session": sorted(api_sessions)[0] if api_sessions else None,
        "has_receipts": bool(receipts),
        "has_platform_ts": platform_ts is not None,
    }


def reconcile_prior_claim(samples: List[Dict[str, Any]], claim: Dict[str, float]) -> Dict[str, Any]:
    """Check the recorded prior hand measurement against every candidate basis we can compute."""
    bases: Dict[str, List[float]] = {}
    for s in samples:
        if s["time_to_first_send_ms"] is not None:
            bases.setdefault("platform_ts -> first_receipt (ttfs_ms)", []).append(s["time_to_first_send_ms"])
        if s["time_to_first_send_flush_basis_ms"] is not None:
            bases.setdefault("inbound_flush -> first_receipt", []).append(s["time_to_first_send_flush_basis_ms"])
        if s["total_from_platform_ms"] is not None:
            bases.setdefault("platform_ts -> last_send (total_ms)", []).append(s["total_from_platform_ms"])
        if s["total_from_last_flush_ms"] is not None and s["inbound_batches"] >= 1:
            bases.setdefault("last_flush -> last_send", []).append(s["total_from_last_flush_ms"])
        if s["declared_time_s"] is not None:
            bases.setdefault("gateway 'response ready time='", []).append(s["declared_time_s"] * 1000.0)
    rows = []
    for name, values in sorted(bases.items()):
        block = stats_block(values)
        rows.append({"basis": name, "n": block["n"], "median": block["median"],
                     "min": block["min"], "worst": block["worst"],
                     "matches_claim": block["n"] == claim["n"] and block["median"] == claim["median_ms"]
                     and block["min"] == claim["min_ms"] and block["worst"] == claim["worst_ms"]})
    # Forensics: for each number in the prior claim, the closest sample across every basis. If the
    # three numbers map to different bases, the prior hand measurement mixed clocks.
    pool = []
    for s in samples:
        for basis, field in (("platform_ts -> first_receipt (ttfs_ms)", "time_to_first_send_ms"),
                             ("inbound_flush -> first_receipt", "time_to_first_send_flush_basis_ms"),
                             ("platform_ts -> last_send (total_ms)", "total_from_platform_ms"),
                             ("last_flush -> last_send", "total_from_last_flush_ms"),
                             ("gateway 'response ready time='", "declared_time_s")):
            v = s.get(field)
            if v is None:
                continue
            if field == "declared_time_s":
                v = v * 1000.0
            pool.append({"basis": basis, "value": v, "ready_at": s["ready_at"], "field": field})
    nearest = []
    for label, target in (("min_ms", claim["min_ms"]), ("median_ms", claim["median_ms"]),
                          ("worst_ms", claim["worst_ms"])):
        if not pool:
            continue
        best = min(pool, key=lambda p: abs(p["value"] - target))
        nearest.append({"claim_field": label, "claim_value_ms": target, "matched_basis": best["basis"],
                        "closest_sample_ms": best["value"],
                        "delta_ms": round(abs(best["value"] - target), 1),
                        "sample_ready_at": best["ready_at"], "sample_field": best["field"]})
    distinct_bases = sorted({n["matched_basis"] for n in nearest})
    # Definitively test "the prior 11 samples were a contiguous 11-turn series": slide a window of
    # 11 over every basis and look for an exact (median, min, worst) hit. No hit => the claim cannot
    # be a real 11-turn window of this data on any basis.
    series = {
        "platform_ts -> first_receipt (ttfs_ms)": lambda s: s["time_to_first_send_ms"],
        "inbound_flush -> first_receipt": lambda s: s["time_to_first_send_flush_basis_ms"],
        "platform_ts -> last_send (total_ms)": lambda s: s["total_from_platform_ms"],
        "last_flush -> last_send": lambda s: s["total_from_last_flush_ms"],
        "gateway 'response ready time='": lambda s: (s["declared_time_s"] * 1000.0) if s["declared_time_s"] is not None else None,
    }
    windows_checked, window_matches = 0, []
    window_size = int(claim["n"])
    for name, fn in series.items():
        values = [fn(s) for s in samples]
        for i in range(0, max(0, len(values) - window_size + 1)):
            window = values[i:i + window_size]
            if any(v is None for v in window):
                continue
            windows_checked += 1
            block = stats_block(window)
            if (block["median"] == claim["median_ms"] and block["min"] == claim["min_ms"]
                    and block["worst"] == claim["worst_ms"]):
                window_matches.append({"basis": name, "start_ready_at": samples[i]["ready_at"]})
    return {
        "prior_claim": claim,
        "candidate_bases": rows,
        "reproduced": any(r["matches_claim"] for r in rows),
        "nearest_sample_per_claim_number": nearest,
        "claim_numbers_map_to_distinct_bases": len(distinct_bases) > 1,
        "contiguous_n_window_search": {"window_size": claim["n"], "windows_checked": windows_checked,
                                       "matches": window_matches},
        "note": ("No candidate basis reproduces the prior claim as a single consistent measurement. "
                 "nearest_sample_per_claim_number shows which basis each prior number actually matches; "
                 "if claim_numbers_map_to_distinct_bases is true the prior hand measurement mixed two "
                 "different clock bases and must not be quoted as one series."),
    }


def render_markdown(measured: List[Dict[str, Any]], agg: Dict[str, Any], window: Dict[str, Any]) -> str:
    """Mechanically reproducible tables, generated from the same numbers as the JSON."""
    out: List[str] = []
    out.append("### Per-class aggregates (generated by telegram_latency_samples.py)")
    out.append("")
    out.append("| class | metric | n | median | p90 | worst | min |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for cls, block in agg["by_class"].items():
        for metric, st in block.items():
            out.append(f"| {cls} | {metric} | {st['n']} | {st['median']} | {st['p90']} | {st['worst']} | {st['min']} |")
    out.append("")
    out.append("### Per-turn samples")
    out.append("")
    out.append("| ready_at | class | in_batches | quiet_ms | ttfs_ms | total_ms | model_calls | tool_calls | sends |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in measured:
        out.append("| {ready} | {cls} | {nb} | {q} | {t} | {tot} | {mc} | {tc} | {sd} |".format(
            ready=s["ready_at"][11:], cls=s["message_class"], nb=s["inbound_batches"],
            q=s["quiet_window_ms"], t=s["time_to_first_send_ms"], tot=s["total_ms"],
            mc=s["model_calls_declared"], tc=s["tool_calls_agent_log"], sd=s["outbound_sends_receipted"]))
    out.append("")
    out.append("### Turns with an observed first outbound send (receipts, or a non-streamed final send)")
    out.append("")
    out.append("| ready_at | class | first_send_basis | quiet_ms | dispatch->send_ms | ttfs_ms | total_ms | model_calls | tool_calls |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in measured:
        if not s["first_send_basis"]:
            continue
        first = s["time_to_first_send_ms"] if s["time_to_first_send_ms"] is not None else s["time_to_first_send_assumed_nonstreamed_ms"]
        out.append("| {ready} | {cls} | {basis} | {q} | {d} | {t} | {tot} | {mc} | {tc} |".format(
            ready=s["ready_at"][:19], cls=s["message_class"], basis=s["first_send_basis"][:44],
            q=s["quiet_window_ms"], d=s["agent_before_first_send_ms"], t=first,
            tot=s["total_ms"], mc=s["model_calls_declared"], tc=s["tool_calls_agent_log"]))
    out.append("")
    out.append(f"Quiet window source: {window.get('source')} "
               f"quiet_seconds={window.get('quiet_seconds')} max_wait_seconds={window.get('max_wait_seconds')}")
    return "\n".join(out)


def by_day(measured: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-day coverage so a thin day cannot masquerade as a cohort-wide result."""
    days: Dict[str, List[Dict[str, Any]]] = {}
    for s in measured:
        days.setdefault(s["date"], []).append(s)
    out: Dict[str, Any] = {}
    for day, rows in sorted(days.items()):
        out[day] = {
            "turns": len(rows),
            "turns_with_quiet_window": sum(1 for r in rows if r["quiet_window_clean_ms"] is not None),
            "turns_with_receipts": sum(1 for r in rows if r["has_receipts"]),
            "turns_with_ttfs": sum(1 for r in rows if r["time_to_first_send_ms"] is not None),
            "quiet_window_ms": stats_block([r["quiet_window_clean_ms"] for r in rows]),
            "total_ms": stats_block([r["total_ms"] for r in rows]),
        }
    return out


def aggregate(measured: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "quiet_window_ms": lambda s: s["quiet_window_clean_ms"],
        "time_to_first_send_ms": lambda s: s["time_to_first_send_ms"],
        "time_to_first_send_assumed_nonstreamed_ms": lambda s: s["time_to_first_send_assumed_nonstreamed_ms"],
        "total_ms": lambda s: s["total_ms"],
        "declared_turn_time_ms": lambda s: (s["declared_time_s"] * 1000.0) if s["declared_time_s"] is not None else None,
        "model_calls": lambda s: s["model_calls_declared"],
        "tool_calls": lambda s: s["tool_calls_agent_log"],
        "stream_ttft_ms": lambda s: s["stream_ttft_ms"],
        "stream_first_visible_ms": lambda s: s["stream_first_visible_ms"],
        "stream_transport_ms": lambda s: s["stream_transport_ms"],
        "fast_path_taken": lambda s: 1.0 if s["fast_path_taken"] else 0.0,
        "fast_path_one_hop": lambda s: None if s["fast_path_one_hop"] is None else (1.0 if s["fast_path_one_hop"] else 0.0),
        "fast_path_api_calls": lambda s: s["fast_path_api_calls"],
        "fast_path_tool_calls": lambda s: s["fast_path_tool_calls"],
    }
    groups: Dict[str, List[Dict[str, Any]]] = {"all": measured}
    for s in measured:
        groups.setdefault(s["message_class"], []).append(s)
    by_class: Dict[str, Dict[str, Any]] = {}
    for cls, rows in sorted(groups.items()):
        by_class[cls] = {name: stats_block([fn(r) for r in rows]) for name, fn in metrics.items()}
    return {"by_class": by_class}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live Telegram turn-latency sampler (read-only, stdlib only).")
    ap.add_argument("--gateway-log", action="append", default=None, help="gateway log path (repeatable)")
    ap.add_argument("--agent-log", action="append", default=None, help="agent log path (repeatable)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="config.yaml for the quiet window")
    ap.add_argument("--date", default=None, help="only turns ending on this YYYY-MM-DD (default: all dates in the logs)")
    ap.add_argument("--all-dates", action="store_true", help="keep every day found in the logs")
    ap.add_argument("--json-out", default=None, help="write the raw samples + aggregates as JSON")
    ap.add_argument("--md-out", default=None, help="write generated markdown tables")
    ap.add_argument("--platform", default="telegram")
    ap.add_argument("--label", default=None, help="free-text label stored in the JSON provenance")
    args = ap.parse_args(argv)

    gateway_paths = args.gateway_log or DEFAULT_GATEWAY
    agent_paths = args.agent_log or DEFAULT_AGENT
    if not any(os.path.exists(p) for p in gateway_paths):
        print(f"no readable gateway log in {gateway_paths}", file=sys.stderr)
        return 2

    window = configured_quiet_window(args.config)
    scan = scan_logs(gateway_paths, agent_paths)
    turns = build_turns(scan, platform=args.platform)
    if not turns:
        print("no telegram turns (response ready lines) found", file=sys.stderr)
        return 2

    if window["quiet_seconds"] is None:
        # Config unreadable: derive the coalescing threshold from the observed bubble->dispatch
        # gaps instead of guessing. p10 of the gaps approximates the intentional window floor.
        probe = [measure_turn(t, None, None)["quiet_window_ms"] for t in turns]
        probe = sorted(v for v in probe if v is not None)
        if probe:
            window["quiet_seconds"] = round(probe[max(0, math.ceil(0.10 * len(probe)) - 1)] / 1000.0, 2)
            window["source"] = "derived_from_observed_gaps_p10(config_unreadable)"
            window["max_wait_seconds"] = window["quiet_seconds"] + 1.0
        else:
            window["quiet_seconds"] = 2.0
            window["max_wait_seconds"] = 5.0
            window["source"] = "assumed_defaults(config_unreadable,no_gaps)"

    measured = [measure_turn(t, window["quiet_seconds"], window["max_wait_seconds"]) for t in turns]
    if args.date and not args.all_dates:
        measured = [s for s in measured if s["date"] == args.date]
    measured.sort(key=lambda s: s["ready_ts"])

    # empirical quiet-window evidence (p10..p90 of the clean gaps)
    clean_quiet = [s["quiet_window_clean_ms"] for s in measured if s["quiet_window_clean_ms"] is not None]
    empirical = {
        "observed_clean_quiet_ms": stats_block(clean_quiet),
        "contaminated_quiet_ms": stats_block([s["quiet_window_ms"] for s in measured
                                              if s["quiet_window_ms"] is not None and s["quiet_window_clean_ms"] is None]),
        "configured_quiet_seconds": window["quiet_seconds"],
        "configured_max_wait_seconds": window["max_wait_seconds"],
        "coalesce_threshold_seconds_used": window["quiet_seconds"],
    }
    if empirical["observed_clean_quiet_ms"]["median"] and window["quiet_seconds"]:
        empirical["derivation"] = (
            "config text_batch_quiet_seconds={q}s is the intentional window; the observed "
            "bubble->dispatch gap median is {m} ms, i.e. config window + Telegram polling/ingress "
            "overhead. Bubbles whose gap exceeds max_wait({w}s)+0.5s slack are excluded as "
            "contaminated by queue/backlog.".format(q=window["quiet_seconds"],
                                                    m=empirical["observed_clean_quiet_ms"]["median"],
                                                    w=window["max_wait_seconds"]))

    agg = aggregate(measured)
    receipt_cohort = [s for s in measured if s["has_receipts"]]
    prior = {"n": 11, "median_ms": 24000.0, "min_ms": 6700.0, "worst_ms": 97500.0}

    validation = []
    for s in measured:
        if s["model_calls_agent_log"] and s["model_calls_declared"]:
            row = {"ready_at": s["ready_at"], "declared": s["model_calls_declared"],
                   "agent_log": s["model_calls_agent_log"]}
            row["match"] = row["declared"] == row["agent_log"]
            validation.append(row)
    report = {
        "provenance": {
            "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "gateway_logs": gateway_paths,
            "agent_logs": agent_paths,
            "config": window,
            "platform": args.platform,
            "date_filter": args.date if not args.all_dates else None,
            "label": args.label,
            "sent_any_message": False,
            "modified_any_service": False,
        },
        "method": {
            "turn_anchor": "gateway.run 'response ready: platform=telegram' line; turn_start = ready_ts - time=",
            "turn_attribution": "each flush/receipt/turn_context belongs to the turn with the latest start that precedes it",
            "quiet_window": "first inbound BATCH dispatch (Flushing text batch ts) - user bubble platform ts (embedded [..] in agent.turn_context)",
            "time_to_first_send": "first telegram_delivery_receipt of the turn - user bubble platform ts",
            "total": "last outbound send (last receipt, else the response-ready ts) - user bubble platform ts",
            "percentiles": "nearest-rank p90 = sorted[ceil(0.9*n)-1]",
            "receipt_scope": "one receipt per NEW message handed to the Bot API; edits, streaming-preview frames and control sends emit none",
        },
        "empirical_quiet_window": empirical,
        "aggregates": agg,
        "prior_claim_reconciliation": reconcile_prior_claim(measured, prior),
        "coverage": {
            "turns_measured": len(measured),
            "turns_with_receipts": len(receipt_cohort),
            "turns_with_quiet_window": len(clean_quiet),
            "turns_with_ttfs": sum(1 for s in measured if s["time_to_first_send_ms"] is not None),
            "turns_with_assumed_first_send_nonstreamed": sum(
                1 for s in measured if s["time_to_first_send_assumed_nonstreamed_ms"] is not None),
            "turns_with_platform_ts": sum(1 for s in measured if s["has_platform_ts"]),
            "class_counts": {c: sum(1 for s in measured if s["message_class"] == c)
                             for c in sorted({s["message_class"] for s in measured})},
            "declared_vs_logged_model_calls": {
                "checked": len(validation),
                "match": sum(1 for v in validation if v["match"]),
                "mismatch": sum(1 for v in validation if not v["match"]),
            },
            "by_day": by_day(measured),
        },
        "samples": measured,
        "post_receipt_window_samples": [s for s in measured if s["has_receipts"]],
        "samples_with_first_send": [s for s in measured if s["first_send_basis"]],
    }

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=False)
        print(f"wrote {args.json_out}")

    md = render_markdown(measured, agg, window)
    if args.md_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.md_out)), exist_ok=True)
        with open(args.md_out, "w") as fh:
            fh.write(md + "\n")
        print(f"wrote {args.md_out}")

    q = empirical["observed_clean_quiet_ms"]
    print(f"turns={len(measured)} receipt_turns={len(receipt_cohort)} quiet_n={q['n']} "
          f"(median {q['median']} ms p90 {q['p90']} worst {q['worst']})")
    for cls, block in agg["by_class"].items():
        ttfs = block["time_to_first_send_ms"]
        tot = block["total_ms"]
        print(f"  [{cls}] n={block['quiet_window_ms']['n']} "
              f"quiet_med={block['quiet_window_ms']['median']} "
              f"ttfs n={ttfs['n']} med={ttfs['median']} p90={ttfs['p90']} worst={ttfs['worst']} | "
              f"total n={tot['n']} med={tot['median']} p90={tot['p90']} worst={tot['worst']}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
