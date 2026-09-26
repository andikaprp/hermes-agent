"""Gateway-wide outbound delivery outcome model (privacy-safe).

Explicit stages (evidence-gated, never invent proof):

* ``prepared``  — payload ready for an outbound path
* ``attempted`` — a send adapter / transport write was invoked
* ``delivered`` — provider/platform acceptance/receipt (never client render)
* ``confirmed`` — explicit user/client acknowledgement only

Rules that shape every instrumented seam:

* Local file saves are **not** delivered (terminal ``local_saved``).
* Calling a send adapter is **only** ``attempted``.
* Provider acceptance (message id / ts) may establish ``delivered``, never ``confirmed``.
* Web chat server enqueue / SSE write is at most ``attempted``; without concrete client
  evidence, ``delivered``/``confirmed`` stay ``unavailable``.
* Filtered / suppressed sends must not read as success.

The carrier is anonymous — allowlisted fields only (no message body, raw chat/user ids,
tokens, or secrets). Digests share the per-process salt with
:mod:`gateway.telegram_delivery_receipt` so Telegram receipts and gateway outcomes
correlate inside one process. Emitting never raises.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.delivery_outcome")

LOG_PREFIX = "gateway_delivery_outcome"

# Digests — same convention as telegram_delivery_receipt (shared salt below).
CHAT_DIGEST_PREFIX = "c"
MESSAGE_DIGEST_PREFIX = "m"
DIGEST_CHARS = 12
MISSING_TOKEN = "none"

STAGE_PREPARED = "prepared"
STAGE_ATTEMPTED = "attempted"
STAGE_DELIVERED = "delivered"
STAGE_CONFIRMED = "confirmed"
STAGES = (STAGE_PREPARED, STAGE_ATTEMPTED, STAGE_DELIVERED, STAGE_CONFIRMED)
_STAGE_RANK = {name: idx for idx, name in enumerate(STAGES)}

# Terminal statuses that are not a successful stage advancement.
STATUS_FAILED = "failed"
STATUS_FILTERED = "filtered"
STATUS_LOCAL_SAVED = "local_saved"
STATUS_UNAVAILABLE = "unavailable"

EVIDENCE_NONE = "none"
EVIDENCE_LOCAL_FILE = "local_file"
EVIDENCE_ADAPTER_CALL = "adapter_call"
EVIDENCE_PROVIDER_ACCEPT = "provider_accept"
EVIDENCE_CLIENT_ACK = "client_ack"
EVIDENCE_SSE_WRITE = "sse_write"
EVIDENCE_SSE_ENQUEUE = "sse_enqueue"
EVIDENCE_FILTERED = "filtered"
EVIDENCE_UNAVAILABLE = "unavailable"
EVIDENCE_FAILURE = "failure"
EVIDENCE_DEAD_TARGET = "dead_target"
EVIDENCE_NO_TRANSPORT = "no_transport"

# Evidence that may justify each stage (anything else is refused → unavailable).
_EVIDENCE_FOR_STAGE = {
    STAGE_PREPARED: frozenset({EVIDENCE_NONE, EVIDENCE_SSE_ENQUEUE}),
    STAGE_ATTEMPTED: frozenset({EVIDENCE_ADAPTER_CALL, EVIDENCE_SSE_WRITE}),
    STAGE_DELIVERED: frozenset({EVIDENCE_PROVIDER_ACCEPT}),
    STAGE_CONFIRMED: frozenset({EVIDENCE_CLIENT_ACK}),
}

CHANNEL_TELEGRAM = "telegram"
CHANNEL_SLACK = "slack"
CHANNEL_WEB_CHAT = "web_chat"
CHANNEL_LOCAL = "local"
CHANNEL_UNKNOWN = "unknown"

# Per-process correlation salt (shared with telegram receipts via re-export).
_PROCESS_SALT = os.urandom(32)


def redacted_token(value: Any, *, prefix: str) -> str:
    """Short keyed digest of *value* (``none`` when absent/blank); never the raw value."""
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip()
    if not text:
        return MISSING_TOKEN
    digest = hmac.new(_PROCESS_SALT, text.encode("utf-8"), hashlib.sha256).hexdigest()[:DIGEST_CHARS]
    return f"{prefix}{digest}"


def normalize_channel(platform: Any) -> str:
    """Map a Platform / string to a stable channel token for the outcome log."""
    if platform is None:
        return CHANNEL_UNKNOWN
    raw = getattr(platform, "value", platform)
    name = str(raw or "").strip().lower()
    if not name:
        return CHANNEL_UNKNOWN
    if name in {"api_server", "web", "webchat", "web_chat", "session_chat"}:
        return CHANNEL_WEB_CHAT
    if name == "local":
        return CHANNEL_LOCAL
    return name


class DeliveryOutcome:
    """One outbound attempt's anonymous outcome. Stage transitions are evidence-gated.

    Emit helpers are idempotent per terminal emit and never raise — bookkeeping must
    not break a live send.
    """

    __slots__ = (
        "_clock", "_logger", "_channel", "_chat", "_mid", "_stage", "_status",
        "_evidence", "_attempt", "_emitted_keys",
    )

    def __init__(
        self,
        *,
        channel: Any,
        chat_id: Any = None,
        attempt: int = 1,
        clock: Callable[[], float] = time.monotonic,
        outcome_logger: Optional[logging.Logger] = None,
    ) -> None:
        self._clock = clock
        self._logger = outcome_logger or logger
        self._channel = normalize_channel(channel)
        self._chat = redacted_token(chat_id, prefix=CHAT_DIGEST_PREFIX)
        self._mid = MISSING_TOKEN
        self._stage: Optional[str] = None
        self._status: Optional[str] = None
        self._evidence = EVIDENCE_NONE
        self._attempt = int(attempt)
        self._emitted_keys: set[str] = set()

    @property
    def stage(self) -> Optional[str]:
        return self._stage

    @property
    def status(self) -> Optional[str]:
        return self._status

    @property
    def evidence(self) -> str:
        return self._evidence

    @property
    def channel(self) -> str:
        return self._channel

    def fields(self) -> dict[str, Any]:
        """Allowlisted payload only — nothing derived from message content."""
        return {
            "mono": round(float(self._clock()), 3),
            "channel": self._channel,
            "chat": self._chat,
            "attempt": self._attempt,
            "mid": self._mid,
            "stage": self._stage or MISSING_TOKEN,
            "status": self._status or MISSING_TOKEN,
            "evidence": self._evidence,
        }

    def prepared(self, *, evidence: str = EVIDENCE_NONE) -> None:
        self._advance(STAGE_PREPARED, evidence=evidence)

    def attempted(self, *, evidence: str = EVIDENCE_ADAPTER_CALL) -> None:
        self._advance(STAGE_ATTEMPTED, evidence=evidence)

    def delivered(self, message_id: Any = None, *, evidence: str = EVIDENCE_PROVIDER_ACCEPT) -> None:
        """Provider/platform acceptance only. Refuses non-provider evidence."""
        if evidence not in _EVIDENCE_FOR_STAGE[STAGE_DELIVERED]:
            self.unavailable(for_stage=STAGE_DELIVERED, evidence=EVIDENCE_UNAVAILABLE)
            return
        if message_id is not None:
            self._mid = redacted_token(message_id, prefix=MESSAGE_DIGEST_PREFIX)
        if self._mid == MISSING_TOKEN:
            # Acceptance without a receipt id is not enough to claim delivered.
            self.unavailable(for_stage=STAGE_DELIVERED, evidence=EVIDENCE_UNAVAILABLE)
            return
        self._advance(STAGE_DELIVERED, evidence=evidence)

    def confirmed(self, *, evidence: str = EVIDENCE_CLIENT_ACK) -> None:
        """Explicit client/user acknowledgement only. Refuses weaker evidence."""
        if evidence not in _EVIDENCE_FOR_STAGE[STAGE_CONFIRMED]:
            self.unavailable(for_stage=STAGE_CONFIRMED, evidence=EVIDENCE_UNAVAILABLE)
            return
        self._advance(STAGE_CONFIRMED, evidence=evidence)

    def failed(self, *, evidence: str = EVIDENCE_FAILURE) -> None:
        self._status = STATUS_FAILED
        self._evidence = evidence or EVIDENCE_FAILURE
        self._emit(key=f"failed:{self._stage or 'none'}")

    def record_attempt_failed(self, *, evidence: str = EVIDENCE_FAILURE) -> None:
        """Mark a raised outbound attempt as failed. Never raises — bookkeeping must not
        mask or replace the original send exception."""
        with contextlib.suppress(Exception):
            self.failed(evidence=evidence)

    def filtered(self, *, evidence: str = EVIDENCE_FILTERED) -> None:
        """Suppressed / filtered — not a successful delivery."""
        self._status = STATUS_FILTERED
        self._evidence = evidence or EVIDENCE_FILTERED
        self._emit(key=f"filtered:{self._stage or 'none'}")

    def local_saved(self) -> None:
        """Local file write completed — explicitly not delivered."""
        self._status = STATUS_LOCAL_SAVED
        self._evidence = EVIDENCE_LOCAL_FILE
        # Stage may be prepared (payload ready) but must never claim delivered.
        if self._stage is None:
            self._stage = STAGE_PREPARED
        self._emit(key="local_saved")

    def unavailable(self, *, for_stage: str, evidence: str = EVIDENCE_UNAVAILABLE) -> None:
        """Higher stage cannot be proven — record that gap instead of inventing proof."""
        self._status = STATUS_UNAVAILABLE
        self._evidence = evidence or EVIDENCE_UNAVAILABLE
        self._emit(key=f"unavailable:{for_stage}")

    def apply_send_result(self, result: Any) -> None:
        """After an adapter call (already ``attempted``): deliver on provider accept, else fail.

        Never marks ``confirmed``. Success without a message id → unavailable delivered.
        """
        get = result.get if isinstance(result, dict) else (
            lambda name, default=None: getattr(result, name, default))
        success = get("success", True) is not False
        if not success:
            self.failed()
            return
        message_id = get("message_id", None)
        if message_id is None and isinstance(result, dict):
            message_id = (result.get("result") or {}).get("message_id") if isinstance(
                result.get("result"), dict) else None
        self.delivered(message_id, evidence=EVIDENCE_PROVIDER_ACCEPT)

    def _advance(self, stage: str, *, evidence: str) -> None:
        allowed = _EVIDENCE_FOR_STAGE.get(stage, frozenset())
        if evidence not in allowed:
            self.unavailable(for_stage=stage, evidence=EVIDENCE_UNAVAILABLE)
            return
        current_rank = _STAGE_RANK.get(self._stage, -1)
        next_rank = _STAGE_RANK[stage]
        if next_rank < current_rank:
            return
        self._stage = stage
        self._evidence = evidence
        # A successful stage advance clears a prior non-terminal status.
        if self._status in (STATUS_UNAVAILABLE, None):
            self._status = None
        self._emit(key=f"stage:{stage}")

    def _emit(self, *, key: str) -> None:
        if key in self._emitted_keys:
            return
        self._emitted_keys.add(key)
        payload = self.fields()
        with contextlib.suppress(Exception):
            self._logger.info(
                LOG_PREFIX + " %s",
                " ".join(f"{k}={v}" for k, v in payload.items()),
                extra={"delivery_outcome": payload},
            )


def observe_send_result(
    *,
    channel: Any,
    chat_id: Any,
    result: Any,
    attempt: int = 1,
    prepared: bool = True,
) -> DeliveryOutcome:
    """Convenience for shared seams: prepared → attempted → apply_send_result."""
    outcome = DeliveryOutcome(channel=channel, chat_id=chat_id, attempt=attempt)
    if prepared:
        outcome.prepared()
    outcome.attempted(evidence=EVIDENCE_ADAPTER_CALL)
    outcome.apply_send_result(result)
    return outcome
