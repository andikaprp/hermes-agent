"""Telegram Mini App → encrypted vault enrollment (credentials never transit Bot API).

Private-chat, owner-allowlisted users open a short-lived HTTPS Mini App form. The
page posts origin/label/identifier/password directly to the Hermes dashboard over
TLS. Telegram ``initData`` is HMAC-validated server-side with the bot token;
single-use challenges bind the submit to that authenticated user. Credential
values are never placed in Telegram updates, callback_data, logs, transcripts,
or model context — only non-secret status reaches the chat after save.

Disabled unless every prerequisite is present (explicit opt-in + HTTPS public
origin + bot token + owner allowlist). See docs: credential-vault.md § Telegram.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Default freshness / TTL — short enough that a stolen initData window is narrow.
DEFAULT_INIT_DATA_MAX_AGE_SECONDS = 300
DEFAULT_CHALLENGE_TTL_SECONDS = 300

_CHALLENGE_LOCK = threading.Lock()
# In-process store: challenge_id → record. Cleared on consume/expiry. No secrets.
_CHALLENGES: Dict[str, Dict[str, Any]] = {}


class EnrollmentError(Exception):
    """Safe-to-surface enrollment failure (never contains credential values)."""


@dataclass(frozen=True)
class EnrollmentPrereqs:
    """Resolved readiness for Telegram vault enrollment."""

    ready: bool
    reason: str = ""
    public_origin: str = ""
    enroll_url: str = ""
    bot_token_present: bool = False
    allowlist: Tuple[str, ...] = ()


def _cfg_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        vault = load_config().get("vault") or {}
        section = vault.get("telegram_enrollment") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _env_csv(name: str) -> Tuple[str, ...]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return ()
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def normalize_public_origin(value: str) -> str:
    """Require ``https://host[:port]`` with no path/query/fragment (path is fixed)."""
    raw = (value or "").strip().rstrip("/")
    if not raw:
        raise EnrollmentError("public_origin is required")
    parts = urlsplit(raw)
    if parts.scheme.lower() != "https":
        raise EnrollmentError("public_origin must be https")
    if not parts.hostname:
        raise EnrollmentError("public_origin must include a host")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise EnrollmentError("public_origin must be scheme://host[:port] only (no path)")
    host = parts.hostname.lower()
    port = parts.port
    if port and port != 443:
        return f"https://{host}:{port}"
    return f"https://{host}"


def enrollment_prereqs(*, config: Optional[Mapping[str, Any]] = None) -> EnrollmentPrereqs:
    """Feature is ready only when every gate is satisfied. Never auto-enables."""
    section = dict(config) if config is not None else _cfg_section()
    if not section.get("enabled"):
        return EnrollmentPrereqs(ready=False, reason="vault.telegram_enrollment.enabled is false (default)")
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    allowlist = _env_csv("TELEGRAM_ALLOWED_USERS")
    if not token:
        return EnrollmentPrereqs(ready=False, reason="TELEGRAM_BOT_TOKEN is not set")
    if not allowlist:
        return EnrollmentPrereqs(
            ready=False, reason="TELEGRAM_ALLOWED_USERS is empty (owner allowlist required)",
            bot_token_present=True)
    try:
        origin = normalize_public_origin(str(section.get("public_origin") or ""))
    except EnrollmentError as exc:
        return EnrollmentPrereqs(
            ready=False, reason=str(exc), bot_token_present=True, allowlist=allowlist)
    return EnrollmentPrereqs(
        ready=True,
        public_origin=origin,
        enroll_url=f"{origin}/vault/enroll",
        bot_token_present=True,
        allowlist=allowlist,
    )


def validate_telegram_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int = DEFAULT_INIT_DATA_MAX_AGE_SECONDS,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Cryptographically validate Telegram WebApp ``initData``; never trust initDataUnsafe.

    Returns parsed fields including ``user`` (dict) and ``auth_date`` (int). Raises
    :class:`EnrollmentError` on any failure (bad HMAC, stale ``auth_date``, missing user).
    """
    if not init_data or not isinstance(init_data, str):
        raise EnrollmentError("initData is required")
    if not bot_token:
        raise EnrollmentError("bot token is required")

    pairs = parse_qsl(init_data, keep_blank_values=True)
    data = {k: v for k, v in pairs}
    received_hash = data.pop("hash", None)
    if not received_hash:
        raise EnrollmentError("initData missing hash")

    # Official algorithm: exclude hash, sort keys, join with newlines; HMAC with
    # secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token).
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    computed = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed, received_hash):
        raise EnrollmentError("initData HMAC validation failed")

    try:
        auth_date = int(data.get("auth_date") or 0)
    except (TypeError, ValueError) as exc:
        raise EnrollmentError("initData auth_date is invalid") from exc
    if auth_date <= 0:
        raise EnrollmentError("initData auth_date is invalid")
    age = (now if now is not None else time.time()) - auth_date
    if age > max_age_seconds or age < -60:
        raise EnrollmentError("initData expired or not yet valid")

    user_raw = data.get("user")
    if not user_raw:
        raise EnrollmentError("initData missing user")
    try:
        user = json.loads(user_raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise EnrollmentError("initData user is invalid") from exc
    if not isinstance(user, dict) or user.get("id") is None:
        raise EnrollmentError("initData user is invalid")

    chat_type = (data.get("chat_type") or "").strip().lower()
    # When Telegram includes chat_type, require a private conversation.
    if chat_type and chat_type not in {"private", "sender"}:
        raise EnrollmentError("enrollment is private-chat only")

    return {
        "user_id": str(user["id"]),
        "user": user,
        "auth_date": auth_date,
        "chat_type": chat_type or "private",
        "raw": data,
    }


def assert_owner_allowlisted(user_id: str, allowlist: Optional[Tuple[str, ...]] = None) -> None:
    allowed = allowlist if allowlist is not None else _env_csv("TELEGRAM_ALLOWED_USERS")
    if not allowed:
        raise EnrollmentError("owner allowlist is empty")
    if str(user_id) not in allowed:
        raise EnrollmentError("user is not on the owner allowlist")


def _purge_expired_locked(now: float) -> None:
    dead = [cid for cid, rec in _CHALLENGES.items() if float(rec.get("expires_at", 0)) <= now]
    for cid in dead:
        _CHALLENGES.pop(cid, None)


def mint_challenge(
    *,
    telegram_user_id: str,
    chat_id: str,
    ttl_seconds: int = DEFAULT_CHALLENGE_TTL_SECONDS,
    now: Optional[float] = None,
) -> str:
    """Create a single-use enrollment challenge bound to ``telegram_user_id``."""
    assert_owner_allowlisted(telegram_user_id)
    ts = now if now is not None else time.time()
    challenge_id = secrets.token_urlsafe(24)
    with _CHALLENGE_LOCK:
        _purge_expired_locked(ts)
        # One live challenge per user — supersede older ones.
        for cid, rec in list(_CHALLENGES.items()):
            if rec.get("telegram_user_id") == str(telegram_user_id):
                _CHALLENGES.pop(cid, None)
        _CHALLENGES[challenge_id] = {
            "telegram_user_id": str(telegram_user_id),
            "chat_id": str(chat_id),
            "created_at": ts,
            "expires_at": ts + max(30, int(ttl_seconds)),
            "consumed": False,
        }
    return challenge_id


def consume_challenge(
    challenge_id: str,
    *,
    telegram_user_id: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Atomically consume a challenge; raises on missing/expired/replay/user mismatch."""
    if not challenge_id:
        raise EnrollmentError("challenge is required")
    ts = now if now is not None else time.time()
    with _CHALLENGE_LOCK:
        _purge_expired_locked(ts)
        rec = _CHALLENGES.get(challenge_id)
        if rec is None:
            raise EnrollmentError("challenge is invalid or expired")
        if rec.get("consumed"):
            raise EnrollmentError("challenge already used")
        if float(rec.get("expires_at", 0)) <= ts:
            _CHALLENGES.pop(challenge_id, None)
            raise EnrollmentError("challenge is invalid or expired")
        if str(rec.get("telegram_user_id")) != str(telegram_user_id):
            raise EnrollmentError("challenge does not match authenticated user")
        rec["consumed"] = True
        _CHALLENGES.pop(challenge_id, None)
        return dict(rec)


def clear_challenges_for_tests() -> None:
    """Test seam: drop every in-memory challenge."""
    with _CHALLENGE_LOCK:
        _CHALLENGES.clear()


def request_origin_allowed(request_origin: str, public_origin: str) -> bool:
    """Strict Origin check against the configured HTTPS public origin."""
    try:
        expected = normalize_public_origin(public_origin)
        got = normalize_public_origin(request_origin) if "://" in (request_origin or "") else ""
    except EnrollmentError:
        return False
    return bool(got) and hmac.compare_digest(got, expected)


def request_host_allowed(host_header: str, public_origin: str) -> bool:
    """Strict Host check against the configured public origin's host[:port]."""
    try:
        expected = normalize_public_origin(public_origin)
    except EnrollmentError:
        return False
    parts = urlsplit(expected)
    expected_host = parts.hostname or ""
    expected_port = parts.port
    raw = (host_header or "").strip().lower()
    if not raw or "://" in raw:
        return False
    if raw.startswith("["):
        return False  # IPv6 Mini App hosts are out of scope for this slice
    if ":" in raw:
        host, port_s = raw.rsplit(":", 1)
        if not port_s.isdigit():
            return False
        port = int(port_s)
    else:
        host, port = raw, None
    if host != expected_host.lower():
        return False
    if expected_port:
        return port == expected_port
    return port in (None, 443)


def save_login_from_enrollment(
    *,
    origin: str,
    label: str,
    identifier: str,
    identifier_type: str,
    password: str,
) -> Dict[str, str]:
    """Write a login item through the existing encrypted vault store. Never logs secrets."""
    from agent.vault_store import VaultError, get_vault_store, scrub_secret_from_text

    secret = {
        "identifier_type": identifier_type,
        "identifier": identifier,
        "password": password,
    }
    try:
        meta = get_vault_store().add_item(
            kind="login",
            label=label,
            origin=origin,
            secret=secret,
        )
        return {"id": meta.id, "label": meta.label, "origin": meta.origin or ""}
    except VaultError as exc:
        raise EnrollmentError(str(exc)) from None
    except Exception as exc:
        raise EnrollmentError(scrub_secret_from_text(str(exc), secret) or "vault write failed") from None
    finally:
        secret.clear()


def notify_telegram_status(
    *,
    bot_token: str,
    chat_id: str,
    ok: bool,
    label: str = "",
) -> None:
    """Best-effort Bot API status only — never includes credential values."""
    text = (
        f"✅ Saved login{f' “{label}”' if label else ''} to Hermes vault."
        if ok
        else "❌ Could not save login to Hermes vault. Try again from a private chat."
    )
    try:
        import urllib.error
        import urllib.request

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — fixed Telegram API host
            resp.read(256)
    except Exception:
        # Never log chat payloads or tokens; status delivery is best-effort.
        logger.info("vault telegram enrollment: status notify failed (non-secret)")


def hermes_home_marker() -> str:
    """Profile home key for diagnostics (never a secret)."""
    return str(get_hermes_home())
