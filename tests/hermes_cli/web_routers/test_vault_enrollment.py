"""HTTP enrollment routes: TLS/Host/Origin, initData, single-use, vault handoff."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.vault_store import get_vault_store
from agent.vault_telegram_enrollment import clear_challenges_for_tests
from hermes_cli.web_routers import vault_enrollment as ve


_FAKE_BOT_TOKEN = "000000000:TEST_BOT_TOKEN_FOR_TESTS_ONLY"
_PUBLIC = "https://vault.test"
_OWNER = "42"
_PASSWORD = "test-only-password-not-real"


def _sign_init_data(fields: dict, bot_token: str = _FAKE_BOT_TOKEN) -> str:
    pairs = dict(fields)
    check = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    pairs["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(pairs)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _FAKE_BOT_TOKEN)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", _OWNER)
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {
            "enabled": True,
            "public_origin": _PUBLIC,
            "init_data_max_age_seconds": 300,
            "challenge_ttl_seconds": 300,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.web_routers.vault_enrollment._section_int",
        lambda key, default: default,
    )
    # Treat TestClient as HTTPS without a real TLS terminator.
    monkeypatch.setattr(ve, "_tls_ok", lambda request: True)
    clear_challenges_for_tests()
    app = FastAPI()
    app.include_router(ve.router)
    client = TestClient(app)
    yield client, home
    clear_challenges_for_tests()


def _headers():
    return {"Host": "vault.test", "Origin": _PUBLIC}


def _init(user_id: str = _OWNER, chat_type: str = "private", skew: int = 0) -> str:
    return _sign_init_data({
        "user": json.dumps({"id": int(user_id), "first_name": "T"}),
        "auth_date": str(int(time.time()) + skew),
        "chat_type": chat_type,
    })


def test_page_disabled_when_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {"enabled": False, "public_origin": _PUBLIC},
    )
    monkeypatch.setattr(ve, "_tls_ok", lambda request: True)
    app = FastAPI()
    app.include_router(ve.router)
    resp = TestClient(app).get("/vault/enroll", headers={"Host": "vault.test"})
    assert resp.status_code == 404


def test_begin_rejects_bad_init_data(app_env):
    client, _ = app_env
    resp = client.post(
        "/api/vault/enroll/begin",
        headers={**_headers(), "X-Telegram-Init-Data": "user=%7B%22id%22%3A42%7D&auth_date=1&hash=00"},
        json={},
    )
    assert resp.status_code == 401


def test_begin_rejects_non_owner(app_env):
    client, _ = app_env
    init = _init(user_id="777")
    resp = client.post(
        "/api/vault/enroll/begin",
        headers={**_headers(), "X-Telegram-Init-Data": init},
        json={"init_data": init},
    )
    assert resp.status_code == 401


def test_begin_rejects_host_mismatch(app_env):
    client, _ = app_env
    init = _init()
    resp = client.post(
        "/api/vault/enroll/begin",
        headers={"Host": "evil.test", "Origin": _PUBLIC, "X-Telegram-Init-Data": init},
        json={"init_data": init},
    )
    assert resp.status_code == 403


def test_enroll_happy_path_and_replay(app_env, monkeypatch):
    client, _ = app_env
    notified = []

    def _fake_notify(**kwargs):
        notified.append({k: v for k, v in kwargs.items() if k != "bot_token"})

    monkeypatch.setattr(ve, "notify_telegram_status", _fake_notify)

    init = _init()
    begin = client.post(
        "/api/vault/enroll/begin",
        headers={**_headers(), "X-Telegram-Init-Data": init},
        json={"init_data": init},
    )
    assert begin.status_code == 200
    challenge_id = begin.json()["challenge_id"]

    body = {
        "challenge_id": challenge_id,
        "init_data": init,
        "origin": "https://example.com",
        "label": "Example",
        "identifier_type": "email",
        "identifier": "user@example.com",
        "password": _PASSWORD,
    }
    saved = client.post(
        "/api/vault/enroll",
        headers={**_headers(), "X-Telegram-Init-Data": init},
        json=body,
    )
    assert saved.status_code == 200
    assert saved.json()["ok"] is True
    assert _PASSWORD not in saved.text
    assert notified and notified[0]["ok"] is True
    assert _PASSWORD not in json.dumps(notified)

    items = get_vault_store().list_items()
    assert len(items) == 1
    assert items[0].label == "Example"
    secret = get_vault_store().resolve_secret(items[0].id)
    assert secret["password"] == _PASSWORD

    replay = client.post(
        "/api/vault/enroll",
        headers={**_headers(), "X-Telegram-Init-Data": init},
        json=body,
    )
    assert replay.status_code == 400


def test_mini_app_html_has_no_send_data(app_env):
    client, _ = app_env
    resp = client.get("/vault/enroll", headers=_headers())
    assert resp.status_code == 200
    assert "sendData" not in resp.text
    assert "Telegram.WebApp.sendData" not in resp.text


def test_raw_forwarded_proto_cannot_bypass_http_guard(tmp_path, monkeypatch):
    """Client-supplied X-Forwarded-Proto must not turn plain HTTP into TLS-ok."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _FAKE_BOT_TOKEN)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", _OWNER)
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {"enabled": True, "public_origin": _PUBLIC},
    )
    # Do NOT monkeypatch _tls_ok — exercise the real ASGI-scheme check.
    app = FastAPI()
    app.include_router(ve.router)
    client = TestClient(app)
    resp = client.post(
        "/api/vault/enroll/begin",
        headers={
            **_headers(),
            "X-Forwarded-Proto": "https",
            "X-Telegram-Init-Data": _init(),
        },
        json={},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "HTTPS required"


def test_tls_ok_follows_trusted_proxy_scheme_only():
    """ProxyHeadersMiddleware may promote scheme only for trusted peers."""
    import asyncio

    from starlette.requests import Request
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    from hermes_cli.web_server_lifecycle import _dashboard_forwarded_allow_ips

    trusted = _dashboard_forwarded_allow_ips({"trusted_proxies": ["172.18.0.0/16"]})

    async def observed_tls(peer: str) -> bool:
        seen: dict[str, bool] = {}

        async def downstream(scope, receive, send):
            seen["ok"] = ve._tls_ok(Request(scope))

        middleware = ProxyHeadersMiddleware(downstream, trusted_hosts=trusted)
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/vault/enroll/begin",
            "raw_path": b"/api/vault/enroll/begin",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"x-forwarded-proto", b"https")],
            "client": (peer, 43120),
            "server": ("vault.test", 443),
        }

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        await middleware(scope, receive, send)
        return seen["ok"]

    assert asyncio.run(observed_tls("172.18.0.9")) is True
    assert asyncio.run(observed_tls("127.0.0.1")) is True
    assert asyncio.run(observed_tls("198.51.100.9")) is False


def test_begin_accepts_https_asgi_scheme(tmp_path, monkeypatch):
    """When the ASGI scheme is already https, enrollment proceeds past TLS gate."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _FAKE_BOT_TOKEN)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", _OWNER)
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {
            "enabled": True,
            "public_origin": _PUBLIC,
            "init_data_max_age_seconds": 300,
            "challenge_ttl_seconds": 300,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.web_routers.vault_enrollment._section_int",
        lambda key, default: default,
    )
    clear_challenges_for_tests()
    app = FastAPI()
    app.include_router(ve.router)

    # Starlette TestClient defaults to http://; force https base URL.
    client = TestClient(app, base_url="https://vault.test")
    init = _init()
    resp = client.post(
        "/api/vault/enroll/begin",
        headers={**_headers(), "X-Telegram-Init-Data": init},
        json={"init_data": init},
    )
    assert resp.status_code == 200
    assert "challenge_id" in resp.json()
    clear_challenges_for_tests()
