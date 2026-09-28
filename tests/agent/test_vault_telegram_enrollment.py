"""Telegram vault enrollment: initData HMAC, challenges, private/owner gates, vault handoff.

Fixtures use fake bot tokens and passwords only — never real credentials.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest

from agent.vault_store import get_vault_store
from agent.vault_telegram_enrollment import (
    EnrollmentError,
    assert_owner_allowlisted,
    clear_challenges_for_tests,
    consume_challenge,
    enrollment_prereqs,
    mint_challenge,
    normalize_public_origin,
    request_host_allowed,
    request_origin_allowed,
    save_login_from_enrollment,
    validate_telegram_init_data,
)


_FAKE_BOT_TOKEN = "000000000:TEST_BOT_TOKEN_FOR_TESTS_ONLY"
_FAKE_PASSWORD = "test-only-password-not-real"


def _sign_init_data(fields: dict, bot_token: str = _FAKE_BOT_TOKEN) -> str:
    """Build a Telegram-shaped initData query string with a valid HMAC."""
    pairs = {k: v for k, v in fields.items()}
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    digest = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    pairs["hash"] = digest
    return urlencode(pairs)


@pytest.fixture(autouse=True)
def _clean_challenges():
    clear_challenges_for_tests()
    yield
    clear_challenges_for_tests()


@pytest.fixture
def owner_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _FAKE_BOT_TOKEN)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42,99")
    return home


def test_prereqs_disabled_by_default(owner_env, monkeypatch):
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {"enabled": False, "public_origin": "https://vault.example"},
    )
    pr = enrollment_prereqs()
    assert pr.ready is False
    assert "enabled is false" in pr.reason


def test_prereqs_require_https_origin_and_allowlist(owner_env, monkeypatch):
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {"enabled": True, "public_origin": "http://insecure.example"},
    )
    assert enrollment_prereqs().ready is False

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "")
    monkeypatch.setattr(
        "agent.vault_telegram_enrollment._cfg_section",
        lambda: {"enabled": True, "public_origin": "https://vault.example"},
    )
    assert enrollment_prereqs().ready is False

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    assert enrollment_prereqs().ready is True
    assert enrollment_prereqs().enroll_url == "https://vault.example/vault/enroll"


def test_init_data_rejects_bad_hmac_and_expiry(owner_env):
    user = json.dumps({"id": 42, "first_name": "A"})
    now = int(time.time())
    good = _sign_init_data({"user": user, "auth_date": str(now), "chat_type": "private"})
    parsed = validate_telegram_init_data(good, _FAKE_BOT_TOKEN, now=float(now))
    assert parsed["user_id"] == "42"

    bad = good[:-4] + "dead"
    with pytest.raises(EnrollmentError, match="HMAC"):
        validate_telegram_init_data(bad, _FAKE_BOT_TOKEN, now=float(now))

    stale = _sign_init_data({"user": user, "auth_date": str(now - 10_000), "chat_type": "private"})
    with pytest.raises(EnrollmentError, match="expired"):
        validate_telegram_init_data(stale, _FAKE_BOT_TOKEN, max_age_seconds=300, now=float(now))


def test_init_data_rejects_group_chat_type(owner_env):
    user = json.dumps({"id": 42})
    now = int(time.time())
    group = _sign_init_data({"user": user, "auth_date": str(now), "chat_type": "group"})
    with pytest.raises(EnrollmentError, match="private-chat"):
        validate_telegram_init_data(group, _FAKE_BOT_TOKEN, now=float(now))


def test_owner_allowlist_gate(owner_env):
    assert_owner_allowlisted("42")
    with pytest.raises(EnrollmentError, match="allowlist"):
        assert_owner_allowlisted("777")


def test_challenge_single_use_and_expiry(owner_env):
    cid = mint_challenge(telegram_user_id="42", chat_id="42", ttl_seconds=60, now=1000.0)
    consume_challenge(cid, telegram_user_id="42", now=1001.0)
    with pytest.raises(EnrollmentError, match="invalid|used|expired"):
        consume_challenge(cid, telegram_user_id="42", now=1002.0)

    cid2 = mint_challenge(telegram_user_id="42", chat_id="42", ttl_seconds=60, now=2000.0)
    with pytest.raises(EnrollmentError, match="expired"):
        # mint_challenge floors TTL at 30s; jump well past expiry.
        consume_challenge(cid2, telegram_user_id="42", now=2100.0)

    cid3 = mint_challenge(telegram_user_id="42", chat_id="42", ttl_seconds=60, now=3000.0)
    with pytest.raises(EnrollmentError, match="match"):
        consume_challenge(cid3, telegram_user_id="99", now=3001.0)


def test_host_origin_checks():
    origin = "https://vault.example"
    assert normalize_public_origin(origin) == origin
    assert request_host_allowed("vault.example", origin)
    assert request_host_allowed("vault.example:443", origin)
    assert not request_host_allowed("evil.example", origin)
    assert request_origin_allowed("https://vault.example", origin)
    assert not request_origin_allowed("https://evil.example", origin)
    with pytest.raises(EnrollmentError):
        normalize_public_origin("http://vault.example")


def test_save_login_encrypted_handoff(owner_env):
    meta = save_login_from_enrollment(
        origin="https://example.com",
        label="Example",
        identifier="user@example.com",
        identifier_type="email",
        password=_FAKE_PASSWORD,
    )
    assert meta["id"].startswith("vault_")
    store = get_vault_store()
    listed = store.list_items()
    assert len(listed) == 1
    assert listed[0].label == "Example"
    assert listed[0].identifier == "user@example.com"
    # Password is in the encrypted payload, not metadata.
    blob = (store._vault_path.read_bytes())
    assert _FAKE_PASSWORD.encode() not in blob
    secret = store.resolve_secret(meta["id"])
    assert secret["password"] == _FAKE_PASSWORD


def test_mini_app_html_never_uses_send_data():
    """Credentials must not ride Telegram.WebApp.sendData / Bot API updates."""
    from hermes_cli.web_routers import vault_enrollment as ve

    assert "sendData" not in ve._MINI_APP_HTML
    assert "web_app_data" not in ve._MINI_APP_HTML.lower()
