"""Gemini native client OAuth vs API-key auth headers.

Regression for LAB-50: when ``gemini.auth: oauth`` / ``access_token`` is set,
``GeminiNativeClient`` must send ``Authorization: Bearer`` and not ``x-goog-api-key``;
API-key mode keeps the key header; missing/expired OAuth tokens fail closed.
"""

from __future__ import annotations

import time

import pytest


FAKE_ACCESS = "ya29.fake-access-token-for-tests"
FAKE_REFRESH = "1//fake-refresh-token-for-tests"
FAKE_API_KEY = "AIza-fake-api-key-for-tests"


class _RecordingHTTP:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})

        class _Resp:
            status_code = 200
            text = "{}"
            headers = {}

            def json(self):
                return {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }

        return _Resp()

    def close(self):
        return None


def test_native_client_sends_bearer_when_oauth_access_token_configured():
    from agent.gemini_native_adapter import GeminiNativeClient

    http = _RecordingHTTP()
    client = GeminiNativeClient(
        access_token=FAKE_ACCESS,
        use_oauth=True,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        http_client=http,
    )
    client.chat.completions.create(model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}])
    headers = http.calls[0]["headers"]
    assert headers.get("Authorization") == f"Bearer {FAKE_ACCESS}"
    assert "x-goog-api-key" not in headers


def test_native_client_falls_back_to_api_key_header_without_oauth():
    from agent.gemini_native_adapter import GeminiNativeClient

    http = _RecordingHTTP()
    client = GeminiNativeClient(
        api_key=FAKE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        http_client=http,
    )
    client.chat.completions.create(model="gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}])
    headers = http.calls[0]["headers"]
    assert headers.get("x-goog-api-key") == FAKE_API_KEY
    assert "Authorization" not in headers


def test_native_client_fail_closed_when_oauth_enabled_without_token():
    from agent.gemini_native_adapter import GeminiNativeClient

    with pytest.raises(RuntimeError, match="OAuth"):
        GeminiNativeClient(use_oauth=True, api_key="")


def test_resolve_oauth_access_token_fail_closed_when_missing(tmp_path, monkeypatch):
    from agent.gemini_oauth import GeminiOAuthError, resolve_gemini_oauth_access_token

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(GeminiOAuthError) as excinfo:
        resolve_gemini_oauth_access_token(hermes_home=tmp_path, refresh=False)
    assert excinfo.value.code == "missing_token"


def test_resolve_oauth_access_token_fail_closed_when_expired(tmp_path, monkeypatch):
    from agent.gemini_oauth import GeminiOAuthError, resolve_gemini_oauth_access_token
    from tools.mcp_oauth import HermesTokenStorage, _write_json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("gemini-api", hermes_home=tmp_path)
    _write_json(storage._tokens_path(), {
        "access_token": FAKE_ACCESS,
        "refresh_token": "",
        "token_type": "Bearer",
        "expires_at": time.time() - 10,
        "expires_in": 0,
    })
    with pytest.raises(GeminiOAuthError) as excinfo:
        resolve_gemini_oauth_access_token(hermes_home=tmp_path, refresh=False)
    assert excinfo.value.code == "expired_token"


def test_resolve_oauth_access_token_returns_unexpired(tmp_path, monkeypatch):
    from agent.gemini_oauth import resolve_gemini_oauth_access_token
    from tools.mcp_oauth import HermesTokenStorage, _write_json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("gemini-api", hermes_home=tmp_path)
    _write_json(storage._tokens_path(), {
        "access_token": FAKE_ACCESS,
        "refresh_token": FAKE_REFRESH,
        "token_type": "Bearer",
        "expires_at": time.time() + 3600,
        "expires_in": 3600,
    })
    assert resolve_gemini_oauth_access_token(hermes_home=tmp_path, refresh=False) == FAKE_ACCESS


def test_gemini_native_auth_kwargs_oauth_mode(tmp_path, monkeypatch):
    from agent.gemini_oauth import gemini_native_auth_kwargs
    from tools.mcp_oauth import HermesTokenStorage, _write_json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    storage = HermesTokenStorage("gemini-api", hermes_home=tmp_path)
    _write_json(storage._tokens_path(), {
        "access_token": FAKE_ACCESS,
        "token_type": "Bearer",
        "expires_at": time.time() + 3600,
    })
    kwargs = gemini_native_auth_kwargs(
        FAKE_API_KEY,
        hermes_home=tmp_path,
        config={"gemini": {"auth": "oauth"}},
    )
    assert kwargs["use_oauth"] is True
    assert kwargs["access_token"] == FAKE_ACCESS
    assert "api_key" not in kwargs


def test_gemini_native_auth_kwargs_api_key_mode():
    from agent.gemini_oauth import gemini_native_auth_kwargs

    kwargs = gemini_native_auth_kwargs(FAKE_API_KEY, config={"gemini": {"auth": "api_key"}})
    assert kwargs["api_key"] == FAKE_API_KEY
    assert kwargs.get("use_oauth") is not True


def test_gemini_auth_mode_defaults_to_api_key():
    from agent.gemini_oauth import gemini_auth_mode

    assert gemini_auth_mode({}) == "api_key"
    assert gemini_auth_mode({"gemini": {"auth": "oauth"}}) == "oauth"
    assert gemini_auth_mode({"gemini": {"auth": "bogus"}}) == "api_key"
