"""Google account OAuth for the native Gemini API (AI Studio Generative Language).

Uses the installed-app (Desktop) OAuth flow documented at
https://ai.google.dev/gemini-api/docs/oauth:

* Auth:  ``https://accounts.google.com/o/oauth2/v2/auth``
* Token: ``https://oauth2.googleapis.com/token``
* Scope: ``https://www.googleapis.com/auth/generative-language``

There is no public AI Studio OAuth ``client_id``. The user must create a Desktop
OAuth client in Google Cloud Console (Generative Language API enabled) and place
the downloaded JSON at ``HERMES_HOME/google_client_secret.json``.

Tokens are persisted via ``HermesTokenStorage`` under slug ``gemini-api``
(``HERMES_HOME/mcp-tokens/gemini-api.json``) — same layout as MCP OAuth, no new
secret plumbing.

A Google Workspace OAuth client *can* request this scope if the same GCP project
enables Generative Language API and the consent screen lists the scope; the
client type is not API-bound. Prefer a dedicated Desktop client for Gemini.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# HermesTokenStorage server_name / mcp-tokens/<slug>.json
GEMINI_OAUTH_TOKEN_SLUG = "gemini-api"

GEMINI_OAUTH_SCOPE = "https://www.googleapis.com/auth/generative-language"
AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

CLIENT_SECRET_FILENAME = "google_client_secret.json"
CALLBACK_PATH = "/oauth2callback"
DEFAULT_REDIRECT_PORT = 8765
CALLBACK_WAIT_SECONDS = 300
TOKEN_REQUEST_TIMEOUT_SECONDS = 20.0
REFRESH_SKEW_SECONDS = 60

_AUTH_MODES = frozenset({"api_key", "oauth"})


class GeminiOAuthError(RuntimeError):
    """Fail-closed Gemini OAuth failure (missing client, missing/expired token, refresh)."""

    def __init__(self, message: str, *, code: str = "gemini_oauth_error") -> None:
        super().__init__(message)
        self.code = code


def client_secret_path(*, hermes_home: Path | None = None) -> Path:
    return Path(hermes_home if hermes_home is not None else get_hermes_home()) / CLIENT_SECRET_FILENAME


def gemini_auth_mode(config: Optional[Dict[str, Any]] = None) -> str:
    """``api_key`` (default) or ``oauth`` from ``config.yaml`` ``gemini.auth``."""
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            return "api_key"
    block = config.get("gemini") if isinstance(config, dict) else None
    if not isinstance(block, dict):
        return "api_key"
    mode = str(block.get("auth") or "api_key").strip().lower()
    return mode if mode in _AUTH_MODES else "api_key"


def gemini_oauth_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return gemini_auth_mode(config) == "oauth"


def gemini_native_auth_kwargs(
    api_key: str = "",
    *,
    base_url: Optional[str] = None,
    hermes_home: Path | None = None,
    config: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Kwargs for ``GeminiNativeClient`` honouring ``gemini.auth`` (oauth → Bearer slot)."""
    kwargs: Dict[str, Any] = dict(extra)
    if base_url is not None:
        kwargs["base_url"] = base_url
    if gemini_oauth_enabled(config):
        token = resolve_gemini_oauth_access_token(hermes_home=hermes_home)
        kwargs["access_token"] = token
        kwargs["use_oauth"] = True
        kwargs.pop("api_key", None)
        project_id = gemini_project_id(config)
        if project_id:
            headers = dict(kwargs.get("default_headers") or {})
            headers.setdefault("x-goog-user-project", project_id)
            kwargs["default_headers"] = headers
        return kwargs
    kwargs["api_key"] = api_key
    return kwargs


def gemini_project_id(config: Optional[Dict[str, Any]] = None) -> str:
    """Optional GCP project for ``x-goog-user-project`` (OAuth calls often need it)."""
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            return ""
    block = config.get("gemini") if isinstance(config, dict) else None
    if not isinstance(block, dict):
        return ""
    return str(block.get("project_id") or "").strip()


def load_oauth_client_credentials(*, hermes_home: Path | None = None) -> Tuple[str, str]:
    """Return ``(client_id, client_secret)`` from ``google_client_secret.json``.

    Accepts the Google Cloud Console download shapes ``installed`` / ``web``.
    """
    path = client_secret_path(hermes_home=hermes_home)
    if not path.is_file():
        raise GeminiOAuthError(
            f"Missing OAuth client file at {path}. Create a Desktop OAuth client in Google Cloud "
            "Console (APIs & Services → Credentials), enable the Generative Language API, download "
            f"the JSON, and save it as {CLIENT_SECRET_FILENAME} under your Hermes home.",
            code="missing_client_secret",
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GeminiOAuthError(
            f"Could not read OAuth client file at {path}: {type(exc).__name__}",
            code="invalid_client_secret",
        ) from exc
    if not isinstance(raw, dict):
        raise GeminiOAuthError("OAuth client file must be a JSON object.", code="invalid_client_secret")
    info = raw.get("installed") or raw.get("web") or raw
    if not isinstance(info, dict):
        raise GeminiOAuthError("OAuth client file has no installed/web client block.", code="invalid_client_secret")
    client_id = str(info.get("client_id") or "").strip()
    client_secret = str(info.get("client_secret") or "").strip()
    if not client_id or not client_secret:
        raise GeminiOAuthError(
            "OAuth client file is missing client_id or client_secret.",
            code="invalid_client_secret",
        )
    return client_id, client_secret


def _token_storage(*, hermes_home: Path | None = None):
    from tools.mcp_oauth import HermesTokenStorage

    return HermesTokenStorage(GEMINI_OAUTH_TOKEN_SLUG, hermes_home=hermes_home)


def _tokens_path(*, hermes_home: Path | None = None) -> Path:
    storage = _token_storage(hermes_home=hermes_home)
    return storage._tokens_path()


def _read_token_payload(*, hermes_home: Path | None = None) -> Optional[Dict[str, Any]]:
    from tools.mcp_oauth import _read_json

    data = _read_json(_tokens_path(hermes_home=hermes_home))
    return data if isinstance(data, dict) else None


def _write_token_payload(payload: Dict[str, Any], *, hermes_home: Path | None = None) -> None:
    """Persist tokens in HermesTokenStorage's on-disk shape (access/refresh/expires)."""
    from tools.mcp_oauth import _write_json

    data = dict(payload)
    expires_in = data.get("expires_in")
    if expires_in is not None and "expires_at" not in data:
        try:
            data["expires_at"] = time.time() + int(expires_in)
        except (TypeError, ValueError):
            pass
    data.setdefault("token_type", "Bearer")
    data["hermes_issuer"] = "https://accounts.google.com"
    _write_json(_tokens_path(hermes_home=hermes_home), data)


def _access_token_unexpired(data: Dict[str, Any], *, skew: float = REFRESH_SKEW_SECONDS) -> Optional[str]:
    token = data.get("access_token")
    if not isinstance(token, str) or not token.strip():
        return None
    expires_at = data.get("expires_at")
    if isinstance(expires_at, (int, float)):
        if expires_at <= time.time() + skew:
            return None
        return token.strip()
    expires_in = data.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in <= skew:
        return None
    # No expiry metadata: treat as usable (caller refreshed recently) rather than inventing TTL.
    if expires_at is None and expires_in is None:
        return token.strip()
    return token.strip()


def _pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _post_token_form(body: Dict[str, str]) -> Dict[str, Any]:
    encoded = urllib.parse.urlencode(body).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TOKEN_REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            exc.read()
        raise GeminiOAuthError(
            f"Token endpoint rejected the request (HTTP {exc.code}).",
            code="token_exchange_failed",
        ) from None
    except urllib.error.URLError as exc:
        raise GeminiOAuthError(
            f"Could not reach Google token endpoint: {type(exc).__name__}",
            code="token_endpoint_unreachable",
        ) from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise GeminiOAuthError("Token endpoint returned non-JSON.", code="token_exchange_failed") from exc
    if not isinstance(data, dict) or not data.get("access_token"):
        raise GeminiOAuthError("Token endpoint response missing access_token.", code="token_exchange_failed")
    return data


def refresh_gemini_oauth_tokens(*, hermes_home: Path | None = None) -> Dict[str, Any]:
    """Refresh using the stored refresh_token; persist and return the new payload."""
    data = _read_token_payload(hermes_home=hermes_home) or {}
    refresh = data.get("refresh_token")
    if not isinstance(refresh, str) or not refresh.strip():
        raise GeminiOAuthError(
            "Gemini OAuth token expired and no refresh_token is stored. "
            "Run `hermes auth add gemini --type oauth` again.",
            code="missing_refresh_token",
        )
    client_id, client_secret = load_oauth_client_credentials(hermes_home=hermes_home)
    refreshed = _post_token_form({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh.strip(),
        "grant_type": "refresh_token",
    })
    # Google may omit refresh_token on refresh; keep the prior one.
    if not refreshed.get("refresh_token"):
        refreshed["refresh_token"] = refresh.strip()
    _write_token_payload(refreshed, hermes_home=hermes_home)
    return refreshed


def resolve_gemini_oauth_access_token(*, hermes_home: Path | None = None, refresh: bool = True) -> str:
    """Return a usable access token from HermesTokenStorage, refreshing when needed.

    Fail-closed: missing or expired-without-refresh raises ``GeminiOAuthError``.
    """
    data = _read_token_payload(hermes_home=hermes_home)
    if not data:
        raise GeminiOAuthError(
            "Gemini OAuth is enabled but no token is stored. "
            "Run `hermes auth add gemini --type oauth` after placing google_client_secret.json "
            f"in {get_hermes_home()}.",
            code="missing_token",
        )
    token = _access_token_unexpired(data)
    if token:
        return token
    if not refresh:
        raise GeminiOAuthError(
            "Gemini OAuth access token is expired.",
            code="expired_token",
        )
    refreshed = refresh_gemini_oauth_tokens(hermes_home=hermes_home)
    token = _access_token_unexpired(refreshed, skew=0)
    if not token:
        raise GeminiOAuthError(
            "Gemini OAuth refresh did not yield a usable access token.",
            code="refresh_failed",
        )
    return token


def run_gemini_oauth_login(
    *,
    open_browser: bool = True,
    hermes_home: Path | None = None,
    timeout_seconds: float = CALLBACK_WAIT_SECONDS,
) -> Dict[str, Any]:
    """Interactive Desktop OAuth consent; stores tokens via HermesTokenStorage."""
    client_id, client_secret = load_oauth_client_credentials(hermes_home=hermes_home)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    redirect_uri = f"http://127.0.0.1:{DEFAULT_REDIRECT_PORT}{CALLBACK_PATH}"

    result: Dict[str, Any] = {"code": None, "error": None, "state": None}
    done = threading.Event()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            result["code"] = (qs.get("code") or [None])[0]
            result["error"] = (qs.get("error") or [None])[0]
            result["state"] = (qs.get("state") or [None])[0]
            body = b"<html><body><p>Hermes Gemini OAuth complete. You can close this tab.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, format, *args):  # noqa: A003
            return

    server = http.server.HTTPServer(("127.0.0.1", DEFAULT_REDIRECT_PORT), _Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True, name="gemini-oauth-callback")
    thread.start()

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GEMINI_OAUTH_SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"
    print("Open this URL to authorize Gemini API access with your Google account:")
    print(auth_url)
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(auth_url)

    if not done.wait(timeout=float(timeout_seconds)):
        server.server_close()
        raise GeminiOAuthError("Timed out waiting for OAuth consent callback.", code="consent_timeout")
    server.server_close()

    if result.get("error"):
        raise GeminiOAuthError(f"OAuth consent failed: {result['error']}", code="consent_denied")
    if result.get("state") != state:
        raise GeminiOAuthError("OAuth state mismatch.", code="state_mismatch")
    code = result.get("code")
    if not isinstance(code, str) or not code:
        raise GeminiOAuthError("OAuth callback missing authorization code.", code="missing_code")

    tokens = _post_token_form({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    _write_token_payload(tokens, hermes_home=hermes_home)
    # Persist client registration sidecar so refresh can find client_id without re-reading secrets shape.
    from tools.mcp_oauth import _write_json

    storage = _token_storage(hermes_home=hermes_home)
    _write_json(storage._client_info_path(), {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uris": [redirect_uri],
        "token_endpoint_auth_method": "client_secret_post",
    })
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "expires_in": tokens.get("expires_in"),
        "token_file": str(_tokens_path(hermes_home=hermes_home)),
    }
