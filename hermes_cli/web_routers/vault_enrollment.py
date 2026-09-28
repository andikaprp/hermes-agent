"""HTTPS Mini App + enrollment API for Telegram → local encrypted vault.

Security boundary (mirrors ``/api/cron/fire``): these paths are on the dashboard
auth allowlist because they carry their own proof — validated Telegram WebApp
``initData`` plus a single-use server challenge. Credential fields never appear
in Telegram Bot API updates, logs, or responses.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from agent.vault_telegram_enrollment import (
    DEFAULT_CHALLENGE_TTL_SECONDS,
    DEFAULT_INIT_DATA_MAX_AGE_SECONDS,
    EnrollmentError,
    assert_owner_allowlisted,
    clear_challenges_for_tests,  # noqa: F401 — re-export for tests
    consume_challenge,
    enrollment_prereqs,
    mint_challenge,
    notify_telegram_status,
    request_host_allowed,
    request_origin_allowed,
    save_login_from_enrollment,
    validate_telegram_init_data,
)
from agent.vault_store import LOGIN_IDENTIFIER_TYPES, scrub_secret_from_text

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Minimal Mini App page: credentials live only in page memory until submit;
# never calls Telegram.WebApp.sendData.
_MINI_APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>Hermes — Save login</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
  :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
  body { margin: 0; padding: 16px; background: var(--tg-theme-bg-color, #111); color: var(--tg-theme-text-color, #eee); }
  h1 { font-size: 1.15rem; margin: 0 0 8px; }
  p.hint { opacity: .8; font-size: .9rem; margin: 0 0 16px; }
  label { display: block; font-size: .8rem; margin: 12px 0 4px; opacity: .9; }
  input, select, button { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 8px; border: 1px solid #555; background: var(--tg-theme-secondary-bg-color, #222); color: inherit; font-size: 1rem; }
  button { margin-top: 16px; background: var(--tg-theme-button-color, #2481cc); color: var(--tg-theme-button-text-color, #fff); border: none; font-weight: 600; }
  button.secondary { background: transparent; border: 1px solid #666; color: inherit; font-weight: 500; }
  .err { color: #f66; font-size: .9rem; margin-top: 12px; white-space: pre-wrap; }
  .ok { color: #6c6; font-size: .9rem; margin-top: 12px; }
</style>
</head>
<body>
  <h1>Save website login</h1>
  <p class="hint">Password stays in this form until you save. It is sent only to Hermes over HTTPS — never through Telegram chat.</p>
  <form id="f" autocomplete="off">
    <label for="origin">Site / origin (https://…)</label>
    <input id="origin" name="origin" type="url" required placeholder="https://example.com" inputmode="url"/>
    <label for="label">Label</label>
    <input id="label" name="label" type="text" required maxlength="120" placeholder="Example login"/>
    <label for="identifier_type">Identifier type</label>
    <select id="identifier_type" name="identifier_type">
      <option value="email">email</option>
      <option value="username">username</option>
      <option value="phone">phone</option>
    </select>
    <label for="identifier">Identifier</label>
    <input id="identifier" name="identifier" type="text" required autocomplete="username"/>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" required autocomplete="current-password"/>
    <button type="submit" id="save">Save to Hermes vault</button>
    <button type="button" class="secondary" id="cancel">Cancel</button>
  </form>
  <div id="msg" class="err" hidden></div>
<script>
(function () {
  const tg = window.Telegram && window.Telegram.WebApp;
  if (tg) { tg.ready(); tg.expand(); }
  const form = document.getElementById("f");
  const msg = document.getElementById("msg");
  let challengeId = null;
  let spent = false;

  function clearSecrets() {
    ["origin","label","identifier","password"].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) el.value = "";
    });
    challengeId = null;
  }

  function show(text, ok) {
    msg.hidden = !text;
    msg.className = ok ? "ok" : "err";
    msg.textContent = text || "";
  }

  function initData() {
    return (tg && tg.initData) ? tg.initData : "";
  }

  async function begin() {
    const data = initData();
    if (!data) { show("Open this form from the Hermes Telegram bot (private chat)."); return; }
    const resp = await fetch("/api/vault/enroll/begin", {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Telegram-Init-Data": data},
      body: JSON.stringify({init_data: data}),
      credentials: "same-origin"
    });
    const body = await resp.json().catch(function () { return {}; });
    if (!resp.ok) { show(body.detail || "Could not start enrollment"); return; }
    challengeId = body.challenge_id;
  }

  begin().catch(function () { show("Could not start enrollment"); });

  document.getElementById("cancel").addEventListener("click", function () {
    clearSecrets();
    show("Cancelled. You can close this window.", true);
    if (tg && tg.close) tg.close();
  });

  form.addEventListener("submit", async function (ev) {
    ev.preventDefault();
    if (spent) return;
    const data = initData();
    if (!data || !challengeId) { show("Enrollment session missing — reopen from Telegram."); return; }
    const payload = {
      challenge_id: challengeId,
      init_data: data,
      origin: document.getElementById("origin").value,
      label: document.getElementById("label").value,
      identifier_type: document.getElementById("identifier_type").value,
      identifier: document.getElementById("identifier").value,
      password: document.getElementById("password").value
    };
    spent = true;
    try {
      const resp = await fetch("/api/vault/enroll", {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-Telegram-Init-Data": data},
        body: JSON.stringify(payload),
        credentials: "same-origin"
      });
      const body = await resp.json().catch(function () { return {}; });
      // Drop secrets from memory immediately regardless of outcome.
      clearSecrets();
      payload.password = "";
      if (!resp.ok) {
        spent = false;
        show(body.detail || "Save failed");
        return;
      }
      show("Saved. You can close this window.", true);
      if (tg && tg.close) setTimeout(function () { tg.close(); }, 800);
    } catch (e) {
      clearSecrets();
      spent = false;
      show("Network error — nothing was sent to Telegram chat.");
    }
  });

  // Credentials must never enter Bot API updates (no WebApp send-data API).
})();
</script>
</body>
</html>
"""


def _bot_token() -> str:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def _section_int(key: str, default: int) -> int:
    try:
        from hermes_cli.config import load_config

        section = (load_config().get("vault") or {}).get("telegram_enrollment") or {}
        val = section.get(key, default)
        return int(val) if val is not None else default
    except Exception:
        return default


def _tls_ok(request: Request) -> bool:
    """Require HTTPS (direct or via X-Forwarded-Proto)."""
    if request.url.scheme == "https":
        return True
    xf = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return xf == "https"


def _guard_request(request: Request, prereqs) -> Optional[JSONResponse]:
    if not prereqs.ready:
        return JSONResponse({"detail": "telegram vault enrollment is not enabled"}, status_code=404)
    if not _tls_ok(request):
        return JSONResponse({"detail": "HTTPS required"}, status_code=403)
    host = request.headers.get("host") or ""
    if not request_host_allowed(host, prereqs.public_origin):
        return JSONResponse({"detail": "Host mismatch"}, status_code=403)
    origin = request.headers.get("origin") or ""
    # Same-origin navigations may omit Origin; when present it must match.
    if origin and not request_origin_allowed(origin, prereqs.public_origin):
        return JSONResponse({"detail": "Origin mismatch"}, status_code=403)
    return None


def _init_data_from(request: Request, body: Dict[str, Any]) -> str:
    header = (request.headers.get("x-telegram-init-data") or "").strip()
    if header:
        return header
    return str(body.get("init_data") or "").strip()


@router.get("/vault/enroll", response_class=HTMLResponse)
async def vault_enroll_page(request: Request) -> Response:
    """Serve the Mini App form. Enabled gate only — secrets never appear here."""
    prereqs = enrollment_prereqs()
    if not prereqs.ready:
        return HTMLResponse(
            "<!DOCTYPE html><html><body><p>Telegram vault enrollment is not configured.</p></body></html>",
            status_code=404)
    if not _tls_ok(request):
        return HTMLResponse(
            "<!DOCTYPE html><html><body><p>HTTPS required.</p></body></html>", status_code=403)
    host = request.headers.get("host") or ""
    if not request_host_allowed(host, prereqs.public_origin):
        return HTMLResponse(
            "<!DOCTYPE html><html><body><p>Host mismatch.</p></body></html>", status_code=403)
    # Assert page source never grows a WebApp send-data call (contract for reviewers/tests).
    assert "sendData" not in _MINI_APP_HTML
    return HTMLResponse(_MINI_APP_HTML, headers={"Cache-Control": "no-store"})


@router.post("/api/vault/enroll/begin")
async def vault_enroll_begin(request: Request) -> JSONResponse:
    """Mint a single-use challenge after validating Telegram initData (no secrets)."""
    prereqs = enrollment_prereqs()
    denied = _guard_request(request, prereqs)
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    init_data = _init_data_from(request, body)
    token = _bot_token()
    try:
        parsed = validate_telegram_init_data(
            init_data, token,
            max_age_seconds=_section_int("init_data_max_age_seconds", DEFAULT_INIT_DATA_MAX_AGE_SECONDS),
        )
        assert_owner_allowlisted(parsed["user_id"], prereqs.allowlist)
        # Private chats: chat_id == user_id for DMs.
        chat_id = parsed["user_id"]
        challenge_id = mint_challenge(
            telegram_user_id=parsed["user_id"],
            chat_id=chat_id,
            ttl_seconds=_section_int("challenge_ttl_seconds", DEFAULT_CHALLENGE_TTL_SECONDS),
        )
    except EnrollmentError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=401)
    return JSONResponse({
        "challenge_id": challenge_id,
        "expires_in": _section_int("challenge_ttl_seconds", DEFAULT_CHALLENGE_TTL_SECONDS),
    })


@router.post("/api/vault/enroll")
async def vault_enroll_submit(request: Request) -> JSONResponse:
    """Accept login fields over HTTPS; write via VaultStore; notify chat with non-secret status."""
    prereqs = enrollment_prereqs()
    denied = _guard_request(request, prereqs)
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "invalid JSON body"}, status_code=400)

    password = str(body.get("password") or "")
    identifier = str(body.get("identifier") or "")
    origin = str(body.get("origin") or "")
    label = str(body.get("label") or "")
    identifier_type = str(body.get("identifier_type") or "")
    challenge_id = str(body.get("challenge_id") or "")
    init_data = _init_data_from(request, body)
    # Drop password from the body dict ASAP so later exception paths cannot echo it.
    body["password"] = ""

    token = _bot_token()
    secret_for_scrub = {"password": password, "identifier": identifier}
    try:
        if identifier_type not in LOGIN_IDENTIFIER_TYPES:
            raise EnrollmentError(f"identifier_type must be one of {LOGIN_IDENTIFIER_TYPES}")
        if not password:
            raise EnrollmentError("password is required")
        parsed = validate_telegram_init_data(
            init_data, token,
            max_age_seconds=_section_int("init_data_max_age_seconds", DEFAULT_INIT_DATA_MAX_AGE_SECONDS),
        )
        assert_owner_allowlisted(parsed["user_id"], prereqs.allowlist)
        challenge = consume_challenge(challenge_id, telegram_user_id=parsed["user_id"])
        meta = save_login_from_enrollment(
            origin=origin,
            label=label,
            identifier=identifier,
            identifier_type=identifier_type,
            password=password,
        )
        notify_telegram_status(
            bot_token=token, chat_id=str(challenge.get("chat_id") or parsed["user_id"]),
            ok=True, label=meta["label"])
        return JSONResponse({"ok": True, "label": meta["label"], "origin": meta["origin"]})
    except EnrollmentError as exc:
        return JSONResponse({"detail": scrub_secret_from_text(str(exc), secret_for_scrub)}, status_code=400)
    except Exception as exc:
        _log.info("vault telegram enrollment submit failed (scrubbed)")
        return JSONResponse(
            {"detail": scrub_secret_from_text(str(exc), secret_for_scrub) or "save failed"},
            status_code=500)
    finally:
        password = ""
        secret_for_scrub.clear()
