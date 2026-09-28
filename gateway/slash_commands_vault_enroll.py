"""Telegram DM ``/vault-add`` — open Mini App enrollment (no credentials in chat)."""

from __future__ import annotations

import logging
from typing import Any

from agent.vault_telegram_enrollment import enrollment_prereqs

logger = logging.getLogger("gateway.run")

_SETUP_HINT = (
    "Telegram vault enrollment is not ready.\n\n"
    "Required (all must be set; feature stays off otherwise):\n"
    "• vault.telegram_enrollment.enabled: true\n"
    "• vault.telegram_enrollment.public_origin: https://your-host\n"
    "• TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS in this profile's .env\n"
    "• Hermes dashboard reachable at that HTTPS origin\n"
    "• BotFather Mini App / domain pointed at {origin}/vault/enroll\n\n"
    "Passwords must never be typed in this chat."
)

_DM_ONLY = "Open a private chat with the bot and run /vault-add there. Groups and channels are refused."
_NOT_TELEGRAM = "/vault-add is only available on Telegram."
_NOT_OWNER = "Only allowlisted owners can enroll credentials from Telegram."


class GatewayVaultEnrollCommandsMixin:
    """Off-turn slash handler: private Telegram DM → Mini App WebApp button."""

    async def _handle_vault_add_command(self, event) -> str:
        src = event.source
        platform = str(getattr(src.platform, "value", src.platform)).lower()
        if platform != "telegram":
            return _NOT_TELEGRAM

        chat_type = getattr(src, "chat_type", None)
        if chat_type not in {"dm", "private"} or not getattr(src, "chat_id", None):
            return _DM_ONLY

        prereqs = enrollment_prereqs()
        if not prereqs.ready:
            return _SETUP_HINT.format(origin=prereqs.public_origin or "https://<public-origin>")

        user_id = str(getattr(src, "user_id", "") or "")
        if user_id not in prereqs.allowlist:
            return _NOT_OWNER

        adapter = self._delivery_adapter_for(src)
        send_prompt = getattr(adapter, "send_vault_enroll_prompt", None) if adapter else None
        if not callable(send_prompt):
            return "This Telegram adapter build cannot open Mini Apps. Update Hermes and retry."

        metadata = self._thread_metadata_for_source(src) if hasattr(self, "_thread_metadata_for_source") else None
        try:
            result = await send_prompt(str(src.chat_id), prereqs.enroll_url, metadata=metadata)
        except Exception:
            logger.warning("/vault-add Mini App prompt failed", exc_info=True)
            return "Could not open the enrollment form. Check gateway logs (no secrets are logged)."

        if result is not None and getattr(result, "success", True) is False:
            return "Could not open the enrollment form. Check that the Mini App URL is HTTPS and registered with BotFather."

        # Message already delivered with the WebApp button; empty ack avoids a duplicate text reply.
        return ""
