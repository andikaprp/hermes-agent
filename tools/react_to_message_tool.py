#!/usr/bin/env python3
"""Agent emoji reaction: the user's message gets a real emoji on the chat platform whose adapter
implements the reaction API (Telegram first), and the desktop app gets the counterpart to the
user's own tapback (same store, one-per-author, ``author="agent"``).

One tool, two surfaces. A chat surface reacts ON the platform — ``message_reactions`` is folded
into exactly the sessions whose live adapter can carry the call (``gateway/run_turn.py`` for a
gateway turn, ``desktop_ui`` for a GUI session). Everything else keeps the original behaviour:
persist through the session DB and emit ``message.reaction`` for live painting. Defaults to the
triggering message; ``messages_back`` steps back over earlier user turns.
"""

import contextlib
import json

from gateway.session_context import NON_MESSAGING_SESSION_SURFACES, get_session_env
from tools import desktop_ui
from tools.registry import no_cache_check_fn, registry, tool_error

# The named toolset (``toolsets.py``) the session-scoped surfaces fold in — the gateway resolver
# for a chat platform whose adapter implements the reaction API, ``desktop_ui`` for a GUI session.
MESSAGING_REACTIONS_TOOLSET = "message_reactions"


def _open_session_db():
    """Open the SessionDB for the profile owning this turn, or ``None``."""
    try:
        from hermes_state_registry import acquire
        return acquire()
    except Exception:
        return None


def _release_session_db(db) -> None:
    """Give the store back the way the other tools do; never fatal."""
    if db is None:
        return
    with contextlib.suppress(Exception):
        from hermes_state_registry import release_or_close
        release_or_close(db)


def _session_chat_platform() -> str:
    """The chat platform this turn arrived on, or ``""`` for a surface with no chat behind it
    (CLI, cron, api_server, the desktop app) — the case where only the session store applies."""
    platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    return "" if platform in NON_MESSAGING_SESSION_SURFACES else platform


def adapter_supports_reactions(adapter) -> bool:
    """Whether ``adapter`` implements the reaction API. Duck-typed on purpose: each platform adds
    ``add_reaction``/``remove_reaction`` in its own slice, so a missing method is a normal "no",
    never an error. The toolset resolver and this tool read the same probe, so the schema only
    ever promises what a call can deliver."""
    return adapter is not None and callable(getattr(adapter, "add_reaction", None))


def _live_reaction_adapter(platform_name: str):
    """``(runner, adapter)`` for the ACTIVE PROFILE's live adapter for ``platform_name`` when it
    implements the reaction API, else ``(None, None)`` — standalone/cron have no adapter at all."""
    if not platform_name:
        return None, None
    try:
        from gateway.config import Platform
        from tools.send_message_senders import _live_adapter
        runner, adapter = _live_adapter(Platform(platform_name))
    except Exception:
        return None, None
    return (runner, adapter) if adapter_supports_reactions(adapter) else (None, None)


def _target_row_id(db, session_key, message_row_id, messages_back):
    """Row id to react to: the explicit row, else the newest USER row stepped back by
    ``messages_back`` (row ids aren't visible to the model; "two messages ago" is how a person
    thinks). ``None`` when the session has no such row."""
    if db is None:
        return None
    if message_row_id is not None:
        return int(message_row_id)
    return db.latest_message_row_id(session_key, role="user", offset=max(0, int(messages_back or 0)))


def _row_platform_id(db, session_key, row_id):
    """The platform's own id for the message behind *row_id* (``None`` when it has none) — the id
    a reaction on the chat platform has to name."""
    if db is None or row_id is None:
        return None
    try:
        return db.message_platform_id(session_key, int(row_id))
    except Exception:
        return None


def _target_message_id(session_key, message_row_id, messages_back):
    """``(row_id, platform_message_id)`` for the message this reaction belongs on, or ``(None, None)``.

    With nothing named it is the message that triggered the turn — the session's reply anchor —
    which is also the newest user row; ``messages_back`` steps to earlier user messages instead.
    """
    anchor = (str(get_session_env("HERMES_SESSION_MESSAGE_ID", "") or "").strip()
              if message_row_id is None and not messages_back else "")
    db = _open_session_db()
    try:
        row_id = _target_row_id(db, session_key, message_row_id, messages_back)
        return row_id, (_row_platform_id(db, session_key, row_id) or anchor or None)
    finally:
        _release_session_db(db)


def _platform_refusal(result) -> str:
    """``""`` when the adapter's reaction call reported success, else what to tell the model."""
    if isinstance(result, dict):
        return str(result.get("error") or "")
    return "" if result else "the platform refused the reaction"


def _react_on_platform(session_key, platform_name, chat_id, runner, adapter,
                       emoji, message_row_id, messages_back) -> str:
    """The chat surface: the reaction the user actually sees on their bubble. An empty ``emoji``
    retracts ours. The store gets a best-effort mirror so other clients show it too."""
    row_id, message_id = _target_message_id(session_key, message_row_id, messages_back)
    if not message_id:
        return tool_error("That message has no id on the platform to react to.")
    # The verb is the one this call needs: a platform may implement attaching but not retracting.
    react = getattr(adapter, "add_reaction" if emoji else "remove_reaction", None)
    if not callable(react):
        return tool_error(f"Platform '{platform_name}' cannot "
                          f"{'add a reaction' if emoji else 'retract a reaction'}.")
    kwargs = {"chat_id": chat_id, "message_id": message_id, **({"emoji": emoji} if emoji else {})}
    try:
        from model_tools import _run_async
        from tools.send_message_tool import _dispatch_on_gateway_loop
        result = _run_async(_dispatch_on_gateway_loop(
            runner, lambda: react(**kwargs),
            "react_to_message: failed to schedule the reaction on the gateway loop"))
    except Exception as exc:
        return tool_error(f"Reaction failed: {exc}")
    refusal = _platform_refusal(result)
    if refusal:
        return tool_error(f"Reaction failed: {refusal}")
    if row_id is not None:
        db = _open_session_db()
        if db is not None:
            with contextlib.suppress(Exception):
                db.set_message_reaction(session_key, int(row_id), emoji or None, author="agent")
        _release_session_db(db)
    return json.dumps({"success": True, "platform": platform_name, "message_id": message_id,
                       "emoji": emoji}, ensure_ascii=False)


def _react_in_store(session_key, emoji, message_row_id, messages_back) -> str:
    """The desktop surface: same store as the user's tapback (``author="agent"``), painted live
    through the renderer bridge. A missing bridge (non-desktop) is not an error — the reaction is
    persisted."""
    db = _open_session_db()
    if db is None:
        return tool_error("Session storage is unavailable.")
    try:
        row_id, target_role = message_row_id, "user"
        if row_id is None:
            back = max(0, int(messages_back or 0))
            row_id = db.latest_message_row_id(session_key, role="user", offset=back)
            if row_id is None:
                return tool_error(f"No user message found {back} back." if back else "No user message to react to yet.")
        else:
            target_role = db.get_message_role(session_key, int(row_id)) or "user"
        try:
            reactions = db.set_message_reaction(session_key, int(row_id), emoji or None, author="agent")
        except Exception as exc:
            return tool_error(f"Failed to set the reaction: {exc}")
        if reactions is None:
            return tool_error(f"Message {row_id} is not part of this conversation.")
        # Paint it live; a missing bridge (non-desktop) is not an error — the reaction is
        # persisted. `role` lets the renderer match a live message without a durable row id.
        with contextlib.suppress(Exception):
            desktop_ui.emit("message.reaction", {"row_id": int(row_id), "reactions": reactions, "role": target_role})
        return json.dumps({"success": True, "row_id": int(row_id), "reactions": reactions}, ensure_ascii=False)
    finally:
        _release_session_db(db)


def react_to_message_tool(emoji: str, message_row_id=None, messages_back=None) -> str:
    """Attach (or with an empty ``emoji`` retract) the agent's reaction on this turn's surface."""
    emoji = (emoji or "").strip()
    session_key = get_session_env("HERMES_SESSION_KEY", "") or get_session_env("HERMES_SESSION_ID", "")
    if not session_key:
        return tool_error("No active session — reactions need a persisted conversation.")
    platform_name = _session_chat_platform()
    runner, adapter = _live_reaction_adapter(platform_name)
    chat_id = str(get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    if adapter is not None and chat_id:
        return _react_on_platform(session_key, platform_name, chat_id, runner, adapter,
                                  emoji, message_row_id, messages_back)
    return _react_in_store(session_key, emoji, message_row_id, messages_back)


@no_cache_check_fn
def check_react_requirements() -> bool:
    """Reachability for the surface THIS turn is on — a chat platform whose live adapter implements
    the reaction API, else the desktop Appearance toggle (``display.message_reactions``, an opt-in).
    The toolset is the surface gate (root AGENTS.md); this only answers whether the surface we are
    on can take the call. Uncached on purpose: the answer follows the session, and the process-wide
    TTL cache would let one surface's ``False`` withdraw the tool from another's turn."""
    platform_name = _session_chat_platform()
    if _live_reaction_adapter(platform_name)[1] is not None:
        return True
    return desktop_ui.user_enabled("message_reactions", default=False)


REACT_TO_MESSAGE_SCHEMA = {
    "name": "react_to_message",
    "description": (
        "Reactions are optional. Never react to every message by default. First decide whether "
        "the user is sending one message or a short burst: treat consecutive bubbles sent close "
        "together as one thought, and if reacting to a burst, react only to the final bubble. Add "
        "a reaction only when it fully communicates the response: 👀 when taking a new request or "
        "investigating; 👍 when the user confirms, approves, corrects, or steers existing work; ✅ "
        "when requested work is fully done; ❤️ for thanks, warmth, or appreciation; 😂 or 💀 for "
        "something genuinely funny; 😢/🫂/😩 for a rough moment depending on the emotion; ❗ for "
        "something important or impressive; ❓ only when the natural response is genuine confusion. "
        "Do not react when a text answer, correction, warning, or question is needed, when the "
        "reaction would only duplicate the text reply, when the message is casual conversation and "
        "a normal reply is better, or when multiple reactions would compete (choose the one that "
        "best matches). If unsure, skip the reaction and reply normally. Never narrate a reaction. "
        "Targets the user's latest message by default — including the final bubble of a burst; "
        "one reaction per message: a different emoji replaces yours, an empty string retracts it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "The emoji to react with (e.g. '❤️', '😂', '👍'). Pass an empty "
                    "string to remove your reaction."
                ),
            },
            "message_row_id": {
                "type": "integer",
                "description": (
                    "Optional. The specific message to react to. Omit to react to the "
                    "user's latest message, which is almost always what you want."
                ),
            },
            "messages_back": {
                "type": "integer",
                "description": (
                    "Optional. React to an EARLIER user message: 1 = the one before "
                    "the latest, 2 = two before, and so on. For when something lands "
                    "late — the joke you only got after answering."
                ),
            },
        },
        "required": ["emoji"],
    },
}


registry.register(
    name="react_to_message", toolset=MESSAGING_REACTIONS_TOOLSET, schema=REACT_TO_MESSAGE_SCHEMA,
    handler=lambda args, **kw: react_to_message_tool(
        emoji=args.get("emoji", ""), message_row_id=args.get("message_row_id"),
        messages_back=args.get("messages_back")),
    check_fn=check_react_requirements, emoji="💛",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'env_var_enabled': ('utils', 'env_var_enabled'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
