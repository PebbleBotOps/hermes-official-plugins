"""google_meet plugin — let the agent join a Meet call, transcribe it, follow up.

Headless Chromium (Playwright) joins the URL, enables live captions and scrapes them into a
transcript. Realtime mode adds agent speech (OpenAI Realtime + virtual audio device); remote
nodes run the bot on another machine. Explicit-by-design: only joins ``https://meet.google.com/``
URLs passed in — no calendar scanning, auto-dial or consent announcement.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

from . import process_manager as pm
from .cli import meet_command as _meet_command, register_cli as _register_meet_cli
from .tools import (
    MEET_CHAT_SCHEMA, MEET_JOIN_SCHEMA, MEET_LEAVE_SCHEMA, MEET_SAY_SCHEMA, MEET_STATUS_SCHEMA,
    MEET_TRANSCRIPT_SCHEMA, check_meet_requirements, handle_meet_chat, handle_meet_join, handle_meet_leave,
    handle_meet_say, handle_meet_status, handle_meet_transcript)

logger = logging.getLogger(__name__)


_TOOLS = (
    ("meet_join",       MEET_JOIN_SCHEMA,       handle_meet_join,       "📞"),
    ("meet_status",     MEET_STATUS_SCHEMA,     handle_meet_status,     "🟢"),
    ("meet_transcript", MEET_TRANSCRIPT_SCHEMA, handle_meet_transcript, "📝"),
    ("meet_leave",      MEET_LEAVE_SCHEMA,      handle_meet_leave,      "👋"),
    ("meet_say",        MEET_SAY_SCHEMA,        handle_meet_say,        "🗣️"),
    ("meet_chat",       MEET_CHAT_SCHEMA,       handle_meet_chat,       "💬"))


def _owns_meeting(owner: str, session_id: str) -> bool:
    """Whether the session ending now is the conversation that joined. Compression rotates the
    session id mid-conversation, so the id recorded at join may be another link of the same
    compression chain as the one finalizing. Unknown on either side (a CLI ``hermes meet join``,
    a record from before this field) keeps the old behaviour: stop. A lookup that fails raises, and
    the caller leaves the bot running (its ``duration`` and lobby limits still apply)."""
    if not owner or not session_id or owner == session_id:
        return True
    from persistence_assembly import open_persistence_reader, profile_scope
    with open_persistence_reader(profile_scope().profile_dir) as store:
        return owner in store.lineage.lineage(session_id)


def _on_session_finalize(session_id: str = "", **kwargs) -> None:
    """Leave a still-running call when the session that joined it really ends, so we don't orphan a
    headless Chromium (never raises). Not ``on_session_end``: that fires after every turn, and the bot
    must outlive the turn that started it."""
    try:
        status = pm.status()
        if status.get("ok") and status.get("alive") and _owns_meeting(status.get("sessionId"), session_id):
            pm.stop(reason="session ended")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("google_meet on_session_finalize cleanup failed: %s", e)


def _core_ships_google_meet() -> bool:
    """True on a Hermes core that still bundles this plugin as ``plugins/google_meet``.

    That copy registers the same tools, ``hermes meet`` command and session hook; registering here too would
    collide with it. Looks only inside the imported ``plugins`` package's own ``__path__``."""
    try:
        import plugins as core_plugins
    except ImportError:
        return False
    return any((Path(p) / "google_meet" / "plugin.yaml").is_file()
               for p in getattr(core_plugins, "__path__", None) or ())


def register(ctx) -> None:
    """Register tools, CLI, and lifecycle hooks (called once by the plugin loader)."""
    if _core_ships_google_meet():
        logger.info("google-meet: this Hermes core still ships Google Meet built in; "
                    "the plugin stays inactive until core drops it")
        return
    # Windows: no tested audio-routing path and flaky guest-join Chromium — refuse rather than half-work.
    system = platform.system().lower()
    if system not in {"linux", "darwin"}:
        logger.info("google_meet plugin: platform=%s not supported (linux/macos only)", system)
        return
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(name=name, toolset="google_meet", schema=schema, handler=handler,
                          check_fn=check_meet_requirements, emoji=emoji)
    ctx.register_cli_command(
        name="meet", help="Google Meet bot (join, transcribe, follow up)",
        setup_fn=_register_meet_cli, handler_fn=_meet_command,
        description=("Let the hermes agent join a Google Meet call and scrape live "
                     "captions into a transcript. See: hermes meet setup"))
    ctx.register_hook("on_session_finalize", _on_session_finalize)
