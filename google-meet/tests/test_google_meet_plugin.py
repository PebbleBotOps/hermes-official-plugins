"""Tests for the google_meet plugin.

Covers the safety-gated pieces that don't require Playwright:

  * URL regex — only ``https://meet.google.com/`` URLs pass
  * Meeting-id extraction from Meet URLs
  * Status / transcript writes round-trip through the file-backed state
  * Tool handlers return well-formed JSON under all branches
  * Process manager refuses unsafe URLs and clears stale state cleanly
  * ``_on_session_finalize`` hook is defensive (no-ops when no bot active) and only stops the bot
    for the conversation that joined

Does NOT spawn a real Chromium — we mock ``subprocess.Popen`` where needed.
"""

from __future__ import annotations

import json
import os
import re
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# URL safety gate
# ---------------------------------------------------------------------------

def test_is_safe_meet_url_accepts_standard_meet_codes():
    from google_meet_plugin.meet_bot import _is_safe_meet_url

    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij")
    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij?pli=1")
    assert _is_safe_meet_url("https://meet.google.com/new")
    assert _is_safe_meet_url("https://meet.google.com/lookup/ABC123")


def test_meeting_id_extraction():
    from google_meet_plugin.meet_bot import _meeting_id_from_url

    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij") == "abc-defg-hij"
    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij?pli=1") == "abc-defg-hij"
    # fallback for codes we can't parse (e.g. /new before redirect)
    fallback = _meeting_id_from_url("https://meet.google.com/new")
    assert fallback.startswith("meet-")


# ---------------------------------------------------------------------------
# _BotState — transcript + status file round-trip
# ---------------------------------------------------------------------------

def test_bot_state_dedupes_captions_and_flushes_status(tmp_path):
    from google_meet_plugin.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alice", "Hey everyone")
    state.record_caption("Alice", "Hey everyone")  # dup — ignored
    state.record_caption("Bob", "Let's start")

    transcript = (out / "transcript.txt").read_text()
    assert "Alice: Hey everyone" in transcript
    assert "Bob: Let's start" in transcript
    # dedup — Alice line appears exactly once
    assert transcript.count("Alice: Hey everyone") == 1

    status = json.loads((out / "status.json").read_text())
    assert status["meetingId"] == "abc-defg-hij"
    assert status["transcriptLines"] == 2
    assert status["transcriptPath"].endswith("transcript.txt")


def test_parse_duration():
    from google_meet_plugin.meet_bot import _parse_duration

    assert _parse_duration("30m") == 30 * 60
    assert _parse_duration("2h") == 2 * 3600
    assert _parse_duration("90s") == 90
    assert _parse_duration("90") == 90
    assert _parse_duration("") is None
    assert _parse_duration("bogus") is None


# ---------------------------------------------------------------------------
# process_manager — refuses unsafe URLs, manages active pointer
# ---------------------------------------------------------------------------

def test_start_refuses_unsafe_url():
    from google_meet_plugin import process_manager as pm

    res = pm.start("https://evil.example.com/abc-defg-hij")
    assert res["ok"] is False
    assert "refusing" in res["error"]


def test_status_reports_no_active_meeting():
    from google_meet_plugin import process_manager as pm

    assert pm.status()["ok"] is False
    assert pm.transcript()["ok"] is False
    assert pm.stop()["ok"] is False


def test_transcript_reads_last_n_lines(tmp_path):
    from google_meet_plugin import process_manager as pm

    meeting_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "transcript.txt").write_text(
        "[10:00:00] Alice: one\n"
        "[10:00:01] Bob: two\n"
        "[10:00:02] Alice: three\n"
    )
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(meeting_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    res = pm.transcript(last=2)
    assert res["ok"] is True
    assert res["total"] == 3
    assert len(res["lines"]) == 2
    assert res["lines"][-1].endswith("Alice: three")


def test_stop_signals_process_and_clears_pointer(tmp_path):
    from google_meet_plugin import process_manager as pm

    pm._write_active({
        "pid": 11111, "meeting_id": "x-y-z",
        "out_dir": str(tmp_path / "x-y-z"),
        "url": "https://meet.google.com/x-y-z",
        "started_at": 0,
    })

    alive_seq = iter([True, True, False])  # alive at first, gone after SIGTERM
    def _alive(pid):
        try:
            return next(alive_seq)
        except StopIteration:
            return False

    sent = []
    def _kill(pid, sig):
        sent.append((pid, sig))

    # Record without a start fingerprint: identity falls back to the bot's exact argv.
    with patch.object(pm, "_pid_alive", side_effect=_alive), \
         patch.object(pm.os, "kill", side_effect=_kill), \
         patch.object(pm.time, "sleep", lambda _s: None), \
         patch("psutil.Process") as process:
        process.return_value.cmdline.return_value = pm._bot_argv()
        res = pm.stop()

    assert res["ok"] is True
    assert (11111, signal.SIGTERM) in sent
    # .active.json cleared
    assert pm._read_active() is None


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("identity", ["legacy-bootstrap", "legacy-core-module", "legacy-data",
                                      "matching-start", "reused-pid"])
def test_stop_checks_real_process_identity(tmp_path, monkeypatch, identity):
    """A pid the OS gave to another process after the bot exited must never be signalled."""
    import subprocess
    import sys

    from gateway.status import get_process_start_time
    from google_meet_plugin import process_manager as pm

    sleeper = "import time; time.sleep(30)"
    monkeypatch.setattr(pm, "_BOT_BOOTSTRAP", sleeper)  # the bot's exact argv, without joining a call
    package = tmp_path / "plugins" / "google_meet"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "meet_bot.py").write_text(sleeper)
    argv = {"legacy-core-module": [sys.executable, "-m", "plugins.google_meet.meet_bot"],
            "legacy-data": [sys.executable, "-c", sleeper, "meet_bot"]}.get(identity, pm._bot_argv())
    proc = subprocess.Popen(argv, cwd=tmp_path, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        record = {"pid": proc.pid, "meeting_id": "example"}
        if identity in {"matching-start", "reused-pid"}:
            fingerprint = get_process_start_time(proc.pid)
            assert fingerprint is not None
            record["pid_start_time"] = fingerprint + (10000 if identity == "reused-pid" else 0)
        pm._write_active(record)
        assert pm.status()["alive"] is (identity not in {"legacy-data", "reused-pid"})
        assert pm.stop()["ok"]
        if identity in {"legacy-data", "reused-pid"}:
            assert proc.poll() is None, "a different process must not receive a signal"
        else:
            proc.wait(timeout=5)
        assert pm._read_active() is None
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


def test_stop_revalidates_identity_before_escalating(monkeypatch):
    from google_meet_plugin import process_manager as pm

    pm._write_active({"pid": 33333, "pid_start_time": 4200, "meeting_id": "example"})
    starts = iter([4200, 99999])
    sent = []
    monkeypatch.setattr(pm, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pm, "_kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(pm.time, "sleep", lambda delay: None)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: next(starts))
    assert pm.stop()["ok"]
    assert sent == [signal.SIGTERM]
    assert pm._read_active() is None


def test_start_records_the_bots_start_fingerprint(monkeypatch):
    from google_meet_plugin import process_manager as pm

    class _FakeProc:
        pid = 4245

    monkeypatch.setattr(pm.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: 777 if pid == 4245 else None)
    assert pm.start("https://meet.google.com/abc-defg-hij")["ok"] is True
    assert pm._read_active()["pid_start_time"] == 777


# ---------------------------------------------------------------------------
# Tool handlers — JSON shape + safety gates
# ---------------------------------------------------------------------------

def test_meet_join_handler_missing_url_returns_error():
    from google_meet_plugin.tools import handle_meet_join

    out = json.loads(handle_meet_join({}))
    assert out["success"] is False
    assert out["error"]


# ---------------------------------------------------------------------------
# _on_session_finalize — defensive cleanup, only for the conversation that joined
# ---------------------------------------------------------------------------

def test_on_session_finalize_noop_when_nothing_active():
    from google_meet_plugin import _on_session_finalize
    # Should not raise and should not call stop().
    with patch("google_meet_plugin.pm.stop") as stop_mock:
        _on_session_finalize(session_id="s1")
    stop_mock.assert_not_called()


def test_bot_outlives_turns_and_other_sessions(monkeypatch):
    """meet_join returns at once and the bot must keep running across turns; only the end of the
    session that joined the call may stop it (on_session_end fires after every turn)."""
    import importlib

    plugin = importlib.import_module("google_meet_plugin")
    from google_meet_plugin import process_manager as pm
    from google_meet_plugin.tools import handle_meet_join

    hooks = {}
    ctx = SimpleNamespace(register_tool=lambda **_k: None, register_cli_command=lambda **_k: None,
                          register_hook=lambda name, fn: hooks.setdefault(name, []).append(fn))
    monkeypatch.setattr(plugin, "_core_ships_google_meet", lambda: False)
    monkeypatch.setattr(plugin.platform, "system", lambda: "Linux")
    plugin.register(ctx)

    def _fake_start(**kw):
        pm._write_active({"pid": 4242, "meeting_id": "abc-defg-hij", "out_dir": "/nonexistent",
                          "url": kw["url"], "started_at": 0, "session_id": kw.get("session_id")})
        return {"ok": True}

    with patch.object(pm, "start", side_effect=_fake_start), \
            patch("google_meet_plugin.tools.check_meet_requirements", return_value=True):
        handle_meet_join({"url": "https://meet.google.com/abc-defg-hij"}, session_id="s1")
    assert pm.status()["sessionId"] == "s1"

    def _fire(name, **kw):
        for fn in hooks.get(name, []):
            fn(**kw)

    real_status = pm.status  # the bot is alive however liveness is checked; the hook decides on ownership
    with patch.object(pm, "status", lambda: {**real_status(), "alive": True}), \
            patch.object(pm, "stop") as stop_mock:
        _fire("on_session_end", session_id="s1", completed=True, interrupted=False)
        _fire("on_session_finalize", session_id="s2")
        stop_mock.assert_not_called()
        _fire("on_session_finalize", session_id="s1")
        stop_mock.assert_called_once()


def test_bot_follows_its_conversation_across_compression():
    """Compression rotates the session id mid-meeting: finalizing the continuation still ends the
    conversation that joined the call, while an unrelated session's end leaves the bot alone."""
    import google_meet_plugin as plugin
    from google_meet_plugin import process_manager as pm
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence, profile_scope

    store = open_persistence(profile_scope())  # the per-test HERMES_HOME's store
    try:
        store.sessions.create(NewSession(session_id="root", source="cli"))
        store.sessions.end("root", "compression")
        store.sessions.create(NewSession(session_id="tip", source="cli", parent_session_id="root"))
        store.sessions.create(NewSession(session_id="other", source="cli"))
    finally:
        store.close()
    pm._write_active({"pid": 4242, "meeting_id": "abc-defg-hij", "out_dir": "/nonexistent",
                      "url": "https://meet.google.com/abc-defg-hij", "started_at": 0, "session_id": "root"})

    real_status = pm.status
    with patch.object(pm, "status", lambda: {**real_status(), "alive": True}), \
            patch.object(pm, "stop") as stop_mock:
        plugin._on_session_finalize(session_id="other")
        stop_mock.assert_not_called()
        plugin._on_session_finalize(session_id="tip")
        stop_mock.assert_called_once()


@pytest.mark.platforms("posix")
def test_served_profile_bot_never_gets_the_launch_profiles_env(monkeypatch, tmp_path):
    """Profile B has no OpenAI key: its bot carries neither launch profile A's key nor A's home,
    and still gets the import path it needs to run this interpreter."""
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from google_meet_plugin import process_manager as pm
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / ".hermes"
    served = launch / "profiles" / "b"
    served.mkdir(parents=True)
    (launch / ".env").write_text("OPENAI_API_KEY=sk-launch-profile\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-launch-profile")  # launch .env loaded at boot
    monkeypatch.setenv("PYTHONPATH", "/opt/hermes-agent")  # launcher-composed import path
    captured = {}

    class _FakeProc:
        pid = 4244

    monkeypatch.setattr(pm.subprocess, "Popen", lambda cmd, **kw: (captured.update(kw["env"]), _FakeProc())[1])

    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(served))
    scope_token = set_secret_scope({}, profile_home=str(served))
    try:
        assert pm.start("https://meet.google.com/abc-defg-hij", mode="realtime")["ok"] is True
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)

    assert "OPENAI_API_KEY" not in captured
    assert "HERMES_MEET_REALTIME_KEY" not in captured
    assert captured["HERMES_HOME"] == str(served)
    assert "/opt/hermes-agent" in captured["PYTHONPATH"].split(os.pathsep)


# ---------------------------------------------------------------------------
# Plugin register() — platform gating + tool registration
# ---------------------------------------------------------------------------

class _RegisterCtx:
    def __init__(self):
        self.tools, self.cli_commands, self.hooks = [], [], []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs["name"])

    def register_cli_command(self, **kwargs):
        self.cli_commands.append(kwargs["name"])

    def register_hook(self, name, fn):
        self.hooks.append(name)


def test_register_on_a_core_without_the_bundled_copy(monkeypatch):
    import importlib

    plugin = importlib.import_module("google_meet_plugin")
    monkeypatch.setattr(plugin, "_core_ships_google_meet", lambda: False)
    monkeypatch.setattr(plugin.platform, "system", lambda: "Linux")
    ctx = _RegisterCtx()
    plugin.register(ctx)
    assert set(ctx.tools) == {"meet_join", "meet_leave", "meet_status", "meet_transcript", "meet_say", "meet_chat"}
    assert ctx.cli_commands == ["meet"] and ctx.hooks == ["on_session_finalize"]


def test_register_stays_inert_on_a_core_that_still_bundles_google_meet(monkeypatch, tmp_path):
    """The bundled copy registers the same tools, ``hermes meet`` and session hook."""
    import importlib

    import plugins as core_plugins

    plugin = importlib.import_module("google_meet_plugin")
    bundled = tmp_path / "core_plugins"
    (bundled / "google_meet").mkdir(parents=True)
    (bundled / "google_meet" / "plugin.yaml").write_text("name: google_meet\n")
    monkeypatch.setattr(core_plugins, "__path__", [str(bundled)])
    ctx = _RegisterCtx()
    plugin.register(ctx)
    assert ctx.tools == [] and ctx.cli_commands == [] and ctx.hooks == []



# ---------------------------------------------------------------------------
# v2: process_manager.enqueue_say + realtime-mode passthrough
# ---------------------------------------------------------------------------

def test_enqueue_say_requires_text():
    from google_meet_plugin import process_manager as pm
    assert pm.enqueue_say("")["ok"] is False
    assert pm.enqueue_say("   ")["ok"] is False


# ---------------------------------------------------------------------------
# v3: NodeClient routing from tool handlers
# ---------------------------------------------------------------------------


def test_cli_register_includes_node_subcommand():
    """`hermes meet` argparse tree includes the node subtree."""
    import argparse
    from google_meet_plugin.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    # Parse a known-good node invocation to prove the subtree is wired.
    ns = parser.parse_args(["node", "list"])
    assert ns.meet_command == "node"
    assert ns.node_cmd == "list"


# ---------------------------------------------------------------------------
# v2.1: new _BotState fields + status dict shape
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Admission detection + barge-in helper
# ---------------------------------------------------------------------------

def test_looks_like_human_speaker():
    from google_meet_plugin.meet_bot import _looks_like_human_speaker

    # Blank, "unknown", "you", and the bot's own name → not human (no barge-in)
    for s in ("", "   ", "Unknown", "unknown", "You", "you", "Hermes Agent", "hermes agent"):
        assert not _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} should NOT be human"
    # Real names → human (barge-in)
    for s in ("Alice", "Bob Lee", "@teknium"):
        assert _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} SHOULD be human"


def test_detect_admission_returns_false_on_error():
    from google_meet_plugin.meet_bot import _ADMISSION_PROBE_JS, _probe

    class _FakePage:
        def evaluate(self, _js): raise RuntimeError("boom")

    assert _probe(_FakePage(), _ADMISSION_PROBE_JS) is False


# ---------------------------------------------------------------------------
# Realtime join path: late Join button, muted mic, PCM pump fed after start-up (#80875)
# ---------------------------------------------------------------------------

class _Toggle:
    """Playwright-locator stand-in: visible when *present*, records clicks on *page*."""

    def __init__(self, page, present, label):
        self.page, self.present, self.label = page, present, label

    first = property(lambda self: self)
    def count(self): return 1 if self.present() else 0
    def is_visible(self): return self.present()
    def click(self, timeout=None): self.page.clicked.append(self.label)


def test_join_polls_for_late_button_and_admission_unmutes_mic(tmp_path):
    """Meet renders Join now / the mic toggle asynchronously; a one-shot click missed it (#80875).
    The mic check is driven through ``_drain_loop``'s admission branch — the production call site."""
    import time

    from google_meet_plugin.meet_bot import _ADMISSION_PROBE_JS, _BotConfig, _BotState, _drain_loop, _join

    class _Page:
        def __init__(self, ready_at, mic_muted, stop):
            self.ready_at, self.mic_muted, self.stop, self.clicked = ready_at, mic_muted, stop, []

        def locator(self, sel):
            muted = "Turn on microphone" in sel
            return _Toggle(self, lambda: self.mic_muted == muted, sel)

        def get_by_role(self, role, name=None, exact=False):
            return _Toggle(self, lambda: name == "Join now" and time.time() >= self.ready_at, name)

        def get_by_text(self, text):  # no waiting-room text: the host let the bot straight in
            return _Toggle(self, lambda: False, text)

        def evaluate(self, js):  # admitted immediately; the caption drain ends the loop after one pass
            if js is _ADMISSION_PROBE_JS:
                return True
            self.stop["stop"] = True
            return []

        def is_closed(self): return False

    def admitted(mic_muted):
        stop = {"stop": False}
        page = _Page(ready_at=0, mic_muted=mic_muted, stop=stop)
        state = _BotState(tmp_path / str(mic_muted), "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
        with patch("google_meet_plugin.meet_bot.time.sleep"):
            _drain_loop(page, _BotConfig(guest_name="Bot", duration_s=0, lobby_timeout=30), state,
                        {"session": None, "enabled": True}, stop)
        assert state.in_call is True
        return page, state

    stop = {"stop": False}
    state = _BotState(tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    page = _Page(ready_at=time.time() + 0.6, mic_muted=True, stop=stop)
    _join(page, _BotConfig(guest_name="Bot"), state, timeout=5.0)
    assert page.clicked == ["Join now"]

    page, state = admitted(mic_muted=True)
    assert state.mic_state == "unmuted_clicked"
    assert any("Turn on microphone" in c for c in page.clicked)
    assert json.loads(state.status_path.read_text(encoding="utf-8"))["micState"] == "unmuted_clicked"
    # Control: an already-live mic is reported, never toggled off.
    live, state = admitted(mic_muted=False)
    assert state.mic_state == "unmuted" and live.clicked == []


# ---------------------------------------------------------------------------
# Listen-only join: camera/mic off, in the call only after the waiting room, captions by a real click
# ---------------------------------------------------------------------------

class _MeetPage:
    """A Meet tab as the bot sees it: pre-join screen → waiting room → call, on a fake clock.

    Toggles are labelled by the action they offer, as Meet labels them ("Turn off camera" while it
    is on). The waiting room already has a "Leave call" button, so the admission probe is True
    there too (#118049 §1). Clicks flip the toggle and are recorded."""

    def __init__(self, *, admit_at, stop, end_at, captions_on=False):
        self.clock, self.admit_at, self.stop, self.end_at = 0.0, admit_at, stop, end_at
        self.devices = {"camera": True, "microphone": True}  # Chromium's fake device: both live
        self.asked, self.captions_on, self.clicked = False, captions_on, []

    def phase(self):
        if not self.asked:
            return "prejoin"
        return "call" if self.clock >= self.admit_at else "lobby"

    def _labels(self):
        labels = {"Ask to join": lambda: setattr(self, "asked", True)} if self.phase() == "prejoin" else {}
        for device, live in self.devices.items():
            labels[f"Turn {'off' if live else 'on'} {device}"] = (
                lambda d=device: self.devices.__setitem__(d, not self.devices[d]))
        if self.phase() == "call":
            labels[f"Turn {'off' if self.captions_on else 'on'} captions"] = (
                lambda: setattr(self, "captions_on", not self.captions_on))
        return labels

    def _match(self, name):
        for label, action in self._labels().items():
            if (name.search(label) if hasattr(name, "search") else label == name):
                return label, action
        return None

    def get_by_role(self, role, name=None, exact=False):
        page = self

        class _Btn(_Toggle):
            def click(self, timeout=None):
                found = page._match(name)
                page.clicked.append(found[0])
                found[1]()

        return _Btn(self, lambda: role == "button" and self._match(name) is not None, name)

    def get_by_text(self, text):
        waiting = "Asking to join… You'll join the call when someone lets you in"
        return _Toggle(self, lambda: self.phase() == "lobby" and bool(text.search(waiting)), text)

    def locator(self, sel):  # guest-name box (pre-join) and the realtime mic toggles
        if "name" in sel:
            box = _Toggle(self, lambda: self.phase() == "prejoin", sel)
            box.fill = lambda *_a, **_k: None
            return box
        return self.get_by_role("button", name=re.compile(re.escape(sel.split('*="')[1].split('"')[0]), re.IGNORECASE))

    def evaluate(self, js):
        from google_meet_plugin.meet_bot import _ADMISSION_PROBE_JS, _DENIED_PROBE_JS
        if js is _ADMISSION_PROBE_JS:
            return self.phase() in {"lobby", "call"}  # "Leave call" is there in both
        if js is _DENIED_PROBE_JS:
            return False
        return []  # caption drain: nobody spoke

    def is_closed(self): return False

    def sleep(self, seconds):
        self.clock += seconds
        if self.clock >= self.end_at:
            self.stop["stop"] = True


def _run_meeting(tmp_path, *, admit_at, end_at, realtime=False, captions_on=False):
    import time
    from types import SimpleNamespace

    from google_meet_plugin import meet_bot
    from google_meet_plugin.meet_bot import _BotConfig, _BotState, _drain_loop, _join

    stop = {"stop": False}
    page = _MeetPage(admit_at=admit_at, stop=stop, end_at=end_at, captions_on=captions_on)
    state = _BotState(tmp_path / "out", "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    clock = SimpleNamespace(time=lambda: page.clock, sleep=page.sleep, strftime=time.strftime,
                            localtime=time.localtime)
    with patch.object(meet_bot, "time", clock):
        _join(page, _BotConfig(guest_name="Bot"), state, timeout=5.0, keep_mic=realtime)
        _drain_loop(page, _BotConfig(guest_name="Bot", duration_s=0, lobby_timeout=300), state,
                    {"session": None, "enabled": realtime}, stop)
    return page, state


def test_transcribe_join_turns_camera_and_mic_off_before_asking(tmp_path):
    """Chromium's fake device would show the green test pattern and play its test tone."""
    page, state = _run_meeting(tmp_path, admit_at=float("inf"), end_at=1)
    assert page.clicked == ["Turn off camera", "Turn off microphone", "Ask to join"]
    assert page.devices == {"camera": False, "microphone": False} and state.lobby_waiting is True


def test_waiting_room_is_not_in_the_call(tmp_path):
    """The waiting room shares the "Leave call" button with the call: no inCall, no caption clicks."""
    _, state = _run_meeting(tmp_path, admit_at=float("inf"), end_at=60)
    assert state.in_call is False and state.lobby_waiting is True
    assert state.captions_enabled_attempted is False
    status = json.loads(state.status_path.read_text(encoding="utf-8"))
    assert status["inCall"] is False and status["lobbyWaiting"] is True


def test_admitted_bot_turns_captions_on_once_and_stays_muted(tmp_path):
    page, state = _run_meeting(tmp_path, admit_at=20, end_at=60)
    assert state.in_call is True and state.lobby_waiting is False and 20 <= state.joined_at < 25
    assert page.clicked.count("Turn on captions") == 1 and page.captions_on is True
    assert state.captions_on is True and state.captions_enabled_attempted is True
    # Listen-only: the mic is never unmuted in the call.
    assert page.devices["microphone"] is False and state.mic_state == "muted"
    assert json.loads(state.status_path.read_text(encoding="utf-8"))["captionsOn"] is True


def test_captions_already_on_are_never_toggled_off(tmp_path):
    page, state = _run_meeting(tmp_path, admit_at=10, end_at=60, captions_on=True)
    assert state.captions_on is True and page.captions_on is True
    assert not any("captions" in c for c in page.clicked)


def test_realtime_join_keeps_the_mic_for_speech(tmp_path):
    page, state = _run_meeting(tmp_path, admit_at=10, end_at=30, realtime=True)
    assert "Turn off microphone" not in page.clicked and page.devices == {"camera": False, "microphone": True}
    assert state.mic_state == "unmuted"


def test_pcm_pump_receives_audio_appended_after_start(tmp_path, monkeypatch):
    """The pump used to read the empty speaker.pcm to EOF and exit before Realtime spoke (#80875)."""
    import subprocess
    import time

    from google_meet_plugin import meet_bot

    pcm, sink = tmp_path / "speaker.pcm", tmp_path / "device.bin"
    pcm.write_bytes(b"")
    real_popen = subprocess.Popen

    def cat_popen(cmd, **kw):  # `cat` stands in for paplay: same stdin / file-EOF semantics
        assert cmd[0] == "paplay" and cmd[-1] == "-"
        kw["stdout"] = open(sink, "wb")
        return real_popen(["cat"], **kw)

    monkeypatch.setattr(meet_bot.subprocess, "Popen", cat_popen)
    rt, stop = {}, {"stop": False}
    state = meet_bot._BotState(tmp_path, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    meet_bot._start_pcm_pump(rt, {"platform": "linux", "write_target": "sink"}, pcm, state, stop)
    time.sleep(0.2)
    with open(pcm, "ab") as f:
        f.write(b"\x01\x02" * 2000)
    deadline = time.time() + 5
    while time.time() < deadline and sink.stat().st_size < 4000:
        time.sleep(0.05)
    assert rt["pcm_pump"].poll() is None
    assert sink.stat().st_size == 4000
    stop["stop"] = True
    meet_bot._teardown_realtime({**rt, "speaker_thread": None, "session": None, "bridge": None})
    assert not rt["pcm_tail_thread"].is_alive()
    assert state.mic_state is None and "micState" in state.status_path.read_text(encoding="utf-8")


def test_pcm_tail_loop_swallows_only_pipe_errors(tmp_path):
    """A closed pump pipe is expected and quiet; any other tail-thread bug must not be silenced."""
    from types import SimpleNamespace

    from google_meet_plugin.meet_bot import _pcm_tail_loop

    pcm = tmp_path / "speaker.pcm"
    pcm.write_bytes(b"\x00" * 16)

    def proc(exc):
        def write(_chunk): raise exc
        return SimpleNamespace(poll=lambda: None, stdin=SimpleNamespace(write=write, flush=lambda: None,
                                                                        close=lambda: None))

    _pcm_tail_loop(proc(BrokenPipeError()), pcm, {"stop": False})  # quiet
    with pytest.raises(RuntimeError):
        _pcm_tail_loop(proc(RuntimeError("bug")), pcm, {"stop": False})


# ---------------------------------------------------------------------------
# Realtime session counters + cancel_response (barge-in)
# ---------------------------------------------------------------------------

def test_realtime_session_cancel_response_when_disconnected():
    from google_meet_plugin.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    # No _ws yet — cancel should no-op and return False.
    assert sess.cancel_response() is False


# ---------------------------------------------------------------------------
# hermes meet install CLI
# ---------------------------------------------------------------------------




@pytest.mark.platforms("linux", "macos")
def test_cmd_install_checks_plugin_dependencies_and_ensures_pm_chromium(monkeypatch, capsys):
    """playwright/websockets come with the plugin install; ``hermes meet install`` only checks them."""
    import importlib.util

    import pm

    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: pytest.fail("the plugin's deps are not a core extra"))
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: calls.append((name, kwargs)))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name in {"playwright", "websockets"} else real_find_spec(name, *a))
    from google_meet_plugin import cli
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("direct installer subprocess"))
    assert cli._cmd_install(realtime=False, assume_yes=True) == 0
    assert calls == [("chromium", {"explicit": True})]
    assert "hermes meet setup" in capsys.readouterr().out


@pytest.mark.platforms("linux", "macos")
def test_cmd_install_reports_missing_plugin_dependencies(monkeypatch, capsys):
    import importlib.util

    import pm

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "playwright" else real_find_spec(name, *a))
    monkeypatch.setattr(pm, "ensure", lambda *a, **kw: pytest.fail("Chromium before the Python deps"))
    from google_meet_plugin import cli
    assert cli._cmd_install(realtime=False, assume_yes=True) == 1
    assert "missing: playwright" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Leaving when the call is over: ended, removed, or everyone else gone
# ---------------------------------------------------------------------------

_CALL_PAGE_TEXT = {"alone": "You\u2019re the only one here", "ended": "The call has ended",
                   "removed": "You were removed from the meeting"}


class _CallPage:
    """An admitted bot's Meet tab on a fake clock. *script* maps a start time to what Meet shows
    from then on: ``"others"`` (people in the call), ``"alone"`` ("You're the only one here"),
    ``"ended"`` (the left-the-call page) or ``"removed"``."""

    def __init__(self, script, stop, end_at):
        self.script, self.stop, self.end_at, self.clock = sorted(script.items()), stop, end_at, 0.0

    def shows(self):
        return [what for at, what in self.script if at <= self.clock][-1]

    def get_by_text(self, pattern):
        return _Toggle(self, lambda: bool(pattern.search(_CALL_PAGE_TEXT.get(self.shows(), ""))), pattern)

    def get_by_role(self, role, name=None, exact=False):
        home = self.shows() in {"ended", "removed"} and name == "Return to home screen"
        return _Toggle(self, lambda: home, name)

    def evaluate(self, _js): return []  # caption drain: nothing new
    def is_closed(self): return False

    def sleep(self, seconds):
        self.clock += seconds
        if self.clock >= self.end_at:
            self.stop["stop"] = True


def _sit_in_call(tmp_path, script, end_at=1800):
    """Run the drain loop for an already-admitted bot; return (leave reason, seconds in the loop)."""
    import time
    from types import SimpleNamespace

    from google_meet_plugin import meet_bot
    from google_meet_plugin.meet_bot import _BotConfig, _BotState, _drain_loop

    stop = {"stop": False}
    page = _CallPage(script, stop, end_at)
    state = _BotState(tmp_path / "out", "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    state.set(in_call=True, joined_at=0.0)
    clock = SimpleNamespace(time=lambda: page.clock, sleep=page.sleep, strftime=time.strftime,
                            localtime=time.localtime)
    with patch.object(meet_bot, "time", clock):
        _drain_loop(page, _BotConfig(guest_name="Bot", duration_s=0, lobby_timeout=300), state,
                    {"session": None}, stop)
    assert json.loads(state.status_path.read_text(encoding="utf-8"))["leaveReason"] == state.leave_reason
    return state.leave_reason, page.clock


def test_bot_leaves_when_the_call_ends(tmp_path):
    """The drain loop only knew duration, lobby, denial and page loss: an ended call kept the bot
    on the "call ended" page until ``duration`` ran out."""
    reason, at = _sit_in_call(tmp_path, {0: "others", 600: "ended"})
    assert reason == "call_ended" and 600 <= at < 610


def test_bot_removed_from_the_call_says_so(tmp_path):
    reason, at = _sit_in_call(tmp_path, {0: "others", 300: "removed"})
    assert reason == "removed" and 300 <= at < 310


def test_bot_leaves_after_everyone_else_has_gone(tmp_path):
    from google_meet_plugin.meet_bot import _ALONE_GRACE_S

    reason, at = _sit_in_call(tmp_path, {0: "others", 900: "alone"})
    assert reason == "alone" and 900 + _ALONE_GRACE_S <= at < 900 + _ALONE_GRACE_S + 10


def test_brief_drop_out_and_early_join_do_not_end_the_meeting(tmp_path):
    from google_meet_plugin.meet_bot import _ALONE_GRACE_S

    # Everyone drops for a minute and comes back: the bot stays, and the wait starts over next time.
    reason, at = _sit_in_call(tmp_path / "drop", {0: "others", 600: "alone", 660: "others", 1000: "alone"})
    assert reason == "alone" and 1000 + _ALONE_GRACE_S <= at < 1000 + _ALONE_GRACE_S + 10
    # A signed-in bot that joins an empty call waits for the others instead of leaving at once.
    reason, at = _sit_in_call(tmp_path / "early", {0: "alone", 900: "others", 1500: "ended"})
    assert reason == "call_ended" and at >= 1500
    # Nobody ever comes: no "alone" exit (only the duration backstop would end it).
    reason, _ = _sit_in_call(tmp_path / "empty", {0: "alone"}, end_at=1200)
    assert reason is None


# ---------------------------------------------------------------------------
# Meeting chat: meet_chat queues, the admitted bot posts
# ---------------------------------------------------------------------------

def _active_meeting(tmp_path):
    from google_meet_plugin import process_manager as pm

    out = tmp_path / "abc-defg-hij"
    out.mkdir()
    pm._write_active({"pid": 0, "meeting_id": "abc-defg-hij", "out_dir": str(out),
                      "url": "https://meet.google.com/abc-defg-hij", "started_at": 0})
    return out


def test_meet_chat_queues_a_message_for_the_bot(tmp_path):
    from google_meet_plugin import process_manager as pm
    from google_meet_plugin.tools import handle_meet_chat

    assert pm.enqueue_chat("hello")["ok"] is False  # no meeting
    out = _active_meeting(tmp_path)
    assert json.loads(handle_meet_chat({"text": "  "}))["success"] is False
    assert pm.enqueue_chat("x" * (pm.CHAT_MAX_CHARS + 1))["ok"] is False  # Meet would refuse it
    res = json.loads(handle_meet_chat({"text": "Hi, I'm taking notes from the captions."}))
    assert res["success"] is True and res["enqueued_id"]
    lines = (out / "chat_queue.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["text"] for ln in lines] == ["Hi, I'm taking notes from the captions."]


class _ChatPage:
    """The call's chat panel: the toolbar toggle opens/closes it; Enter in its box posts."""

    def __init__(self, *, panel_open=False, works=True):
        self.panel_open, self.works, self.posted, self.clicked, self.typed = panel_open, works, [], [], ""
        self.on_press = lambda: None

    def locator(self, sel):
        page = self

        class _Box(_Toggle):
            def fill(self, text, timeout=None):
                page.typed = text

            def press(self, key, timeout=None):
                page.on_press()
                if not page.works:
                    raise RuntimeError("Meet did not take the message")
                page.posted.append(page.typed)

        return _Box(self, lambda: "message" in sel and self.panel_open, sel)

    def get_by_role(self, role, name=None, exact=False):
        page = self

        class _Btn(_Toggle):
            def click(self, timeout=None):
                page.clicked.append(name)
                page.panel_open = not page.panel_open

        return _Btn(self, lambda: name == "Chat with everyone", name)


def _post(page, state, queue, at):
    import time
    from types import SimpleNamespace

    from google_meet_plugin import meet_bot

    clock = SimpleNamespace(time=lambda: at, sleep=lambda _s: None, strftime=time.strftime,
                            localtime=time.localtime)
    with patch.object(meet_bot, "time", clock):
        meet_bot._post_queued_chat(page, state, queue)


def test_bot_posts_queued_chat_opening_the_panel_only_when_closed(tmp_path):
    from google_meet_plugin import process_manager as pm
    from google_meet_plugin.meet_bot import _BotState

    out = _active_meeting(tmp_path)
    state = _BotState(out, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    pm.enqueue_chat("one")
    pm.enqueue_chat("two")
    page = _ChatPage(panel_open=False)
    _post(page, state, out / "chat_queue.jsonl", at=100.0)
    assert page.posted == ["one", "two"] and page.clicked == ["Chat with everyone"]  # opened once, left open
    assert state.last_chat_at == 100.0 and state.chat_error is None
    assert (out / "chat_queue.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads(state.status_path.read_text(encoding="utf-8"))["lastChatAt"] == 100.0


def test_failed_chat_is_retried_in_order_then_dropped(tmp_path):
    from google_meet_plugin import process_manager as pm
    from google_meet_plugin.meet_bot import _BotState

    out = _active_meeting(tmp_path)
    queue = out / "chat_queue.jsonl"
    state = _BotState(out, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
    first = pm.enqueue_chat("first")["enqueued_id"]
    pm.enqueue_chat("second")
    page = _ChatPage(panel_open=True, works=False)
    page.on_press = lambda: pm.enqueue_chat("third")  # meet_chat runs while the bot posts "first"
    _post(page, state, queue, at=0.0)        # try 1 of "first"; "second" waits behind it
    page.on_press = lambda: None
    assert [json.loads(ln)["text"] for ln in queue.read_text(encoding="utf-8").splitlines()] == [
        "first", "second", "third"]          # the retries go back ahead of what was queued meanwhile
    _post(page, state, queue, at=5.0)        # too early to retry: nothing tried
    _post(page, state, queue, at=11.0)       # try 2
    assert state.chat_error is None
    _post(page, state, queue, at=22.0)       # try 3: "first" is dropped, "second" gets its try 1
    assert first in state.chat_error and state.last_chat_at is None
    page.works = True
    _post(page, state, queue, at=33.0)
    assert page.posted == ["second", "third"] and state.last_chat_at == 33.0
    assert first in state.chat_error  # the dropped message stays reported


def test_chat_waits_for_admission(tmp_path):
    """Queued before the host lets the bot in: posted only once it is in the call."""
    from google_meet_plugin import process_manager as pm
    from google_meet_plugin.meet_bot import _BotConfig, _BotState, _drain_loop

    out = _active_meeting(tmp_path)
    pm.enqueue_chat("hello")
    for admitted in (False, True):
        stop = {"stop": False}
        page = _ChatPage(panel_open=True)
        page.evaluate = lambda js, stop=stop: stop.__setitem__("stop", True) or False  # one pass
        page.get_by_text = lambda _t, page=page: _Toggle(page, lambda: False, _t)
        page.is_closed = lambda: False
        state = _BotState(out, "abc-defg-hij", "https://meet.google.com/abc-defg-hij")
        state.set(in_call=admitted)
        with patch("google_meet_plugin.meet_bot.time.sleep"):
            _drain_loop(page, _BotConfig(guest_name="Bot", duration_s=0, lobby_timeout=300), state,
                        {"session": None}, stop)
        assert page.posted == (["hello"] if admitted else [])
