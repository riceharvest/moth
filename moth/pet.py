"""moth as a desktop pet:  python -m moth.pet      (needs PySide6, which moth itself doesn't: pip install PySide6)

A small moth flies around your desktop and says moth's lines in a speech bubble. It drifts on smoothed noise, is
mildly drawn to the mouse cursor (moths love light), now and then lands on a screen edge to rest, and dozes (wings
folded, "z") after a few idle minutes. A new line from moth makes it flutter. Click it to talk to moth, drag it to
move it, right-click for a menu (open the chat page, mute bubbles, sleep, harness display, skin, updates, quit).

One command: when no moth answers on its URL (GET /api/health, default http://127.0.0.1:4747), the pet starts
`python -m moth.serve` itself (same interpreter, CPU, this install's weights; moth/launch.py) and dozes with "waking
up…" while the model loads. That server is the pet's: quitting the pet (menu, Ctrl-C, SIGTERM/SIGHUP, closing the
window) stops it, and on Linux it also goes if the pet is killed. A moth that was already running is used and never
stopped. Something else on the port, or less than ~3 GB of free memory: nothing is started and the bubble says why.
The server's output goes to ~/.moth/serve.log (live/serve.log in the dev checkout).

It only reads moth and never posts agent turns, so moth never hears a turn twice. Lines come from GET /api/says (token-free, moth's own lines only); when ~/.moth/token is readable it
polls the token-guarded /api/log instead (same indexes) to add a tiny "claude · repo" caption to reactions.
What you type goes to POST /api/chat; the reply comes back through the same poll (the POST's own events are ignored,
or every line would show twice). History is skipped on start; a new log epoch (moth restarted or the chat was reset)
starts over. When moth is down the pet dozes with "zzz… (moth server asleep)" once and retries with backoff.

The window: transparent, frameless, always on top, never takes focus (only the little input box does, when you
click). The moth is real art, not drawn by code: "202206 Silkmoth adult female icon" by DBCLS (Wikimedia Commons,
CC BY 4.0; the original SVG and credits are in moth/assets/). It is a top view with the wings spread. The SVG is
rendered with QtSvg and cut along the body into a left wing, a right wing and the body. The wing frames are those
wings squashed toward the body and turned a little: a 5-frame wingbeat, "resting" (wings narrow and swept back) and
"dozing" (swept back further, plus drifting "z" glyphs). Excited is a faster wingbeat. In flight the moth leans a
little toward where it is going (top view, so it is never mirrored). The chat page's header icon and favicon
(moth/static/index.html) are the same art.

Platforms: X11, macOS, Windows. Wayland doesn't let a client place its own window, so on a Wayland session with
XWayland the pet asks Qt for xcb first (QT_QPA_PLATFORM=xcb;wayland, only when QT_QPA_PLATFORM is unset). Without
XWayland it falls back to native Wayland: it can't fly around, so it sits wherever the compositor puts it (drag
still works). Under XWayland the cursor position only updates while the pointer is over an X window, so the light
attraction fades out when the cursor seems not to move. Window edges aren't detected (no portable way): it lands on
the screen's edges (the available area, so on top of a panel/taskbar).

  python -m moth.pet [--size 64] [--no-wander] [--url http://127.0.0.1:4747] [--token-file ~/.moth/token]
                     [--skin ID|FOLDER|moth] [--no-server] [--model DIR] [--server-arg=--flag ...] [-- moth.serve flags]
      --no-server: never start a server, just wait for one. --model: the weights for a server the pet starts.
      Flags for that server: --server-arg=--react-every --server-arg=0, or after --:  -- --react-every 0
  python -m moth.pet --export-petdex DIR    the same moth as a Petdex / Codex pet: DIR/pet.json + spritesheet.png

Petdex (https://petdex.dev, github.com/crafter-station/petdex) pets are a pet.json plus one 1536x1872 spritesheet of
192x208 cells, 8 columns x 9 rows (idle 6 frames, running-right 8, running-left 8, waving 4, jumping 5, failed 8,
waiting 6, running 6, review 6). The export fills every row from the same wing frames (scaled, tilted and offset;
none drawn new) and writes the art's CC BY credit into pet.json's description and DIR/CREDITS.txt. Codex's TUI
loads custom pets from $CODEX_HOME/pets/<id>/ (codex-rs/tui/src/pets/
model.rs, which rejects any other sheet size); Hermes from <hermes home>/pets/<slug>/ (agent/pet/store.py); the
Petdex desktop app from ~/.petdex/pets/<slug>/. The export only writes DIR; copy it to one of those yourself.

Skins (--skin ID|FOLDER|moth, or `python -m moth config pet_skin ID`; right-click "skin"): any Petdex pet instead
of the moth. An id is looked up in ~/.petdex/pets/<id>/ (`npx petdex install`), then $CODEX_HOME/pets/<id>/ and
<hermes home>/pets/<id>/; a folder path is used as is. Its pet.json names the sheet (spritesheetPath, inside the
folder), which must be the Petdex 1536x1872 grid. Rows to states: flying = running-right / running-left (the other
one mirrored when a row is empty), resting = idle, dozing = waiting (else idle, slowed), excited = jumping in the air
or waving on the ground, waiting on a reply = review. The cell is scaled to the pet's size; the bubble sits above
the pet's visible top. A missing or bad pet (no pet.json, wrong sheet size, unreadable image, empty idle row) means
moth plus a bubble saying why. moth ships no third-party pets.

Tray icon (where the platform has one; skipped silently otherwise, e.g. some Wayland setups): the same menu as the
right-click one: open chat page, mute bubbles, sleep, show moth in harnesses (moth/config.py show_in_harness),
skin, check for updates, quit.

Updates: once a day (config update_check, default on; last_update_check in ~/.moth/config.json) and on "check for
updates", `python -m moth update --check` runs in a child process off the UI thread. A new version is said in a
bubble ("run `python -m moth update`", or `pip install -U moth-companion` for newer pip code); nothing is ever
installed by the pet. HF_HUB_OFFLINE=1 or MOTH_OFFLINE=1: no check.

Only stdlib is needed for the logic (Wander, BubbleQueue, Poller, skin lookup and the row mapping); PySide6 is
imported when a window or a sprite sheet is made, so the logic is testable without a display.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import quote

from moth import config  # stdlib only: ~/.moth/config.json (pet_skin, update_check, show_in_harness)
from moth import launch  # stdlib only: starts moth.serve when nothing answers
from moth.tail import NAMES, clean  # stdlib only, same harness names and escape-stripping as moth tail

DEFAULT_URL = os.environ.get("MOTH_URL", "http://127.0.0.1:4747")
TOKEN_HEADER = "X-Moth-Token"
POLL = 1.5               # s between polls while moth answers
MAX_BACKOFF = 30.0       # s between retries at most while moth is down
SKIP_HISTORY = "1000000000"   # first poll: "since" past the end = no events, just the current end of the log
WRAP = 32                # bubble line width (characters)
PAGE_LINES = 5           # a longer line is shown as several bubbles, one after another
MAX_PAGES = 3            # ... at most this many per line (the last one ends in "…")
MAX_QUEUED = 4           # bubbles waiting; a burst drops the oldest
DOZE_AFTER = 180.0       # s without a new line or a click before the moth dozes
ASLEEP = "zzz… (moth server asleep)"
WAKING = "waking up…"    # while a server the pet started loads the model
DOWNLOADING = "downloading moth…"   # a pip install's first run: fetching the weights before the server can start
NOTICE_TTL = 30.0        # s a "can't wake moth" message stays up
CHAT_TIMEOUT = 90.0      # POST /api/chat waits for moth's whole turn (seconds on CPU)

# the page's palette (moth/static/index.html :root)
LAMP = "#e8b765"
NIGHT, NIGHT2, PAPER, INK, DUST, LINE = "#120f0c", "#1a1611", "#f1e6d0", "#2a2118", "#8c7d68", "#2c251d"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # moth is local: never via a proxy


# --- talking to moth (stdlib) ---------------------------------------------------------------------------------
def token_path(path: str | None = None) -> str:
    return os.path.expanduser(path or os.environ.get("MOTH_TOKEN_FILE") or os.path.join("~", ".moth", "token"))


def read_token(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def http_json(url: str, token: str | None = None, body: dict | None = None, timeout: float = 3.0):
    headers, data = {}, None
    if token:
        headers[TOKEN_HEADER] = token
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read(4 << 20))


def caption_for(card: dict) -> str:
    """'claude · moth-repo' for an agent card of the log (what moth reacted to)."""
    src = card.get("source")
    who = NAMES.get(src, src) if isinstance(src, str) and src else "agent"
    label = card.get("label")
    if isinstance(label, str) and label.strip():
        who += f" · {label.strip()}"
    who = " ".join(clean(who).split())
    return who if len(who) <= WRAP else who[:WRAP - 1] + "…"


class Poller:
    """moth's new lines, one poll at a time (the GUI runs it on a thread). poll() never raises.

    Returns (lines, status): lines = [(text, caption or None)], status = "up" / "down" when that changed, else None.
    The first answer only sets the cursor (history is skipped). With a readable token it polls /api/log (moth's lines
    plus the agent cards, for captions), else /api/says; a token the server refuses is not tried again until the
    file changes."""

    def __init__(self, url: str = DEFAULT_URL, token_file: str | None = None):
        self.url = url.rstrip("/")
        self.token_file = token_path(token_file)
        self.cursor: int | None = None
        self.epoch: str | None = None
        self.up: bool | None = None       # None until the first poll answers or fails
        self.fails = 0
        self.card: str | None = None      # caption of the last agent card moth hasn't answered yet
        self.bad_token: str | None = None

    def interval(self) -> float:
        return POLL if not self.fails else min(MAX_BACKOFF, POLL * 2 ** self.fails)

    def _fetch(self) -> dict:
        tok = read_token(self.token_file)
        use_log = bool(tok) and tok != self.bad_token
        since = SKIP_HISTORY if self.cursor is None else str(self.cursor)
        q = f"since={since}" + (f"&epoch={quote(self.epoch)}" if self.cursor is not None and self.epoch else "")
        if use_log:
            try:
                return http_json(f"{self.url}/api/log?{q}", tok)
            except urllib.error.HTTPError as e:
                if e.code != 401:
                    raise
                self.bad_token = tok   # wrong token: moth's lines without captions
        return http_json(f"{self.url}/api/says?{q}")

    def poll(self) -> tuple[list[tuple[str, str | None]], str | None]:
        try:
            d = self._fetch()
            if not (isinstance(d, dict) and isinstance(d.get("events"), list) and isinstance(d.get("next"), int)
                    and isinstance(d.get("epoch"), str)):
                raise ValueError("not moth")
        except Exception:  # noqa: BLE001  down, slow, not moth, garbage: all "asleep"
            self.fails += 1
            changed = self.up is not False
            self.up = False
            return [], "down" if changed else None
        status = None if self.up else "up"
        self.up, self.fails = True, 0
        first = self.cursor is None
        new_epoch = self.epoch is not None and d["epoch"] != self.epoch
        self.cursor, self.epoch = d["next"], d["epoch"]
        if first:
            return [], status
        if new_epoch:
            self.card = None
        lines = self.lines(d["events"])
        return (lines[-2:] if new_epoch else lines), status

    def lines(self, events: list) -> list[tuple[str, str | None]]:
        out = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if "agent" in ev:
                if not ev.get("skipped"):
                    self.card = caption_for(ev)
            elif "you" in ev:
                self.card = None   # what follows is moth answering you
            elif isinstance(ev.get("say"), str) and ev["say"].strip():
                out.append((ev["say"], self.card))
                self.card = None
        return out


def send_chat(url: str, token_file: str, text: str) -> str | None:
    """POST what the user typed; None when sent, else a short reason (shown in the bubble). The reply's events are
    ignored on purpose: the poll shows them."""
    tok = read_token(token_file)
    if not tok:
        return f"i can't find my token ({token_file}), so i can't hear you"
    try:
        http_json(f"{url.rstrip('/')}/api/chat", tok, {"text": text.strip()[:500]}, timeout=CHAT_TIMEOUT)
    except urllib.error.HTTPError as e:
        return "my token doesn't match moth's (~/.moth/token)" if e.code == 401 else f"moth said no ({e.code})"
    except Exception:  # noqa: BLE001
        return "i couldn't reach moth"
    return None


# --- speech bubbles (stdlib) ----------------------------------------------------------------------------------
def wrap_text(text: str, width: int = WRAP) -> list[str]:
    """moth's lowercase, except inside `backticks` (commands and names to copy: `pip install -U ...`)."""
    flat = " ".join(clean(text).split())
    flat = re.sub(r"`[^`]*`|[^`]+", lambda m: m.group(0) if m.group(0).startswith("`") else m.group(0).lower(), flat)
    return textwrap.wrap(flat, width, break_on_hyphens=False) or ["..."]


def bubble_ttl(lines: list[str]) -> float:
    """6 s for a word, 10 s for a full bubble."""
    n = sum(len(x) for x in lines)
    return round(min(10.0, max(6.0, 5.0 + n / 30)), 2)


@dataclass
class Bubble:
    lines: list[str]
    caption: str | None
    ttl: float
    shown: float | None = None


class BubbleQueue:
    """One bubble at a time; the next one waits until the current one has faded. A long line becomes up to
    MAX_PAGES bubbles. At most MAX_QUEUED wait (a burst drops the oldest)."""

    FADE_IN, FADE_OUT = 0.25, 0.8

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.waiting: list[Bubble] = []
        self.now_showing: Bubble | None = None

    def push(self, text: str, caption: str | None = None, ttl: float | None = None):
        lines = wrap_text(text)
        pages = [lines[i:i + PAGE_LINES] for i in range(0, len(lines), PAGE_LINES)]
        if len(pages) > MAX_PAGES:
            pages = pages[:MAX_PAGES]
            last = pages[-1][-1]
            pages[-1][-1] = (last if len(last) < WRAP else last[:WRAP - 1].rstrip()) + "…"
        for i, page in enumerate(pages):
            self.waiting.append(Bubble(page, caption if i == 0 else None, ttl or bubble_ttl(page)))
        del self.waiting[:-MAX_QUEUED]

    def fade(self, text: str):
        """Let the bubble saying `text` go now (fades out if showing; dropped if waiting)."""
        lines = wrap_text(text)
        self.waiting = [b for b in self.waiting if b.lines != lines]
        b = self.now_showing
        if b is not None and b.lines == lines and b.shown is not None:
            b.ttl = min(b.ttl, self.clock() - b.shown + self.FADE_OUT)

    def clear(self):
        self.waiting.clear()
        self.now_showing = None

    def current(self) -> tuple[Bubble | None, float]:
        """(the bubble to draw, its opacity 0..1)."""
        now = self.clock()
        b = self.now_showing
        if b is not None and now - b.shown >= b.ttl:
            b = self.now_showing = None
        if b is None and self.waiting:
            b = self.now_showing = self.waiting.pop(0)
            b.shown = now
        if b is None:
            return None, 0.0
        age = now - b.shown
        alpha = min(1.0, age / self.FADE_IN, (b.ttl - age) / self.FADE_OUT)
        return b, max(0.0, alpha)


# --- flight (stdlib) ------------------------------------------------------------------------------------------
FLY, LAND, REST, DOZE, HELD = "fly", "land", "rest", "doze", "held"


class Noise:
    """Smooth 1-D noise in about [-1, 1]: three sines at unrelated frequencies with random phases."""

    def __init__(self, rng: random.Random, base: float):
        self.parts = [(base * m, rng.uniform(0, math.tau), w) for m, w in ((1.0, .55), (2.31, .3), (4.77, .15))]

    def __call__(self, t: float) -> float:
        return sum(w * math.sin(f * t + p) for f, p, w in self.parts)


class Wander:
    """Where the moth is and what it is doing. Positions are the moth's centre in screen pixels; bounds is the
    available area (x, y, w, h) of the screen it is on. step(dt, now, cursor) moves it on.

    fly: drifts on smoothed noise, steers away from the edges, is drawn to the cursor (orbits it rather than sitting
    on it) while the cursor moves. After 25-70 s it picks a spot on an edge (land) and rests there 10-40 s. No new
    line and no click for DOZE_AFTER s, or asleep (menu / moth down): lands and dozes. excite(): flutters for a
    moment (and takes off). wander=False: sits in the bottom-right corner and only flutters there."""

    SPEED = 70.0

    def __init__(self, bounds, size: int = 56, wander: bool = True, rng: random.Random | None = None,
                 now: float = 0.0):
        self.rng = rng or random.Random()
        self.size, self.wander = size, wander
        self.bounds = tuple(bounds)
        self.turn, self.pace = Noise(self.rng, 0.35), Noise(self.rng, 0.21)
        self.heading = self.rng.uniform(0, math.tau)
        self.vx = self.vy = 0.0
        self.state, self.since = REST, now
        self.spot: tuple[float, float] | None = None
        self.doze_on_landing = False
        self.rest_for = self.rng.uniform(4, 10)
        self.fly_for = 0.0
        self.excited_until = -1.0
        self.idle_since = now
        self.asleep = False          # menu "sleep" or moth down: dozes until woken
        self.cursor = None
        self.cursor_seen = -1e9
        self.attend: tuple[float, float] | None = None   # hover near this point (the input box is open)
        self.x, self.y = self.home()

    # -- places
    def home(self) -> tuple[float, float]:
        bx, by, bw, bh = self.bounds
        return bx + bw - self.size * 1.2, by + bh - self.size * 0.38

    def landing_spot(self) -> tuple[float, float]:
        bx, by, bw, bh = self.bounds
        s, r = self.size, self.rng.random()
        if r < 0.45:   # on top of the panel / bottom edge
            return self.rng.uniform(bx + s, bx + bw - s), by + bh - s * 0.38
        if r < 0.85:   # clinging to a side, head up (a moth on a wall seen from the front is the resting delta)
            x = bx + s * 0.44 if self.rng.random() < 0.5 else bx + bw - s * 0.44
            return x, self.rng.uniform(by + bh * 0.25, by + bh - s)
        return self.rng.uniform(bx + s, bx + bw - s), by + s * 0.42   # hanging from the top edge

    def nearest_floor(self) -> tuple[float, float]:
        bx, by, bw, bh = self.bounds
        return min(max(self.x, bx + self.size), bx + bw - self.size), by + bh - self.size * 0.38

    # -- events
    @property
    def excited(self) -> bool:
        return self._now < self.excited_until

    _now = 0.0

    def excite(self, now: float, seconds: float = 2.5):
        self._now = now
        self.idle_since = now
        self.excited_until = now + seconds
        if self.state in (REST, DOZE) and not self.asleep and self.wander:
            self.take_off(now)
        elif self.state == DOZE and not self.asleep:
            self.state, self.since = REST, now

    def poke(self, now: float):
        """A click / the user did something: wakes it (unless put to sleep from the menu)."""
        self.idle_since = now
        if self.state == DOZE and not self.asleep:
            self.state, self.since = REST, now
            self.rest_for = self.rng.uniform(2, 6)

    def sleep(self, now: float, on: bool = True):
        self.asleep = on
        self.idle_since = now
        if on and self.state in (FLY, LAND):
            self.land(now, self.nearest_floor() if self.wander else self.home(), doze=True)
        elif on and self.state == REST:
            self.state, self.since = DOZE, now
        elif not on and self.state == DOZE:
            self.state, self.since = REST, now
            self.rest_for = self.rng.uniform(1, 4)

    def grab(self, now: float):
        self.state, self.since = HELD, now
        self.idle_since = now

    def drop(self, now: float, x: float, y: float):
        self.x, self.y = x, y
        self.idle_since = now
        bx, by, bw, bh = self.bounds
        near = min(x - bx, bx + bw - x, y - by, by + bh - y) < self.size * 0.8
        if not self.wander or near or self.asleep:
            self.state, self.since = (DOZE if self.asleep else REST), now
            self.rest_for = self.rng.uniform(10, 30)
        else:
            self.take_off(now)

    def take_off(self, now: float):
        self.state, self.since = FLY, now
        self.fly_for = self.rng.uniform(25, 70)
        bx, by, bw, bh = self.bounds
        self.heading = math.atan2((by + bh / 2) - self.y, (bx + bw / 2) - self.x) + self.rng.uniform(-.8, .8)
        self.vx, self.vy = math.cos(self.heading) * 60, math.sin(self.heading) * 60 - 40

    def land(self, now: float, spot, doze: bool = False):
        self.state, self.since, self.spot, self.doze_on_landing = LAND, now, spot, doze

    def see_cursor(self, now: float, cursor):
        if cursor is not None and cursor != self.cursor:
            self.cursor, self.cursor_seen = cursor, now

    # -- the step
    def step(self, dt: float, now: float, cursor=None, bounds=None):
        self._now = now
        dt = min(max(dt, 0.0), 0.25)   # a stall (suspend, drag) must not fling the moth across the screen
        if bounds is not None:
            self.bounds = tuple(bounds)
        self.see_cursor(now, cursor)
        idle = now - self.idle_since >= DOZE_AFTER
        st, t_in = self.state, now - self.since
        if st == HELD:
            return
        if st == REST:
            if self.asleep or (idle and not self.excited):
                self.state, self.since = DOZE, now
            elif self.wander and (self.excited or t_in > self.rest_for) and self.attend is None:
                self.take_off(now)
            elif self.attend is not None and self.wander:
                self.state, self.since = FLY, now
            return
        if st == DOZE:
            return
        if not self.wander and st == FLY:
            st = self.state = LAND
            self.spot = self.home()
        if st == FLY:
            if self.asleep or (idle and not self.excited and self.attend is None):
                self.land(now, self.nearest_floor(), doze=True)
            elif t_in > self.fly_for and not self.excited and self.attend is None:
                self.land(now, self.landing_spot())
        self._fly(dt, now)
        self._clamp()

    def _fly(self, dt: float, now: float):
        bx, by, bw, bh = self.bounds
        ex = self.excited
        if self.state == LAND or self.attend is not None:
            tx, ty = self.spot if self.state == LAND else self.attend
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if self.state == LAND and dist < 3.0:
                self.x, self.y, self.vx, self.vy = tx, ty, 0.0, 0.0
                self.state, self.since = (DOZE if self.doze_on_landing or self.asleep else REST), now
                self.rest_for = self.rng.uniform(10, 40)
                return
            speed = min(self.SPEED * 1.3, dist * 1.6) + (0 if self.state == LAND else 12 * self.turn(now * 3))
            want = (dx / (dist or 1) * speed, dy / (dist or 1) * speed)
            if self.attend is not None and self.state != LAND:   # hover: a lazy figure-eight around the point
                want = (want[0] + 25 * math.sin(now * 1.7), want[1] + 14 * math.sin(now * 3.4))
            k = 3.0
        else:
            self.heading += self.turn(now) * (4.0 if ex else 1.6) * dt
            speed = self.SPEED * (1.0 + 0.35 * self.pace(now)) * (1.8 if ex else 1.0)
            wx, wy = math.cos(self.heading) * speed, math.sin(self.heading) * speed
            # moths love light: drawn to the cursor while it moves, circling it instead of sitting on it
            c, w = self.cursor, math.exp(-(now - self.cursor_seen) / 8.0)
            if c is not None and w > 0.02:
                dx, dy = c[0] - self.x, c[1] - self.y
                r = math.hypot(dx, dy) or 1.0
                pull = 0.45 * w * min(1.0, max(-0.6, (r - 120) / 260))
                wx += (dx / r * pull + (-dy / r) * 0.35 * w * (r < 300)) * speed
                wy += (dy / r * pull + (dx / r) * 0.35 * w * (r < 300)) * speed
            # soft walls: steer back in near the edges
            m = self.size * 1.5
            for d, ax, ay in ((self.x - bx, 1, 0), (bx + bw - self.x, -1, 0), (self.y - by, 0, 1),
                              (by + bh - self.y, 0, -1)):
                if d < m:
                    push = (1 - max(d, 0) / m) ** 2 * speed * 2.2
                    wx, wy = wx + ax * push, wy + ay * push
            self.heading = math.atan2(wy, wx) if (wx or wy) else self.heading
            want, k = (wx, wy), 2.0
        a = min(1.0, dt * k)
        self.vx += (want[0] - self.vx) * a
        self.vy += (want[1] - self.vy) * a
        self.x += self.vx * dt
        self.y += self.vy * dt

    def _clamp(self):
        bx, by, bw, bh = self.bounds
        s = self.size
        self.x = min(max(self.x, bx + s * 0.4), bx + bw - s * 0.4)
        self.y = min(max(self.y, by + s * 0.4), by + bh - s * 0.38)

    # -- what the painter and the timer want to know
    def flying(self) -> bool:
        return self.state in (FLY, LAND, HELD) or (self.excited and self.state == REST)

    def fps(self) -> int:
        return 30 if self.flying() else (12 if self.state == REST else 8)


# --- the sprite (Qt) ------------------------------------------------------------------------------------------
# The moth is real art, not drawn here: "202206 Silkmoth adult female icon" by DataBase Center for Life Science
# (DBCLS), https://commons.wikimedia.org/wiki/File:202206_Silkmoth_adult_female_icon.svg, used under CC BY 4.0.
# moth/assets/ holds the original SVG (unmodified) and CREDITS.md. Its wings and body are one outline path, so the
# render is cut along the body instead: BODY_LEFT is the left half of the body's outline (head, the ring of dots
# round the thorax, the abdomen) in the SVG's 1000x1000 units, mirrored for the right half. Each wing is the art on
# its side of the centre line minus the body; the body layer is the art inside the outline plus the antennae (which
# the wing layers leave out). A frame squashes each wing toward its root (PIVOT, at the body's side) and turns it
# a little, then draws the body on top. Nothing is drawn new; only scaled, turned, cut and layered.
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ART_SVG = os.path.join(ASSETS, "silkmoth_adult_female_icon.svg")
ART_BOX = (0.0, 80.0, 1000.0, 780.0)   # the part of the 1000x1000 canvas the art fills (its outline: x 8-992, y 90-838)
ART_ASPECT = ART_BOX[3] / ART_BOX[2]   # height / width
BODY_LEFT = [(500, 366), (452, 370), (440, 384), (436, 404), (424, 412), (404, 432), (394, 458), (396, 490),
             (408, 516), (410, 530), (404, 560), (402, 610), (406, 690), (416, 730), (426, 770), (440, 810),
             (456, 842), (476, 852), (500, 852)]
PIVOT = (410.0, 470.0)                 # the left wing's root; the right wing's is mirrored
# frames: (wing width as a fraction of the open wing, turn in degrees; negative sweeps the tips back)
WINGS = [(1.0, 0), (0.8, -2), (0.58, -4), (0.38, -5), (0.24, -5),   # 0-4: open .. up (a wingbeat)
         (0.62, -20),                                               # 5: resting, wings narrow and swept back
         (0.5, -30)]                                                # 6: dozing, folded down further
OPEN, UP, RESTING, DOZING = 0, 4, 5, 6
FLAP = (0, 1, 2, 3, 4, 3, 2, 1)   # one wingbeat
FRAME_S = 0.05                    # s a frame: a wingbeat in 0.4 s
SHIVER_S = 0.36                   # a resting wing flick: 6 frames of 0.06 s (pose_for)
STILL_MS = 250                    # ms between ticks while the moth sits still (no flick, fade or flight due)
DOZE_FPS = 4                      # frames per second of the doze z's (the moth; a skin keeps its own rate)
MAX_TILT = 14.0                   # degrees: leaning toward where it flies
ART_CREDIT = ("moth art: \"202206 Silkmoth adult female icon\" by DBCLS (Wikimedia Commons), CC BY 4.0, "
              "modified (wing frames)")


@dataclass
class Pose:
    """How the moth looks in one frame: which wing frame (WINGS), the lean in degrees (clockwise, toward the
    right), and z: the doze animation phase (None = no z's)."""
    frame: int = RESTING
    tilt: float = 0.0
    z: float | None = None


_ART: dict = {}


def _svg_parts() -> tuple[bytes, bytes]:
    """(the art without its antennae, only the antennae) as SVG documents."""
    if "svg" not in _ART:
        try:
            with open(ART_SVG, encoding="utf-8") as f:
                src = f.read()
        except OSError as e:
            raise FileNotFoundError(f"moth's art is missing: {ART_SVG}") from e
        antennae = re.findall(r'<path class="st3"[^>]*/>', src, re.DOTALL)
        if '<path' not in src or len(antennae) != 2:
            raise FileNotFoundError(f"moth's art isn't the expected silkmoth SVG: {ART_SVG}")
        wings = src
        for a in antennae:
            wings = wings.replace(a, "")
        only = re.sub(r'<path(?! class="st3")[^>]*/>', "", src, flags=re.DOTALL)
        _ART["svg"] = (wings.encode(), only.encode())
    return _ART["svg"]


def _body_path(half: str | None = None):
    """The body outline as a QPainterPath (SVG units); "left"/"right": that side of the canvas minus the body."""
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QPainterPath, QPolygonF

    body = QPainterPath()
    body.addPolygon(QPolygonF([QPointF(x, y) for x, y in BODY_LEFT + [(1000 - x, y) for x, y in reversed(BODY_LEFT)]]))
    body.closeSubpath()
    if half is None:
        return body
    side = QPainterPath()
    side.addRect(QRectF(0, 0, 500, 1000) if half == "left" else QRectF(500, 0, 500, 1000))
    return side.subtracted(body)


def _layer(svg: bytes, pw: int, clip=None):
    """The art (or part of it) rendered pw px wide, cropped to ART_BOX, clipped to `clip` (SVG units)."""
    from PySide6.QtCore import QByteArray, QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    r = QSvgRenderer(QByteArray(svg))
    if not r.isValid():
        raise FileNotFoundError(f"moth's art can't be read: {ART_SVG}")
    s = pw / ART_BOX[2]
    img = QImage(pw, round(ART_BOX[3] * s), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(s, s)
    p.translate(-ART_BOX[0], -ART_BOX[1])
    if clip is not None:
        p.setClipPath(clip)
    r.render(p, QRectF(0, 0, 1000, 1000))
    p.end()
    return img


def load_art():
    """Check the art is there and readable (raises FileNotFoundError); renders it once, small."""
    sprites(16)


def sprites(pw: int) -> list:
    """All WINGS frames as QImages pw px wide (ART_ASPECT high). Composed at 2x from the cut layers, then scaled
    down smoothly. Cached per width."""
    if pw not in _ART:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPainter, QTransform

        big = pw * 2
        wings_svg, antennae_svg = _svg_parts()
        left, right = _layer(wings_svg, big, _body_path("left")), _layer(wings_svg, big, _body_path("right"))
        body = _layer(wings_svg, big, _body_path())
        p = QPainter(body)
        p.drawImage(0, 0, _layer(antennae_svg, big))
        p.end()
        s = big / ART_BOX[2]
        out = []
        for sx, turn in WINGS:
            img = QImage(body.size(), QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(0)
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            for layer, side in ((left, 1), (right, -1)):
                px = PIVOT[0] if side == 1 else 1000 - PIVOT[0]
                x, y = (px - ART_BOX[0]) * s, (PIVOT[1] - ART_BOX[1]) * s
                t = QTransform()
                t.translate(x, y)
                t.rotate(turn * side)
                t.scale(sx, 1)
                t.translate(-x, -y)
                p.setTransform(t)
                p.drawImage(0, 0, layer)
            p.resetTransform()
            p.drawImage(0, 0, body)
            p.end()
            out.append(img.scaled(pw, round(pw * ART_ASPECT), Qt.AspectRatioMode.IgnoreAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
        _ART[pw] = out
    return _ART[pw]


def paint_moth(p, cx: float, cy: float, width: float, pose: Pose):
    """Draw the moth `width` px wide centred on (cx, cy) with QPainter p, leaning pose.tilt degrees (plus the doze
    z's)."""
    from PySide6.QtCore import QPointF

    dpr = p.device().devicePixelRatioF() or 1.0   # HiDPI / fractional scaling: render at device pixels
    pw = max(8, round(width * dpr))
    img = sprites(pw)[pose.frame]
    img.setDevicePixelRatio(pw / width)
    w, h = width, width * ART_ASPECT
    p.save()
    p.setRenderHint(p.RenderHint.SmoothPixmapTransform, True)
    p.translate(cx, cy)
    if pose.tilt:
        p.rotate(pose.tilt)
    p.drawImage(QPointF(-w / 2, -h / 2), img)
    p.restore()
    if pose.z is not None:
        paint_z(p, cx, cy, w, h, pose.z)


_Z_GLYPHS: dict = {}   # pixel size -> the "z" outline at (0, 0); building one runs the font's hinter, so it's cached


def z_glyph(px: int):
    zp = _Z_GLYPHS.get(px)
    if zp is None:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QFont, QPainterPath
        f = QFont()
        f.setFamilies(["Fraunces", "Georgia", "DejaVu Serif", "serif"])
        f.setItalic(True)
        f.setWeight(QFont.Weight.DemiBold)
        f.setPixelSize(px)
        zp = _Z_GLYPHS[px] = QPainterPath()
        zp.addText(QPointF(0, 0), f, "z")
    return zp


def paint_z(p, cx: float, cy: float, w: float, h: float, phase: float):
    """The doze z's (a font glyph, not art): they drift up and to the right from the head (cx, cy - 0.36 h), fading."""
    from PySide6.QtGui import QColor, QPen

    p.save()
    p.setRenderHint(p.RenderHint.Antialiasing, True)
    for i in range(3):
        ph = (phase * 0.45 + i / 3) % 1.0
        al = max(0.0, min(1.0, ph * 5)) * min(1.0, (1 - ph) * 2.2)
        # addText(point) puts the glyph at the point cut to 1/64 px (QFixed::fromReal); the same cut keeps the pixels
        x, y = cx + w * (0.06 + 0.22 * ph), cy - h * (0.36 + 0.55 * ph)
        zp = z_glyph(max(9, int(h * (0.24 + 0.18 * ph)))).translated(int(x * 64) / 64, int(y * 64) / 64)
        c, e = QColor(LAMP), QColor(INK)
        c.setAlphaF(al)
        e.setAlphaF(al * 0.8)
        p.setPen(QPen(e, 0.8))
        p.setBrush(c)
        p.drawPath(zp)
    p.restore()


def pose_for(w: Wander, now: float, anim: dict) -> Pose:
    """The frame for Wander's state. anim keeps what carries over between frames (lean, wingbeat, shivers)."""
    dt = min(0.25, max(0.0, now - anim.get("t", now)))
    anim["t"] = now
    want = 0.0
    if w.flying() and w.state != HELD:   # lean toward where it flies (left/right speed only: it's a top view)
        want = max(-1.0, min(1.0, w.vx / Wander.SPEED)) * MAX_TILT
    anim["tilt"] = anim.get("tilt", 0.0) + (want - anim.get("tilt", 0.0)) * min(1.0, dt * 5)
    tilt = anim["tilt"] if abs(anim["tilt"]) > 0.2 else 0.0
    if w.state == DOZE:
        return Pose(DOZING, tilt, z=now)
    if w.flying():
        per = 0.032 if (w.excited or w.state == HELD) else FRAME_S   # excited / held: a faster wingbeat
        anim["beat"] = anim.get("beat", 0.0) + dt / per
        return Pose(FLAP[int(anim["beat"]) % len(FLAP)], tilt)
    if anim.get("thinking"):             # waiting on moth's reply: slow wingbeats where it sits
        return Pose(FLAP[int(now / 0.09) % len(FLAP)], tilt)
    if now > anim.get("shiver_next", 0):   # resting: now and then the wings flick open once
        anim["shiver_at"] = now
        anim["shiver_next"] = now + w.rng.uniform(4, 11)
    k = int((now - anim.get("shiver_at", -9)) / 0.06)
    return Pose((2, 1, OPEN, OPEN, 1, 2)[k] if 0 <= k < 6 else RESTING, tilt)


def bob(w: Wander, now: float) -> float:
    return math.sin(now * math.tau * 1.2) * w.size * 0.035 if w.flying() and w.state != HELD else 0.0


# --- Petdex / Codex sprite sheet --------------------------------------------------------------------------------
CELL_W, CELL_H, COLS = 192, 208, 8
PETDEX_ROWS = [("idle", 6), ("running-right", 8), ("running-left", 8), ("waving", 4), ("jumping", 5),
               ("failed", 8), ("waiting", 6), ("running", 6), ("review", 6)]
PETDEX_W = 160   # moth width in the 192-wide cell: room for the lean (a 14-degree lean is ~185 px wide)


def petdex_frame(state: str, i: int, n: int) -> tuple[Pose, float, float]:
    """(pose, dx, dy) of frame i of n for a Petdex row; dx/dy offset the moth in the cell (px). Every row reuses
    the WINGS frames (leaning / offset); none is drawn new."""
    ph = i / n * math.tau
    if state == "idle":                       # resting, the wings flick open once
        return Pose((RESTING, RESTING, 2, OPEN, 2, RESTING)[i % 6]), 0, 0
    if state in ("running-right", "running-left"):   # flying, leaning into it
        lean = MAX_TILT if state == "running-right" else -MAX_TILT
        return Pose(FLAP[i % 8], lean), 0, -8 * math.sin(ph * 2)
    if state == "waving":                     # a wingbeat with a bob
        return Pose((OPEN, 2, UP, 2)[i % 4]), 0, -6 * math.sin(ph)
    if state == "jumping":                    # wings up, a hop, wings down
        hop = (0.0, -0.6, -1.0, -0.6, 0.0)[i % 5]
        return Pose((RESTING, UP, 2, OPEN, RESTING)[i % 5]), 0, 26 * hop
    if state == "failed":                     # wings fold down, it shudders, droops and sinks
        k = i / (n - 1)
        return Pose(RESTING if i < 2 else DOZING, 6 * k + 3 * (1 if i % 2 else -1) * (1 - k)), 0, 12 * k
    if state == "waiting":                    # resting and still, breathing a little
        return Pose(RESTING), 0, 2 * math.sin(ph)
    if state == "running":                    # a wingbeat, bobbing
        return Pose((OPEN, 1, 3, UP, 3, 1)[i % 6]), 0, -6 * math.sin(ph * 2)
    if state == "review":                     # looks one way, then the other
        return Pose(RESTING, (0, -10, -10, 0, 10, 10)[i % 6]), 0, 0
    return Pose(), 0, 0


def render_petdex_sheet():
    """The 1536x1872 sheet as a QImage (needs a QGuiApplication)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter

    img = QImage(CELL_W * COLS, CELL_H * len(PETDEX_ROWS), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    for row, (state, n) in enumerate(PETDEX_ROWS):
        for i in range(n):
            pose, dx, dy = petdex_frame(state, i, n)
            cx = i * CELL_W + CELL_W / 2 + dx
            cy = row * CELL_H + CELL_H / 2 + 10 + dy     # a little low: room for hops above
            p.save()
            p.setClipRect(i * CELL_W, row * CELL_H, CELL_W, CELL_H)
            paint_moth(p, cx, cy, PETDEX_W, pose)
            p.restore()
    p.end()
    return img


def export_petdex(out_dir: str) -> list[str]:
    """Write out_dir/pet.json + out_dir/spritesheet.png (and CREDITS.txt: the art's CC BY 4.0 attribution travels
    with the pet); returns the two pet paths."""
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv[:1])
    os.makedirs(out_dir, exist_ok=True)
    sheet = os.path.join(out_dir, "spritesheet.png")
    if not render_petdex_sheet().save(sheet, "PNG"):
        raise OSError(f"could not write {sheet}")
    meta = {"id": "moth", "displayName": "moth",
            "description": f"a silly, cute, slightly dim moth that lives on a graphics card. {ART_CREDIT}",
            "spritesheetPath": "spritesheet.png"}
    pj = os.path.join(out_dir, "pet.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "CREDITS.txt"), "w", encoding="utf-8") as f:
        f.write(PETDEX_CREDITS)
    return [pj, sheet]


PETDEX_CREDITS = """\
spritesheet.png: "202206 Silkmoth adult female icon" by DataBase Center for Life Science (DBCLS)
  https://commons.wikimedia.org/wiki/File:202206_Silkmoth_adult_female_icon.svg (doi:10.7875/togopic.2022.231)
  License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
  Changes by moth (https://huggingface.co/darioooooo0o/moth-112m): the art was cut along the body into two wings
  and the body; wing frames were made by squashing the wings toward the body and turning them a little; the frames
  were scaled, leaned and offset into the Petdex 8x9 grid. Nothing was drawn new.
"""


# --- skins: an installed Petdex pet instead of the moth ---------------------------------------------------------
SHEET_W, SHEET_H = CELL_W * COLS, CELL_H * len(PETDEX_ROWS)   # 1536x1872: the only size Petdex / Codex accept
MAX_SHEET_BYTES = 32 << 20     # a 1536x1872 PNG/WebP is a few MB at most
SKIN_H = 1.3                   # a skin cell is drawn this many pet sizes high (the window is 1.75 sizes square)
SKIN_FPS = {"idle": 6, "running-right": 10, "running-left": 10, "waving": 8, "jumping": 10, "failed": 8,
            "waiting": 4, "running": 10, "review": 6}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SkinError(Exception):
    """Why a skin can't be worn (shown in the bubble; the pet is moth instead)."""


def pet_dirs() -> list[str]:
    """Where installed Petdex pets live, in lookup order: the Petdex app, Codex, Hermes."""
    home = os.path.expanduser("~")
    return [os.path.join(home, ".petdex", "pets"),
            os.path.join(os.environ.get("CODEX_HOME") or os.path.join(home, ".codex"), "pets"),
            os.path.join(os.environ.get("HERMES_HOME") or os.path.join(home, ".hermes"), "pets")]


def _is_id(spec: str) -> bool:
    return bool(_ID.fullmatch(spec)) and ".." not in spec


def find_skin(spec: str | None) -> str | None:
    """The folder of a skin (it has a pet.json), or None: moth itself, or nothing found. A spec that looks like a path
    (has a slash, starts with ~ or .) is only a folder; an id is looked up in pet_dirs(), then as a folder here."""
    spec = (spec or "").strip()
    if not spec or spec == "moth":
        return None
    folder = os.path.abspath(os.path.expanduser(spec))
    if "/" in spec or os.sep in spec or spec.startswith(("~", ".")):
        return folder if os.path.isfile(os.path.join(folder, "pet.json")) else None
    if _is_id(spec):
        for d in pet_dirs():
            f = os.path.join(d, spec)
            if os.path.isfile(os.path.join(f, "pet.json")):
                return f
    return folder if os.path.isfile(os.path.join(folder, "pet.json")) else None


def _display_name(meta: dict, fallback: str) -> str:
    for k in ("displayName", "id"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            name = " ".join(clean(v).split())
            return name if len(name) <= 40 else name[:39] + "…"
    return fallback


def installed_skins() -> list[tuple[str, str, str]]:
    """[(id, folder, display name)] of the installed Petdex pets, first found wins per id, sorted by id. An installed
    "moth" (the export) is left out: the built-in moth is always there."""
    seen: dict[str, tuple[str, str, str]] = {}
    for d in pet_dirs():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for sid in names:
            f = os.path.join(d, sid)
            if sid in seen or sid == "moth" or not _is_id(sid) or not os.path.isfile(os.path.join(f, "pet.json")):
                continue
            try:
                with open(os.path.join(f, "pet.json"), encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                meta = {}
            seen[sid] = (sid, f, _display_name(meta if isinstance(meta, dict) else {}, sid))
    return [seen[k] for k in sorted(seen)]


def read_pet(folder: str) -> tuple[str, str]:
    """(display name, sheet path) from folder/pet.json; SkinError when it isn't a usable Petdex pet. The sheet must
    be a file inside the folder (spritesheetPath; spritesheet.png / .webp when the key is missing)."""
    try:
        with open(os.path.join(folder, "pet.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise SkinError("it has no pet.json") from None
    except (OSError, ValueError) as e:
        raise SkinError(f"its pet.json can't be read ({type(e).__name__})") from None
    if not isinstance(meta, dict):
        raise SkinError("its pet.json isn't a JSON object")
    rel = meta.get("spritesheetPath")
    if rel is None:
        rel = next((n for n in ("spritesheet.png", "spritesheet.webp") if os.path.isfile(os.path.join(folder, n))),
                   "spritesheet.png")
    if not isinstance(rel, str) or not rel.strip() or os.path.isabs(rel):
        raise SkinError("its pet.json's spritesheetPath isn't a file in the pet's folder")
    base = os.path.realpath(folder)
    sheet = os.path.realpath(os.path.join(base, rel))
    if os.path.commonpath([sheet, base]) != base:
        raise SkinError("its pet.json's spritesheetPath points outside the pet's folder")
    if not os.path.isfile(sheet):
        raise SkinError(f"its sprite sheet is missing ({rel})")
    if os.path.getsize(sheet) > MAX_SHEET_BYTES:
        raise SkinError(f"its sprite sheet is too big ({os.path.getsize(sheet) >> 20} MB)")
    return _display_name(meta, os.path.basename(base)), sheet


def skin_row(state: str, flying: bool, excited: bool, vx: float, thinking: bool, rows) -> tuple[str, bool, float]:
    """(Petdex row, mirrored, frames per second) for what the pet is doing; rows = the rows the sheet has frames in
    (idle always). Flying: running-right / running-left by direction, the other row mirrored if one is empty, else
    "running"; excited: jumping in the air, waving on the ground; resting: idle (review while waiting on a reply);
    dozing: waiting, else idle at a third of its speed."""
    def pick(*names):
        return next((n for n in names if n in rows), "idle")
    if state == DOZE:
        return ("waiting", False, SKIN_FPS["waiting"]) if "waiting" in rows else ("idle", False, SKIN_FPS["idle"] / 3)
    if state == HELD:
        r = pick("waving", "jumping")
        return r, False, SKIN_FPS[r]
    if flying:
        if excited and "jumping" in rows:
            return "jumping", False, SKIN_FPS["jumping"]
        want, other = ("running-right", "running-left") if vx >= 0 else ("running-left", "running-right")
        if want in rows:
            return want, False, SKIN_FPS[want]
        if other in rows:
            return other, True, SKIN_FPS[other]
        r = pick("running")
        return r, False, SKIN_FPS[r]
    if excited:
        r = pick("waving", "jumping")
        return r, False, SKIN_FPS[r]
    if thinking and "review" in rows:
        return "review", False, SKIN_FPS["review"]
    return "idle", False, SKIN_FPS["idle"]


class Skin:
    """A Petdex pet cut from its sheet: rows[name] = its non-empty 192x208 cells (QImages); top / bottom: the
    highest and lowest opaque pixel row over all frames (cell px), to place the bubble on the visible pet."""

    def __init__(self, sid: str, name: str, folder: str, rows: dict, top: int, bottom: int):
        self.id, self.name, self.folder, self.rows, self.top, self.bottom = sid, name, folder, rows, top, bottom
        self._scaled: dict = {}

    def size(self, pet_size: float) -> tuple[float, float]:
        """(width, height) a cell is drawn at for a pet `pet_size` px wide."""
        h = pet_size * SKIN_H
        return h * CELL_W / CELL_H, h

    def frames(self, row: str, mirrored: bool, ph: int) -> list:
        """The row's frames scaled to ph device pixels high (cached)."""
        key = (row, mirrored, ph)
        if key not in self._scaled:
            from PySide6.QtCore import Qt
            pw = max(1, round(ph * CELL_W / CELL_H))
            out = []
            for img in self.rows.get(row) or self.rows["idle"]:
                if mirrored:
                    img = img.flipped(Qt.Orientation.Horizontal) if hasattr(img, "flipped") else img.mirrored(True, False)
                out.append(img.scaled(pw, ph, Qt.AspectRatioMode.IgnoreAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
            if len(self._scaled) > 24:
                self._scaled.clear()
            self._scaled[key] = out
        return self._scaled[key]

    def icon_image(self, px: int = 64):
        """The first idle frame centred on a transparent square (the tray icon)."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPainter
        img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        f = self.rows["idle"][0].copy(0, self.top, CELL_W, max(1, self.bottom - self.top + 1))
        f = f.scaled(px, px, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        p = QPainter(img)
        p.drawImage((px - f.width()) // 2, (px - f.height()) // 2, f)
        p.end()
        return img


def load_skin(spec: str) -> Skin:
    """The Petdex pet `spec` (id or folder) as a Skin; SkinError (why) when it can't be worn. Needs PySide6 (QImage;
    no application object)."""
    folder = find_skin(spec)
    if folder is None:
        raise SkinError(f"there's no petdex pet {spec!r} installed" if spec and spec != "moth" else "moth is built in")
    name, sheet = read_pet(folder)
    from PySide6.QtGui import QImage, QImageReader

    reader = QImageReader(sheet)
    dims = reader.size()   # from the header, before decoding anything
    if dims.isValid() and (dims.width(), dims.height()) != (SHEET_W, SHEET_H):
        raise SkinError(f"its sheet is {dims.width()}x{dims.height()}, a petdex sheet is {SHEET_W}x{SHEET_H}")
    img = reader.read()
    if img.isNull():
        raise SkinError(f"its sheet can't be read ({reader.errorString() or 'not an image'})")
    if (img.width(), img.height()) != (SHEET_W, SHEET_H):
        raise SkinError(f"its sheet is {img.width()}x{img.height()}, a petdex sheet is {SHEET_W}x{SHEET_H}")
    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    alpha = img.convertToFormat(QImage.Format.Format_Alpha8)
    bpl = alpha.bytesPerLine()
    buf = bytes(alpha.constBits())[:bpl * SHEET_H].translate(bytes([0] * 16 + list(range(16, 256))))   # ignore haze
    rows: dict = {}
    extent: dict = {}   # row -> (top, bottom) of its opaque pixels
    for r, (row, n) in enumerate(PETDEX_ROWS):
        frames, top, bottom = [], CELL_H, -1
        for i in range(n):
            x0, y0 = i * CELL_W, r * CELL_H
            ys = [y for y in range(CELL_H) if buf[(y0 + y) * bpl + x0:(y0 + y) * bpl + x0 + CELL_W].strip(b"\0")]
            if not ys:
                continue   # an empty cell: not a frame
            frames.append(img.copy(x0, y0, CELL_W, CELL_H))
            top, bottom = min(top, ys[0]), max(bottom, ys[-1])
        if frames:
            rows[row], extent[row] = frames, (top, bottom)
    if "idle" not in rows:
        raise SkinError("its idle row is empty")
    # the bubble follows the everyday rows (a jump may poke into it for a moment)
    common = [extent[r] for r in ("idle", "running-right", "running-left", "waiting") if r in extent]
    top, bottom = min(t for t, _ in common), max(b for _, b in common)
    return Skin(os.path.basename(os.path.normpath(folder)), name, folder, rows, top, bottom)


def wear(spec: str | None) -> tuple[Skin | None, str | None]:
    """(the skin, None) or (None, why not) for spec; ("moth" or nothing: (None, None), the built-in moth)."""
    spec = (spec or "moth").strip() or "moth"
    if spec == "moth":
        return None, None
    try:
        return load_skin(spec), None
    except SkinError as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001  a broken pet never stops the moth
        return None, f"it broke ({type(e).__name__})"


def paint_skin(p, skin: Skin, cx: float, cy: float, pet_size: float, row: str, mirrored: bool, i: int):
    """Draw frame i (wrapping) of a skin row centred on (cx, cy)."""
    from PySide6.QtCore import QPointF
    dpr = p.device().devicePixelRatioF() or 1.0
    w, h = skin.size(pet_size)
    ph = max(8, round(h * dpr))
    frames = skin.frames(row, mirrored, ph)
    img = frames[i % len(frames)]
    img.setDevicePixelRatio(ph / h)
    p.save()
    p.setRenderHint(p.RenderHint.SmoothPixmapTransform, True)
    p.drawImage(QPointF(cx - w / 2, cy - h / 2), img)
    p.restore()


# --- updates ----------------------------------------------------------------------------------------------------
UPDATE_EVERY = 24 * 3600.0     # s between automatic checks (wall clock, kept in config.json across restarts)
UPDATE_EVAL = 600.0            # s between looking whether one is due (the pet runs for days)
UPDATE_TIMEOUT = 180.0
PIP_HINT = "pip install -U moth-companion"


def update_due(enabled: bool, offline: bool, last: float, now: float) -> bool:
    return enabled and not offline and (now - last >= UPDATE_EVERY or last > now + 3600)   # clock went back: check


def check_for_update(timeout: float = UPDATE_TIMEOUT) -> tuple[str, str]:
    """("update", what to run) / ("none", "") / ("error", why). Runs `python -m moth update --check` (exit 10 =
    update available) as a child process: the network calls and huggingface_hub stay out of the pet, and its
    printing can't mix with the pet's. Blocking: call it from a thread."""
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the folder holding moth/: same code
    try:
        p = subprocess.run([sys.executable, "-m", "moth", "update", "--check"], cwd=code_dir, capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "error", "the check took too long"
    except OSError as e:
        return "error", str(e)
    out = p.stdout or ""
    if p.returncode == 10:
        if "newer code:" in out:
            return "update", PIP_HINT
        return "update", "moth update" if launch.weights.mode() == "pip" else "python -m moth update"
    if p.returncode == 0:
        return "none", ""
    last = [x for x in (out + "\n" + (p.stderr or "")).splitlines() if x.strip()]
    return "error", (last[-1].strip() if last else f"exit {p.returncode}")[:160]


def update_line(hint: str) -> str:
    return f"there's a new me! `{hint}`" if hint == PIP_HINT else f"there's a new me! run `{hint}`"


# --- platform -------------------------------------------------------------------------------------------------
def choose_platform(env=None, platform: str | None = None) -> str:
    """Before QApplication: on a Wayland session ask for XWayland first (a Wayland client can't place its own
    windows). Respects an explicit QT_QPA_PLATFORM. Returns what was decided: "set", "xwayland", "wayland", "native"."""
    env = os.environ if env is None else env
    platform = platform or sys.platform
    if env.get("QT_QPA_PLATFORM"):
        return "set"
    if not platform.startswith("linux") or not (env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland"):
        return "native"
    if env.get("DISPLAY"):
        env["QT_QPA_PLATFORM"] = "xcb;wayland"   # Qt 6 tries these in order (xcb needs libxcb-cursor0)
        return "xwayland"
    return "wayland"


def _config(getter, default):
    """A config.py value, or default when the file can't be read (the pet never fails over a setting)."""
    try:
        return getter()
    except Exception:  # noqa: BLE001
        return default


# --- the windows (Qt) -----------------------------------------------------------------------------------------
def run(a, on_ready=None) -> int:
    """The pet's windows until quit. on_ready(pet): called once they exist (tests / offscreen renders)."""
    decided = choose_platform()
    try:
        import PySide6.QtWidgets  # noqa: F401
    except ImportError:
        print("the desktop pet needs PySide6 (moth itself doesn't):  pip install PySide6", file=sys.stderr)
        return 2

    # Ctrl+C, kill, the terminal closing: quit and stop a server the pet started. Before the windows exist that is
    # an exception through the finally below; after, Pet.quit (the Qt timer lets Python run the handler).
    import signal
    quit_now: list = []

    def on_signal(_sig, _frame):
        if quit_now:
            quit_now[0]()
        else:
            raise KeyboardInterrupt
    old = {}
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                old[sig] = signal.signal(sig, on_signal)
            except (ValueError, OSError):   # not the main thread (tests): the caller handles signals
                pass
    try:
        return _boot_and_run(a, on_ready, decided, quit_now)
    except KeyboardInterrupt:
        return 130
    finally:
        for sig, h in old.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError, TypeError):
                pass


def _boot_and_run(a, on_ready, decided: str, quit_now: list) -> int:
    # before QApplication and any thread: the server (if the pet starts one) is spawned from this thread
    boot = launch.prepare(a.url, no_server=getattr(a, "no_server", False), model=getattr(a, "model", None),
                          extra=list(getattr(a, "server_arg", None) or []) + list(getattr(a, "passthrough", None) or []),
                          token_file=a.token_file)
    if boot.message:
        print(f"moth pet: {boot.message}", file=sys.stderr, flush=True)
    try:
        return _run(a, on_ready, boot, decided, quit_now)
    finally:
        if boot.server:
            boot.server.stop()   # also on aboutToQuit and atexit; stop() runs once


def _run(a, on_ready, boot, decided: str, quit_now: list) -> int:
    """The windows and the Qt loop (run() has checked PySide6 and handled the server)."""
    from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, QUrl
    from PySide6.QtGui import (
        QActionGroup,
        QColor,
        QCursor,
        QDesktopServices,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QIcon,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRegion,
    )
    from PySide6.QtWidgets import QApplication, QLineEdit, QMenu, QSystemTrayIcon, QVBoxLayout, QWidget

    app = QApplication.instance() or QApplication(sys.argv[:1])
    if not isinstance(app, QApplication):   # a QGuiApplication made elsewhere: widgets would abort the process
        print("moth pet: needs a QApplication, found a QGuiApplication", file=sys.stderr)
        return 1
    app.setQuitOnLastWindowClosed(False)
    plat = QGuiApplication.platformName()
    native_wayland = plat.startswith("wayland")
    if native_wayland:
        print("moth pet: native Wayland (no XWayland): the compositor places the window, so moth sits still where "
              "it is put (drag works).", file=sys.stderr)
    tiling = any(x in (os.environ.get("XDG_CURRENT_DESKTOP", "") + os.environ.get("DESKTOP_SESSION", "")).lower()
                 for x in ("hyprland", "sway", "niri", "river", "i3"))
    bypass = plat == "xcb" and decided == "xwayland" and tiling   # unmanaged: never tiled, never focused
    size = int(min(160, max(32, a.size)))
    W = H = int(size * 1.75)
    try:
        load_art()
    except (FileNotFoundError, ImportError) as e:   # a pet without its art would only show its bubble
        print(f"moth pet: {e}", file=sys.stderr)
        return 1
    url, tfile = a.url.rstrip("/"), token_path(a.token_file)

    base = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus)
    if bypass:
        base |= Qt.WindowType.X11BypassWindowManagerHint

    def float_window(wd: QWidget, flags):
        wd.setWindowFlags(flags)
        wd.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wd.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if sys.platform == "darwin":
            wd.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)

    def screen_bounds(x, y):
        s = QGuiApplication.screenAt(QPoint(int(x), int(y))) or QGuiApplication.primaryScreen()
        g = s.availableGeometry()
        return g.x(), g.y(), g.width(), g.height()

    serif = QFont()
    serif.setFamilies(["Fraunces", "Georgia", "DejaVu Serif", "serif"])
    serif.setPixelSize(14)
    mono = QFont()
    mono.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Consolas", "monospace"])
    mono.setPixelSize(10)

    class BubbleWin(QWidget):
        """The speech bubble: its own window, transparent for input, so it never catches a click."""

        PAD, TAIL, RAD = 11, 9, 12

        def __init__(self):
            super().__init__()
            float_window(self, base | Qt.WindowType.WindowTransparentForInput)
            self.bubble, self.alpha, self.below, self.tail_x = None, 0.0, False, 0.5

        def set_bubble(self, b, alpha):
            changed = b is not self.bubble
            if changed:
                self.bubble = b
                fm, fc = QFontMetrics(serif), QFontMetrics(mono)
                tw = max(fm.horizontalAdvance(x) for x in b.lines)
                if b.caption:
                    tw = max(tw, fc.horizontalAdvance(b.caption))
                th = fm.lineSpacing() * len(b.lines) + (fc.lineSpacing() + 2 if b.caption else 0)
                self.resize(int(tw + 2 * self.PAD + 2), int(th + 2 * self.PAD + self.TAIL + 2))
            if changed or abs(alpha - self.alpha) > 0.02:
                self.alpha = alpha
                self.update()

        def place(self, mx, my, bounds, up=None, down=None):
            """Above the pet (up px over its centre, the sprite's visible top), or below (down px) at the top."""
            bx, by, bw, _bh = bounds
            w, h = self.width(), self.height()
            top = my - (size * 0.62 if up is None else up) - h
            self.below = top < by
            y = my + (size * 0.55 if down is None else down) if self.below else top
            x = min(max(mx - w * 0.35, bx + 4), bx + bw - w - 4)
            self.tail_x = min(max((mx - x) / w, 0.15), 0.85)
            self.move(int(x), int(y))

        def paintEvent(self, _e):
            b = self.bubble
            if not b:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setOpacity(self.alpha)
            w, h, T = self.width(), self.height(), self.TAIL
            body = QRectF(1, (T if self.below else 0) + 1, w - 2, h - T - 2)
            path = QPainterPath()
            path.addRoundedRect(body, self.RAD, self.RAD)
            tx = w * self.tail_x
            tail = QPainterPath()
            if self.below:
                tail.moveTo(tx - 7, body.top() + 1)
                tail.quadTo(tx - 1, body.top() - 2, tx - 3, 1)
                tail.quadTo(tx + 2, body.top() - 1, tx + 7, body.top() + 1)
            else:
                tail.moveTo(tx - 7, body.bottom() - 1)
                tail.quadTo(tx - 1, body.bottom() + 2, tx - 3, h - 1)
                tail.quadTo(tx + 2, body.bottom() + 1, tx + 7, body.bottom() - 1)
            path = path.united(tail)
            fill = QColor(NIGHT2)
            fill.setAlphaF(0.95)
            p.setPen(QPen(QColor(LAMP), 1.2))
            p.setBrush(fill)
            p.drawPath(path)
            y = body.top() + self.PAD
            if b.caption:
                p.setFont(mono)
                p.setPen(QColor(DUST))
                fc = QFontMetrics(mono)
                p.drawText(QPointF(self.PAD + 1, y + fc.ascent()), b.caption)
                y += fc.lineSpacing() + 2
            p.setFont(serif)
            p.setPen(QColor(PAPER))
            fm = QFontMetrics(serif)
            for line in b.lines:
                p.drawText(QPointF(self.PAD + 1, y + fm.ascent()), line)
                y += fm.lineSpacing()
            p.end()

    class InputWin(QWidget):
        """Click the moth: a small box; Enter sends to moth, Esc or clicking elsewhere closes it."""

        def __init__(self, on_send):
            super().__init__()
            flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.on_send = on_send
            lay = QVBoxLayout(self)
            lay.setContentsMargins(4, 4, 4, 4)
            self.edit = QLineEdit(self)
            self.edit.setMaxLength(500)
            self.edit.setPlaceholderText("talk to moth…")
            self.edit.setFont(serif)
            self.edit.setStyleSheet(
                f"QLineEdit {{ background: {NIGHT2}; color: {PAPER}; border: 1px solid {LAMP}; border-radius: 10px;"
                f" padding: 7px 11px; selection-background-color: {LAMP}; selection-color: {INK}; }}")
            lay.addWidget(self.edit)
            self.setFixedWidth(270)
            self.edit.returnPressed.connect(self.send)
            app.focusChanged.connect(self._focus)

        def open_at(self, mx, my, bounds):
            bx, by, bw, bh = bounds
            self.adjustSize()
            x = min(max(mx - self.width() / 2, bx + 4), bx + bw - self.width() - 4)
            y = my + size * 0.6
            if y + self.height() > by + bh:
                y = my - size * 0.7 - self.height()
            self.move(int(x), int(y))
            self.show()
            self.raise_()
            self.activateWindow()
            self.edit.setFocus()

        def send(self):
            text = self.edit.text().strip()
            self.edit.clear()
            self.hide()
            if text:
                self.on_send(text)

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Escape:
                self.edit.clear()
                self.hide()
            else:
                super().keyPressEvent(e)

        def _focus(self, _old, new):
            if self.isVisible() and new is None:
                QTimer.singleShot(150, lambda: self.hide() if not self.isActiveWindow() else None)

    class MothWin(QWidget):
        def __init__(self):
            super().__init__()
            float_window(self, base)
            self.setFixedSize(W, H)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.apply_mask(None)
            self.pose, self.dy = Pose(), 0.0
            self.look = None   # a skin's frame: (skin, row, mirrored, frame index, doze phase or None)
            self.press, self.dragging, self.grab_off = None, False, QPoint()

        def apply_mask(self, skin):
            """Click-through outside the sprite (+ bob) and the z's band above it. A static mask (no per-frame
            XShape), set again when the skin changes: a Petdex cell is taller than the moth."""
            m = 4 + int(size * 0.04)
            if skin is None:
                t = math.radians(MAX_TILT)   # the art's box at its biggest lean, plus the bob
                SW = int(size * (math.cos(t) + ART_ASPECT * math.sin(t))) + 2
                SH = int(size * (math.sin(t) + ART_ASPECT * math.cos(t)) + size * 0.07) + 2
            else:
                sw, sh = skin.size(size)
                SW, SH = int(sw) + 2, int(sh) + 2
            self.setMask(QRegion((W - SW) // 2 - m, (H - SH) // 2 - m, SW + 2 * m, SH + 2 * m)
                         .united(QRegion(0, 0, W, int(H * 0.4))))

        def paintEvent(self, _e):
            p = QPainter(self)
            if self.look is None:
                paint_moth(p, W / 2, H / 2 + self.dy, size, self.pose)
            else:
                skin, row, mirrored, i, z = self.look
                paint_skin(p, skin, W / 2, H / 2, size, row, mirrored, i)
                if z is not None:   # dozing: the moth's z's, from the pet's visible top
                    sw, sh = skin.size(size)
                    zh = size * ART_ASPECT
                    paint_z(p, W / 2 + sw * 0.12, H / 2 - sh / 2 + skin.top * sh / CELL_H + zh * 0.6, size, zh, z)
            p.end()

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self.press = e.globalPosition().toPoint()
                self.grab_off = self.press - self.pos()
                self.dragging = False
            elif e.button() == Qt.MouseButton.RightButton:
                pet.menu(e.globalPosition().toPoint())

        def mouseMoveEvent(self, e):
            if self.press is None:
                return
            g = e.globalPosition().toPoint()
            if not self.dragging and (g - self.press).manhattanLength() > 5:
                self.dragging = True
                pet.grab()
                if native_wayland and self.windowHandle() and self.windowHandle().startSystemMove():
                    self.press = None
                    return
            if self.dragging:
                self.move(g - self.grab_off)

        def mouseReleaseEvent(self, e):
            if e.button() != Qt.MouseButton.LeftButton or self.press is None:
                return
            self.press = None
            if self.dragging:
                pet.drop()
            else:
                pet.clicked()

        def closeEvent(self, e):   # the window closed from outside (WM close, Alt-F4): quit like the menu does
            e.accept()
            pet.quit()

    class Pet:
        def __init__(self):
            scr = QGuiApplication.primaryScreen().availableGeometry()
            bounds = (scr.x(), scr.y(), scr.width(), scr.height())
            now = time.monotonic()
            self.w = Wander(bounds, size, wander=not a.no_wander and not native_wayland, now=now)
            self.anim: dict = {}
            self.bubbles = BubbleQueue()
            self.muted = False
            self.thinking_until = 0.0
            self.events: queue.Queue = queue.Queue()
            self.stop = threading.Event()
            self.win, self.bub = MothWin(), BubbleWin()
            self.inp = InputWin(self.send)
            self.last = now
            self.last_raise = now
            self.complained = False
            self.win.move(int(self.w.x - W / 2), int(self.w.y - H / 2))
            self.win.show()
            self.timer = QTimer()
            self.timer.timeout.connect(self.tick)
            self.timer.start(33)
            self.poller = Poller(url, a.token_file)
            self.boot = boot
            self.booting = boot.state == "starting"
            self.downloading = boot.state == "downloading"
            self.started = threading.Event()   # set by the GUI once it has started the server after a download
            self.quiet_down = boot.state == "error"   # the bubble already says why; no "zzz" on top until moth is up
            if boot.state in ("starting", "downloading", "error"):
                self.server_down = True
                self.w.sleep(now, True)
                if self.booting:
                    self.say(WAKING, ttl=launch.READY_TIMEOUT)
                elif self.downloading:
                    self.say(DOWNLOADING, ttl=1e9)   # no time limit: it goes when the download is done
                else:
                    self.say(boot.message, ttl=NOTICE_TTL)
            # the look: --skin beats config.json's pet_skin; a skin that can't be worn = moth, and the bubble says why
            spec = getattr(a, "skin", None) or _config(config.pet_skin, "moth")
            self.skin, why = wear(spec)
            self.skin_anim: dict = {"row": None, "since": now}
            if why:
                self.say(f"i can't wear {spec!r}: {why}. i'm moth for now", ttl=NOTICE_TTL)
            self.win.apply_mask(self.skin)
            # tray icon where there is a tray (the same menu as the right-click one); none: nothing, silently
            self.tray_menu = self.make_menu()
            self.fill_menu(self.tray_menu)
            self.tray_menu.aboutToShow.connect(lambda: self.fill_menu(self.tray_menu))
            self.tray = None
            self.tray_until = now + 120.0   # started at login, the panel's tray may come up after the pet: retry
            self.next_tray = now
            self.make_tray(now)
            self.update_running = False
            self.next_update_eval = now + 30.0   # the first daily check: once moth is up and settled
            threading.Thread(target=self.bg, daemon=True, name="moth-pet-poll").start()

        # threads: never touch Qt, only the queue
        def bg(self):
            if self.downloading:   # pip install, first run: fetch the weights, then the GUI thread starts the server
                err = self.boot.fetch()
                if self.stop.is_set():
                    return
                self.events.put(("downloaded", err))
                if err:
                    print(f"moth pet: {err}", file=sys.stderr, flush=True)
                else:
                    while not self.started.wait(0.2):
                        if self.stop.is_set():
                            return
            if self.booting:   # our server is loading the model: wait for it, then poll as usual
                ok, why = self.boot.server.wait_ready(self.stop)
                if self.stop.is_set():   # quit while it loaded: aboutToQuit stops it
                    return
                self.events.put(("boot", (ok, why)))
                print(f"moth pet: {why or 'moth is awake'}", file=sys.stderr, flush=True)
            self.poll_loop()

        def poll_loop(self):
            while not self.stop.is_set():
                lines, status = self.poller.poll()
                if status:
                    self.events.put(("status", status))
                if lines:
                    self.events.put(("lines", lines))
                self.stop.wait(self.poller.interval())

        def send(self, text: str):
            self.thinking_until = time.monotonic() + CHAT_TIMEOUT
            self.w.attend = None

            def go():
                err = send_chat(url, tfile, text)
                self.events.put(("sent", err))
            threading.Thread(target=go, daemon=True, name="moth-pet-chat").start()

        # GUI
        def say(self, text, caption=None, ttl=None):
            if not self.muted:
                self.bubbles.push(text, caption, ttl)

        def make_tray(self, now):
            """The tray icon, if there is a tray (tried every 10 s for the first 2 minutes); else nothing, silently."""
            if self.tray is not None or now > self.tray_until or now < self.next_tray:
                return
            self.next_tray = now + 10.0
            try:
                if QSystemTrayIcon.isSystemTrayAvailable():
                    self.tray = QSystemTrayIcon(self.icon(), app)
                    self.tray.setToolTip("moth")
                    self.tray.setContextMenu(self.tray_menu)
                    self.tray.show()
            except Exception:  # noqa: BLE001  a tray is a nicety
                self.tray = None

        def notify(self, text, ttl=None):
            """say(), or the tray's own message when bubbles are muted (an update notice shouldn't vanish)."""
            if self.muted and self.tray is not None:
                self.tray.showMessage("moth", text, self.icon(), 10000)
            else:
                self.say(text, ttl=ttl)

        def quit(self):   # once: a second app.quit() would stay queued and end the next exec() at once (tests)
            if not self.stop.is_set():
                self.stop.set()
                if getattr(self, "tray", None) is not None:
                    self.tray.hide()
                app.quit()

        def handle(self, now):
            while True:
                try:
                    kind, val = self.events.get_nowait()
                except queue.Empty:
                    return
                if kind == "lines":
                    self.thinking_until = 0.0
                    for text, cap in val:
                        self.say(text, cap)
                    self.w.excite(now)
                elif kind == "downloaded":   # in the GUI thread: start the server here (PR_SET_PDEATHSIG)
                    self.downloading = False
                    self.bubbles.fade(DOWNLOADING)
                    err = val
                    if not err:
                        try:
                            self.boot.server.start()
                            self.booting = True
                            self.say(WAKING, ttl=launch.READY_TIMEOUT)
                        except OSError as e:
                            err = f"can't wake moth: {e}"
                    if err:
                        self.quiet_down = True
                        self.say(err, ttl=NOTICE_TTL)
                    self.started.set()
                elif kind == "boot":
                    ok, why = val
                    self.booting = False
                    self.bubbles.fade(WAKING)
                    if not ok:   # it died or never answered: say so once, then poll quietly
                        self.quiet_down = True
                        self.say(why, ttl=NOTICE_TTL)
                elif kind == "status" and val == "down":
                    self.w.sleep(now, True)
                    self.server_down = True
                    srv = self.boot.server
                    if self.booting or self.quiet_down:
                        pass
                    elif srv and srv.exited() and not srv.stopped:
                        self.say(f"moth's server stopped (see {srv.log})", ttl=NOTICE_TTL)
                    else:
                        self.say(ASLEEP)
                elif kind == "status" and val == "up":
                    self.quiet_down = False
                    if getattr(self, "server_down", False) and not self.menu_sleep:
                        self.w.sleep(now, False)
                        self.w.excite(now, 1.5)
                    self.server_down = False
                elif kind == "sent" and val:
                    self.thinking_until = 0.0
                    self.say(val)
                elif kind == "update":
                    self.update_running = False
                    res, info, manual = val
                    if res == "update":
                        self.notify(update_line(info), ttl=20)
                        self.w.excite(now, 1.5)
                    elif manual and res == "none":
                        self.notify("i'm the newest me already", ttl=8)
                    elif manual:
                        self.notify(f"i couldn't look for a new me: {info}", ttl=NOTICE_TTL)

        menu_sleep = False
        drawn = None       # (pose, dy, look) last handed to the window
        cursor_at = -1e9   # when the cursor was last read

        def tick(self):
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001  a pet never crashes; say it once and keep going
                if not self.complained:
                    self.complained = True
                    print(f"moth pet: {type(e).__name__}: {e}", file=sys.stderr)

        def _tick(self):
            now = time.monotonic()
            dt, self.last = now - self.last, now
            self.handle(now)
            if self.tray is None:
                self.make_tray(now)
            if now >= self.next_update_eval:
                self.next_update_eval = now + UPDATE_EVAL
                if not (self.booting or self.downloading):
                    self.maybe_check_updates()
            w = self.w
            if w.attend is not None and not self.inp.isVisible():
                w.attend = None   # the input box closed (Esc / clicked elsewhere): back to wandering
            if w.state == HELD:
                pos = self.win.pos()
                w.x, w.y = pos.x() + W / 2, pos.y() + H / 2
                if self.win.press is None and not (QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton):
                    self.drop()   # a compositor-driven move (Wayland) ends without a release event
            else:
                cursor = None   # QCursor.pos() is a round trip to the X server: every frame only while flying
                if w.state in (FLY, LAND) or now - self.cursor_at >= 1.0:
                    c, self.cursor_at = QCursor.pos(), now
                    cursor = (c.x(), c.y())
                w.step(dt, now, cursor=cursor, bounds=screen_bounds(w.x, w.y))
            self.anim["thinking"] = now < self.thinking_until
            pose = pose_for(w, now, self.anim)
            self.win.pose, self.win.dy = pose, bob(w, now)
            if self.skin is not None:
                row, mirrored, rate = skin_row(w.state, w.flying(), w.excited, w.vx, self.anim["thinking"],
                                               self.skin.rows)
                if self.skin_anim["row"] != (row, mirrored):
                    self.skin_anim.update(row=(row, mirrored), since=now)
                i = int((now - self.skin_anim["since"]) * rate)
                self.win.look = (self.skin, row, mirrored, i, now if w.state == DOZE else None)
                self.win.dy = 0.0
            else:
                self.win.look = None
            if w.state != HELD and not native_wayland:
                tx, ty = round(w.x - W / 2), round(w.y - H / 2)
                if (tx, ty) != (self.win.x(), self.win.y()):
                    self.win.move(tx, ty)
            drawn = (self.win.pose, self.win.dy, self.win.look)
            if drawn != self.drawn:   # the same frame again (a resting moth is a still image): no repaint
                self.drawn = drawn
                self.win.update()
            if bypass and now - self.last_raise > 3:   # unmanaged windows don't stay on top by themselves
                self.last_raise = now
                self.win.raise_()
                self.bub.raise_()
            b, alpha = self.bubbles.current()
            if b is None or self.muted:
                if self.bub.isVisible():
                    self.bub.hide()
            else:
                self.bub.set_bubble(b, alpha)
                g = self.win.geometry()
                up, down = self.bubble_gaps()
                self.bub.place(g.center().x(), g.center().y() + self.win.dy, screen_bounds(w.x, w.y), up, down)
                if not self.bub.isVisible():
                    self.bub.show()
            fps = 30 if (b is not None and alpha < 1) else w.fps()
            iv = int(1000 / fps)
            slow, due = None, now + 3600.0
            if (fps == w.fps() and w.state == REST and not w.excited and self.skin is None and not self.anim["thinking"]
                    and now - self.anim.get("shiver_at", -9.0) > SHIVER_S):
                # sitting still: a few ticks a second, but on time for the next wing flick (then 12 fps through it)
                slow, due = STILL_MS, self.anim.get("shiver_next", now)
            elif fps == w.fps() and w.state == DOZE and self.skin is None:
                slow = 1000 // DOZE_FPS   # asleep (all day, mostly): the z's drift on at a lower frame rate
            if slow:
                if b is not None and not self.muted:   # ... and on time for the bubble's fade-out
                    due = min(due, b.shown + b.ttl - self.bubbles.FADE_OUT)
                iv = max(iv, min(slow, int((due - now) * 1000) + 1))
            if self.timer.interval() != iv:
                self.timer.setInterval(iv)

        def kick(self):
            """Tick now (a click / a drag starts): don't wait out a still moth's slow tick."""
            self.timer.start(0)

        def grab(self):
            self.w.grab(time.monotonic())
            self.kick()

        def drop(self):
            pos = self.win.pos()
            sb = screen_bounds(pos.x() + W / 2, pos.y() + H / 2)
            self.w.bounds = sb
            self.w.drop(time.monotonic(), pos.x() + W / 2, pos.y() + H / 2)
            self.kick()

        def clicked(self):
            now = time.monotonic()
            self.w.poke(now)
            self.kick()
            if self.inp.isVisible():   # a second click closes the box (if the box never got focus, Esc can't)
                self.inp.hide()
                return
            g = self.win.geometry()
            self.inp.open_at(g.center().x(), g.center().y(), screen_bounds(self.w.x, self.w.y))
            if self.w.wander and not self.w.asleep:
                self.w.attend = (self.w.x, self.w.y)

        def bubble_gaps(self):
            """(px from the window centre up to the sprite's visible top, down to its bottom) plus a little air;
            (None, None) for the moth (BubbleWin's own numbers)."""
            if self.skin is None:
                return None, None
            _sw, sh = self.skin.size(size)
            k = sh / CELL_H
            return sh / 2 - self.skin.top * k + 4, self.skin.bottom * k - sh / 2 + 4

        def icon(self):
            """The tray icon: the skin's idle frame, or the moth with its wings open, on a square."""
            if self.skin is not None:
                return QIcon(QPixmap.fromImage(self.skin.icon_image(64)))
            img = QImage(64, 64, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(Qt.GlobalColor.transparent)
            art = sprites(64)[OPEN]
            p = QPainter(img)
            p.drawImage(0, (64 - art.height()) // 2, art)
            p.end()
            return QIcon(QPixmap.fromImage(img))

        # -- the menu (right-click on the moth, and the tray's)
        def make_menu(self):
            m = QMenu()
            m.setStyleSheet(f"QMenu {{ background: {NIGHT2}; color: {PAPER}; border: 1px solid {LINE}; padding: 4px; }}"
                            f"QMenu::item {{ padding: 5px 18px; }} QMenu::item:selected {{ background: {LINE};"
                            f" color: {LAMP}; }} QMenu::item:disabled {{ color: {DUST}; }}"
                            f" QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px; }}")
            m.setFont(serif)
            return m

        def fill_menu(self, m):
            """(Re)build m from the current state: labels, checks and the installed skins."""
            for sub in getattr(m, "_moth_subs", ()):
                sub.deleteLater()
            m.clear()
            m.addAction("open chat page").triggered.connect(self.open_chat)
            m.addAction("unmute bubbles" if self.muted else "mute bubbles").triggered.connect(self.toggle_mute)
            m.addAction("wake up" if self.menu_sleep else "sleep").triggered.connect(self.toggle_sleep)
            m.addSeparator()
            h = m.addAction("show moth in harnesses")
            h.setCheckable(True)
            h.setChecked(bool(_config(config.show_in_harness, False)))
            h.triggered.connect(self.set_harness)
            sub = self.make_menu()
            sub.setTitle("skin")
            m.addMenu(sub)
            self.fill_skins(sub)
            m._moth_subs = [sub]
            m.addAction("check for updates").triggered.connect(lambda *_: self.check_updates(manual=True))
            m.addSeparator()
            m.addAction("quit").triggered.connect(lambda *_: QTimer.singleShot(0, self.quit))

        def fill_skins(self, sub):
            group = QActionGroup(sub)
            group.setExclusive(True)
            mine = os.path.realpath(self.skin.folder) if self.skin is not None else None
            entries = [("moth", "moth", self.skin is None)]
            for sid, folder, name in installed_skins():
                entries.append((sid, name if name.lower() == sid.lower() else f"{name} ({sid})",
                                os.path.realpath(folder) == mine))
            if mine is not None and not any(on for *_, on in entries):   # worn from a folder outside the pet dirs
                entries.append((self.skin.folder, self.skin.name, True))
            for spec, label, on in entries:
                act = sub.addAction(label)
                act.setCheckable(True)
                act.setChecked(on)
                group.addAction(act)
                act.triggered.connect(lambda *_, s=spec: self.choose_skin(s))
            if len(entries) == 1:
                sub.addSeparator()
                sub.addAction("more: npx petdex install <pet>").setEnabled(False)

        def menu(self, at):
            m = self.make_menu()
            self.fill_menu(m)
            m.exec(at)

        # -- what the menu does
        def open_chat(self, *_):
            QDesktopServices.openUrl(QUrl(url))

        def toggle_mute(self, *_):
            self.muted = not self.muted
            if self.muted:
                self.bubbles.clear()

        def toggle_sleep(self, *_):
            self.menu_sleep = not self.menu_sleep
            self.w.sleep(time.monotonic(), self.menu_sleep or getattr(self, "server_down", False))

        def set_harness(self, on=None, *_):
            on = (not _config(config.show_in_harness, False)) if on is None else bool(on)
            try:
                config.write("show_in_harness", on)
            except OSError as e:
                self.say(f"i couldn't save that ({e})", ttl=NOTICE_TTL)
                return
            msg = "i'll show up inside your harnesses" if on else "i'll stay out of your harnesses"
            if config.lookup("show_in_harness")[1] == "env":
                msg += f" (but `{config.ENV['show_in_harness']}` is set and wins in this shell)"
            self.say(msg, ttl=8)

        def choose_skin(self, spec):
            skin, why = wear(spec)
            if why:
                self.say(f"i can't wear {spec!r}: {why}", ttl=NOTICE_TTL)
                return
            self.set_skin(skin)
            try:
                config.write("pet_skin", spec)
            except OSError as e:
                self.say(f"i couldn't save my new look ({e})", ttl=NOTICE_TTL)

        def set_skin(self, skin):
            self.skin = skin
            self.skin_anim.update(row=None, since=time.monotonic())
            self.win.apply_mask(skin)
            self.win.look = None
            if self.tray is not None:
                self.tray.setIcon(self.icon())
            self.w.excite(time.monotonic(), 1.5)
            self.win.update()

        # -- updates: never installs anything, only says so
        def maybe_check_updates(self):
            due = update_due(bool(_config(config.update_check, True)), config.offline(),
                             _config(config.last_update_check, 0.0), time.time())
            if due:
                self.check_updates(manual=False)

        def check_updates(self, manual=False):
            if self.update_running:
                if manual:
                    self.say("i'm already looking", ttl=6)
                return
            if config.offline():
                if manual:
                    self.say("i'm offline (`HF_HUB_OFFLINE` / `MOTH_OFFLINE`), so i won't look for a new me", ttl=10)
                return
            self.update_running = True
            try:
                config.write(config.LAST_UPDATE_CHECK, round(time.time()))
            except OSError:
                pass
            if manual:
                self.say("looking for a new me…", ttl=6)

            def go():
                res, info = check_for_update()
                if not self.stop.is_set():
                    self.events.put(("update", (res, info, manual)))
            threading.Thread(target=go, daemon=True, name="moth-pet-update").start()

        def __del__(self):
            self.stop.set()

    pet = Pet()
    if boot.server:
        app.aboutToQuit.connect(boot.server.stop)   # menu quit, Ctrl-C, SIGTERM/SIGHUP, window closed
    quit_now.append(pet.quit)   # signals from here on quit through Qt
    if on_ready:
        on_ready(pet)
    try:
        return app.exec()
    finally:
        pet.stop.set()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="moth as a desktop pet (needs: pip install PySide6)")
    ap.add_argument("--size", type=int, default=64, help="moth width in px (32-160, default 64)")
    ap.add_argument("--no-wander", action="store_true", help="sit in the bottom-right corner instead of flying around")
    ap.add_argument("--url", default=DEFAULT_URL, help="moth server (default $MOTH_URL or http://127.0.0.1:4747)")
    ap.add_argument("--token-file", default=None, help="default $MOTH_TOKEN_FILE or ~/.moth/token")
    ap.add_argument("--skin", metavar="ID|FOLDER", default=None,
                    help="look like an installed Petdex pet (~/.petdex/pets/<id>, $CODEX_HOME/pets, <hermes home>/pets) "
                         "or a pet folder; 'moth' for moth. Default: pet_skin in ~/.moth/config.json (moth)")
    ap.add_argument("--export-petdex", metavar="DIR", default=None,
                    help="write the moth as a Petdex/Codex pet (DIR/pet.json + spritesheet.png) and exit")
    ap.add_argument("--no-server", action="store_true",
                    help="never start a moth server, just wait for one (by default the pet starts one when none "
                         "answers, and stops it when the pet quits)")
    ap.add_argument("--model", metavar="DIR", default=None,
                    help="weights for a server the pet starts (default: this install's, see moth/launch.py)")
    ap.add_argument("--server-arg", metavar="ARG", action="append", default=[],
                    help="one flag for a server the pet starts, repeatable: --server-arg=--react-every "
                         "--server-arg=0 (or put moth.serve flags after --:  -- --react-every 0)")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    """The pet's flags; everything after a bare -- goes to the server the pet starts (a.passthrough)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = build_parser().parse_args(argv)
    a.passthrough = rest
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_petdex:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # no window needed to draw a sheet
        try:
            paths = export_petdex(a.export_petdex)
        except ImportError:
            print("the export needs PySide6:  pip install PySide6", file=sys.stderr)
            return 2
        print("\n".join(paths))
        print("use it: copy the folder to $CODEX_HOME/pets/moth (Codex: /pets), <hermes home>/pets/moth "
              "(hermes pets select moth) or ~/.petdex/pets/moth (Petdex desktop)")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
