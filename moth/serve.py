"""Chat with moth in the browser:  python -m moth.serve   ->  http://127.0.0.1:4747   (or just: python -m moth)

One conversation at a time. moth's home (~/.moth/home in the release layout, live/home/ here) survives restarts,
so what it writes stays. Its bash tool is moth/shell.py: a tiny pure-Python shell that knows only the commands
moth was trained on (ls, cat, head, wc, echo), stays inside the home and answers exactly like the bash it trained with.
No subprocess, no sandbox needed, so it runs the same on Linux, macOS and Windows.
The KV cache is kept across turns; when it fills up, the oldest turns are dropped and the rest is re-read.
Runs on CPU by default (112M params: fast enough).

Local server hardening: only Host 127.0.0.1/localhost:<port> (DNS rebinding), only local pages over CORS, and a
per-install token (~/.moth/token, made on first run) for everything that reads or drives the chat. The page moth
serves gets the token in its HTML; adapters and displays read the file. GET /api/says is the token-free view
(moth's own lines only) for desktop pets that can't read files.

Several coding-agent sessions at once (any mix of harnesses): each adapter sends a session id and a label (the project
folder), shown on the cards as "claude · moth-repo said". Reactions are collected over a short window and moth reacts
to the most interesting message (Moth.hear_agent), and its history keeps one agent turn per session, so it doesn't
mix them up (Moth._one_turn_per_session). The model's input stays exactly agent_turn(text): no session names.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import random
import re
import secrets
import shutil
import socketserver
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import torch
from tokenizers import Tokenizer

TA = os.environ.get("TA_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TA)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_agent.chat import parse_assistant, render  # noqa: E402
from tiny_agent.checkpoint import load_model  # noqa: E402
from tiny_agent.generate import KVCache, append  # noqa: E402
from tiny_agent.tools import BASH_TIMEOUT_S, Workspace, truncate  # noqa: E402
from moth.agent import AGENT_TAG, agent_turn  # noqa: E402
from moth.home import load_pool, sample_home  # noqa: E402
from moth.norepeat import PhraseMemory, vocab_classes  # noqa: E402
from moth.persona import SYSTEM_STUDENT  # noqa: E402
from moth.shell import MARKER, Shell, _windows_unsafe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LEN = 4096        # KV cache size (hard limit)
CTX_BUDGET = 3000     # trained on chats + stitched sessions up to 4k tokens (quality flat to ~4k, tests/ctx_probe.py)
CTX_KEEP = 1800       # when over budget, keep the system prompt + the newest turns that fit in this
REPLY_ROOM = 200      # tokens kept free for moth's next reply
CALL_ROOM = 60        # ... and for each later tool call in the same turn
COMMON = ("i i'm im the a an and or to of it it's its is in on at my me you your we be was are not no yes "
          "just so that this but like with for do don't can can't oh . , ! ? ' \" - : ... s t m")
FALLBACK = "oh. my head went all fuzzy for a second. what were we saying?"
SPECIALS = ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>",
            "<tool_response>", "</tool_response>", "<|endoftext|>", "<|pad|>")   # + whatever the tokenizer adds

# Pages allowed to read moth over CORS. Full log, chat, agent, reset: local dev pages only (http(s)://localhost or
# 127.0.0.1, any port). /api/says (moth's own lines, nothing else) also: packaged desktop shells (Hermes Desktop
# = file:// page = Origin "null"; app://). No wildcard anywhere.
LOCAL_ORIGIN = re.compile(r"https?://(localhost|127\.0\.0\.1)(:\d+)?")
PET_ORIGIN = re.compile(r"app://[^/\s]*|file://|null")
TOKEN_HEADER = "X-Moth-Token"
MAX_BODY = 64 * 1024
LOG_SIZE = 500

# Reacting to the coding agent: at most once per REACT_EVERY s, unless it sounds bad or is very long; never while
# the human is mid-vent (chatted in the last QUIET_AFTER_CHAT s): the reaction waits.
REACT_EVERY = 120.0
QUIET_AFTER_CHAT = 60.0
LONG_AGENT_WORDS = 250
URGENT = re.compile(r"\b(fail\w*|errors?|couldn'?t|could not|can'?t|cannot|sorry|apologi\w+|broke\w*|crash\w*)\b",
                    re.I)
COMPACT_WORDS = 25    # what moth keeps of an agent message in its history once it has reacted
# Several agent sessions at once: when moth may react, it waits BATCH_WINDOW s for more finished turns, then reacts
# to the most interesting one (interest()) and shows the others as skipped cards.
BATCH_WINDOW = 4.0
QUESTION = re.compile(r"\?|\b(should i|shall i|do you want|would you like|want me to|let me know|which one|"
                      r"which option|your call|okay to|ok to)\b", re.I)
ACTIVE_AFTER = 30 * 60.0   # a session that posted in the last 30 min counts as active (the page's status line)
SOURCE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")   # "claude-code", "codex", "hermes", "pi", "opencode", "dsh"
SESSION_JUNK = re.compile(r"[^A-Za-z0-9._:-]")   # session ids: uuids, "ses_...", "20261018_120000_ab12"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_ansi(text: str) -> str:
    """Terminal colour/cursor codes and other control characters out; newlines and tabs stay."""
    return ANSI.sub("", text)


def clean_session(v) -> str | None:
    """An adapter's session id: a short id charset, at most 80 chars, else None."""
    if not isinstance(v, str):
        return None
    return SESSION_JUNK.sub("", v)[:80] or None


def clean_label(v) -> str | None:
    """An adapter's session label (the project folder's name): one plain line of at most 40 chars, else None."""
    if not isinstance(v, str):
        return None
    return " ".join(strip_ansi(v).split())[:40] or None


def interest(text: str) -> int:
    """Which of several agent turns moth reacts to: failure/apology/error 3 > a question to the user 2 > long 1 >
    routine "done" 0. The question is looked for at the end, where agents ask it."""
    if URGENT.search(text):
        return 3
    words = text.split()
    if QUESTION.search(" ".join(words[-60:])):
        return 2
    return 1 if len(words) > LONG_AGENT_WORDS else 0


def default_paths() -> dict:
    """Weights next to code/ (the Hugging Face repo layout) or this project's runs/ and data/."""
    root = os.path.dirname(HERE)
    repo = os.path.dirname(root)
    from moth import weights   # stdlib only
    if weights.mode() == "hf":   # also a download still missing its weights: main() says how to finish it
        return {"model": repo, "tokenizer": os.path.join(repo, "tokenizer.json"),
                "home": os.path.join(os.path.expanduser("~"), ".moth", "home"),
                "pool": os.path.join(HERE, "file_pool.json")}
    if weights.mode() == "pip":   # pip install: weights in ~/.cache/moth/<repo>/, fetched in main() (not on --help)
        cache = weights.cache_dir()
        return {"model": cache, "tokenizer": os.path.join(cache, "tokenizer.json"),
                "home": os.path.join(os.path.expanduser("~"), ".moth", "home"),
                "pool": os.path.join(HERE, "file_pool.json")}
    return {"model": os.path.join(root, "runs/sft1/model"),
            "tokenizer": os.path.join(root, "data/tiny-agent-112m/tokenizer.json"),
            "home": os.path.join(root, "live/home"),
            "pool": os.path.join(HERE, "file_pool.json")}


INSTALL_ROOT = os.path.dirname(os.path.dirname(HERE))   # the download folder (moth-112m/), for moth update


def version() -> str:
    try:
        with open(os.path.join(HERE, "VERSION")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def pidfile_path(port: int) -> str:
    return os.path.join(os.path.expanduser("~"), ".moth", f"serve-{port}.pid")


def write_pidfile(port: int) -> str | None:
    """~/.moth/serve-<port>.pid, so `python -m moth update` can tell a moth from this install is running."""
    path = pidfile_path(port)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"pid": os.getpid(), "port": port, "install_root": INSTALL_ROOT}, f)
        return path
    except OSError:
        return None   # moth runs fine without it; update then just can't see this server


def remove_pidfile(path: str | None):
    try:
        with open(path) as f:
            mine = json.load(f).get("pid") == os.getpid()   # not one a newer moth on this port wrote
        if mine:
            os.remove(path)   # after the file is closed: Windows can't delete an open file
    except (OSError, TypeError, ValueError, AttributeError):
        pass


def token_path() -> str:
    return os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")


def load_token(path: str | None = None) -> str:
    """This install's secret, made on first run (owner-only file). Adapters send it as X-Moth-Token."""
    path = path or token_path()
    try:
        with open(path) as f:
            tok = f.read().strip()
        if tok:
            return tok
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    return tok


def _opt(a, name: str, default):
    """An option eval.py's namespace may not have."""
    v = getattr(a, name, None)
    return default if v is None else v


def _is_marker_path(path: str) -> bool:
    """Some part of path is moth's home marker as a case-insensitive file system or Windows would see it."""
    return any(p.lower().rstrip(". ") == MARKER for p in path.replace("\\", "/").split("/"))


class Home(Workspace):
    """A Workspace rooted at a fixed, persistent directory. bash goes to moth.shell, read/write as in training."""

    def __init__(self, root: str, seed: int, pool_file: str | None):
        self.files, self.read_paths, self.submitted = {}, set(), None
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
            # "new home" only ever deletes folders with this
            with open(os.path.join(self.root, MARKER), "w", encoding="utf-8", newline="\n") as f:
                f.write("moth made this folder. its \"new home\" button may delete it.\n")
            for rel, content in sample_home(random.Random(seed), load_pool(pool_file)).items():
                p = os.path.join(self.root, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
        self.shell = Shell(self.root)

    @property
    def owned(self) -> bool:
        """moth created this folder itself (so it may delete it)."""
        return os.path.isfile(os.path.join(self.root, MARKER)) and not os.path.islink(os.path.join(self.root, MARKER))

    def close(self):
        pass

    def _resolve(self, path: str) -> str | None:
        """As Workspace's, but moth's marker is off limits: a write must never make a folder look like moth's. On
        case-insensitive file systems ".MOTH-HOME" is the marker, and on Windows ".moth-home." / ".moth-home " too,
        so any path part that folds to it is refused, on every OS (and Windows-only path tricks, as in moth.shell)."""
        if isinstance(path, str) and (_is_marker_path(path) or (os.name == "nt" and _windows_unsafe(path))):
            return None
        full = super()._resolve(path)
        if full is None or _is_marker_path(os.path.relpath(full, self.root).replace(os.sep, "/")):
            return None
        return full

    def tool_bash(self, command: str, timeout: int = BASH_TIMEOUT_S) -> str:
        return self.shell.run(str(command))

    def tree(self) -> list[dict]:
        out = []
        for d in sorted(os.listdir(self.root)):
            full = os.path.join(self.root, d)
            if os.path.isdir(full):
                out.append({"dir": d, "files": sorted(f for f in os.listdir(full) if not f.startswith("."))})
        return out


class Log:
    """Everything shown in the chat window, polled by displays: the last `size` events, with absolute indexes.
    The epoch changes on every restart and reset, so a display knows to start over instead of guessing."""

    def __init__(self, size: int = LOG_SIZE):
        self.lock = threading.Lock()
        self.size = size
        self.clear()

    def clear(self):
        self.events: deque = deque(maxlen=self.size)
        self.start = 0                       # absolute index of events[0]
        self.epoch = secrets.token_hex(4)

    def add(self, *evs: dict):
        with self.lock:
            for ev in evs:
                if len(self.events) == self.size:
                    self.start += 1
                self.events.append(ev)

    def since(self, n: int, epoch: str | None = None, says_only: bool = False) -> dict:
        """Events from absolute index n (clamped to what is kept). A different epoch: everything kept."""
        with self.lock:
            end = self.start + len(self.events)
            if epoch is not None and epoch != self.epoch:
                n = self.start
            n = min(max(n, self.start), end)
            evs = list(self.events)[n - self.start:]
            if says_only:
                evs = [{"say": e["say"]} for e in evs if "say" in e]
            return {"events": evs, "next": end, "epoch": self.epoch}

    def __len__(self):
        return self.start + len(self.events)


class Moth:
    def __init__(self, a):
        self.tok = Tokenizer.from_file(a.tokenizer)
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        # CPU: Engram tables stay bf16 and memory-mapped (~1 GB less RAM, identical outputs)
        self.model = load_model(a.model, device=a.device, dtype=dtype, low_mem=a.device == "cpu").eval()
        quant = self.quant = resolve_quant(a)
        if quant != "none":   # CPU speed modes (CPU_QUANT is the default there): slightly different outputs than fp32
            if a.device != "cpu":
                raise SystemExit("--quant is for --device cpu")
            from tiny_agent.generate import quantize_cpu
            quantize_cpu(self.model, quant)
        self.kv_dtype = dtype
        # one KV cache for the whole session, reset for every re-read: the decoder's captured GPU graph is tied to it
        self._kv = KVCache(self.model, 1, MAX_LEN, a.device, dtype=dtype)
        self.setup(a)
        if a.device != "cpu":   # capture the GPU graphs (decode + short-chunk feeds) now, not on the first turns
            self.cache = self._kv
            self._decoder().warmup(a.max_turn_tokens, self.im_end, a.temperature, a.top_p,
                                   getattr(a, "rep_penalty", 1.0))
            self.cache = None

    def setup(self, a):
        """Everything but the weights (tests build a Moth with a fake model and call this)."""
        self.a = a
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.ban = [self.tok.token_to_id(t) for t in ("<tool_call>", "<think>")]   # for turns where moth must talk
        self.lock = threading.RLock()
        self.home = Home(a.home, a.seed, a.pool_file)
        self.log = Log()
        self.budget = getattr(a, "ctx_budget", None) or CTX_BUDGET
        self.keep = self.budget * CTX_KEEP // CTX_BUDGET
        self.common = {i for w in COMMON.split() for v in (w, " " + w) for i in self.enc(v)}
        # moth's last reactions, for the phrase ban (moth/norepeat.py); kept across reset() and "new chat" on purpose,
        # wiped by forget(everything=True) (moth forget) and by stopping moth
        self.phrases = PhraseMemory(_opt(a, "phrase_memory", 16), _opt(a, "phrase_n", 3))
        self.ban_chat = bool(getattr(a, "phrase_ban_chat", False))
        self.word_ids, self.boring_ids = vocab_classes(self.tok, self.common)
        self._hook = None     # this turn's logits_hook (set in _chat)
        self._carry = set()   # this turn's extra repetition-penalty ids: the last reactions' (set in _chat)
        try:
            added = [t.content for t in self.tok.get_added_tokens_decoder().values()]
        except AttributeError:   # older tokenizers
            added = []
        specials = sorted(set(SPECIALS) | set(added), key=len, reverse=True)
        self.specials = re.compile("|".join(re.escape(s) for s in specials))
        self.react_every = _opt(a, "react_every", REACT_EVERY)
        self.quiet_after = _opt(a, "quiet_after", QUIET_AFTER_CHAT)
        self.clock = time.monotonic
        self.batch_window = _opt(a, "batch_window", BATCH_WINDOW)
        self.last_react = self.last_user = float("-inf")
        self.pending = None           # the agent message (an item(), see hear_agent) moth will react to next
        self.batch, self.batch_until = None, 0.0   # agent messages collected in the open batch window, and its end
        self.sessions: dict[str, dict] = {}         # session key -> {"source", "session", "label", "last", "skips"}
        self.wake = threading.Condition(threading.RLock())
        self.reset()

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def defuse(self, text: str) -> str:
        """Literal special-token strings in outside text ("<|im_end|>" in a file or a message) would be read as the
        real tokens and could end a turn or fake a tool call: "<|im_end|>" -> "< |im_end|>"."""
        return self.specials.sub(lambda m: "< " + m.group(0)[1:], text)

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_STUDENT}]
        self.cache = None
        self.said = deque(maxlen=3)    # token ids of moth's last replies (repetition penalty)
        self.protect: set[int] = set()  # ids from this turn's user message and tool results: free to copy

    def forget(self, everything: bool = False):
        """New chat (the page's button / POST /api/reset): history, log (new epoch), any pending reaction or open
        batch, and the list of active agent sessions. Kept on purpose: self.phrases, the token ids of moth's own last
        reactions (RAM only, never shown anywhere), so a new chat doesn't bring back the same joke.
        everything=True (POST /api/reset {"everything": true}, what `moth forget` sends a running moth): those too,
        so nothing moth remembers about this session is left in RAM. Stopping moth wipes all of it as well."""
        with self.lock:
            self.reset()
            self.log.clear()
            # what the last turn left behind (rebuilt at the next turn anyway): the phrase-ban hook (reaction n-grams,
            # words of user messages), the reactions' penalty ids, and the decoder that still holds the old KV cache
            self._hook, self._carry, self._dec = None, set(), None
            if everything:
                self.phrases.clear()
            with self.wake:
                self.pending, self.batch = None, None
                self.sessions.clear()

    # --- the model ------------------------------------------------------------------------------------------
    def _prefill_all(self):
        """(Re)build the cache from self.messages, dropping the oldest turns if they don't fit."""
        while True:
            ids = self.enc(render(self.messages, add_generation_prompt=True))
            if len(ids) < self.keep or len(self.messages) <= 2:
                break
            # drop the oldest user turn and everything up to the next user turn
            i = next((k for k in range(2, len(self.messages)) if self.messages[k]["role"] == "user"), None)
            if i is None:   # only the current turn is left: keep it even if it's long
                break
            self.messages = self.messages[:1] + self.messages[i:]
        self.cache = self._new_cache()
        logits = self._feed(ids)
        if logits is None:   # the current turn alone is over budget (tool results are cut to prevent this)
            raise RuntimeError(f"turn too long for moth's context ({len(ids)} tokens)")
        return logits

    def _new_cache(self):
        kv = getattr(self, "_kv", None)
        if kv is None:        # tests build a Moth without weights
            return KVCache(self.model, 1, MAX_LEN, self.a.device, dtype=next(self.model.parameters()).dtype)
        kv.reset([0])         # same as a fresh cache: everything past the length is masked
        return kv

    def _feed(self, ids):
        """Append ids to the cache in one pass (tiny_agent.generate.append: the whole prompt in one forward)."""
        if int(self.cache.lens_cpu[0]) + len(ids) + REPLY_ROOM > self.budget:
            return None   # over budget: the caller re-reads a trimmed conversation
        x = torch.tensor([ids], device=self.a.device)
        return append(self.model, self.cache, x, [len(ids)])

    def _pen_ids(self) -> set[int]:
        """Token ids of moth's last replies that the repetition penalty hits this turn (not common words, not ids from
        this turn's user message or tool results): fixed for a whole turn."""
        if getattr(self.a, "rep_penalty", 1.0) == 1.0 or not (self.said or self._carry):
            return set()
        return ({i for r in self.said for i in r} | self._carry) - self.common - self.protect

    def _sample(self, logits, talk_only=False):
        """One token from logits (1, V): ban (talk_only), repetition penalty, temperature, top-p. On the logits'
        device, vectorised (masks, no per-id indexing); the same values and draws as the old per-token code."""
        from tiny_agent.generate import sample_top_p
        V = logits.size(-1)
        pm = torch.zeros(V, dtype=torch.bool)
        ids = self._pen_ids()
        if ids:
            pm[list(ids)] = True
        bm = torch.zeros(V, dtype=torch.bool)
        if talk_only:
            bm[self.ban] = True
        return int(sample_top_p(logits, self.a.temperature, self.a.top_p, getattr(self.a, "rep_penalty", 1.0),
                                pm.to(logits.device) if ids else None, bm.to(logits.device) if talk_only else None))

    def _decoder(self):
        from tiny_agent.generate import Decoder
        dec = getattr(self, "_dec", None)
        if dec is None or dec.cache is not self.cache:
            graph = False if getattr(self.a, "no_graph", False) else None   # None: device graphs on CUDA/XPU
            dec = self._dec = Decoder(self.model, self.cache, graph=graph, compile=getattr(self.a, "compile", False))
        return dec

    def _turn(self, logits, talk_only=False) -> str:
        """Sample moth's turn until <|im_end|> (appended to the cache too) or max_turn_tokens (then <|im_end|> is
        appended). CPU: an eager per-token loop, token-for-token the same as the old one for a given seed; CUDA/XPU:
        one captured graph per token, sampling on the device, tokens read back every few steps (generate.Decoder)."""
        out = self._decoder().generate(
            logits, self.a.max_turn_tokens, self.im_end, temperature=self.a.temperature, top_p=self.a.top_p,
            pen=getattr(self.a, "rep_penalty", 1.0), pen_ids=self._pen_ids(), ban_ids=self.ban if talk_only else (),
            logits_hook=self._hook)
        return self.tok.decode(out, skip_special_tokens=False)

    def _cut(self, text: str, max_tokens: int) -> str:
        """Shorten one tool result to max_tokens, with the truncation marker moth saw in training."""
        ids = self.enc(text)
        if len(ids) <= max_tokens:
            return text
        keep = int(len(text) * max_tokens / len(ids)) - 80   # room for the marker line
        text = truncate(text, keep) if keep >= 30 else "[... truncated, too long ...]"
        ids = self.enc(text)
        return text if len(ids) <= max_tokens else self.tok.decode(ids[:max(1, max_tokens)])

    def _fit_results(self, results: list[str], turn_msg: dict, rounds_left: int) -> list[str]:
        """Cut tool results so the current turn fits the context budget even after every older turn is dropped:
        a turn that can't be trimmed must never overflow (that used to leave the cache empty and crash).
        Each round's results get at most half of what is left, after room for the later rounds' calls."""
        start = next((i for i, m in enumerate(self.messages) if m is turn_msg), 1)
        fixed = len(self.enc(render(self.messages[:1] + self.messages[start:], add_generation_prompt=True)))
        room = self.budget - REPLY_ROOM - fixed - 16 * len(results) - 8 - CALL_ROOM * rounds_left
        per = max(8, room // (2 * max(1, len(results))) if rounds_left else room // max(1, len(results)))
        return [self._cut(r, per) for r in results]

    # --- one turn -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, user: str, talk_only: bool = False) -> list[dict]:
        """Returns the events of moth's turn: {"tool": name, "args":…, "result":…} and {"say": text}.
        talk_only: no tools in any round (reactions to the coding agent). Any failure rolls the turn back."""
        with self.lock:
            snap = (list(self.messages), list(self.said), set(self.protect))
            try:
                return self._chat(user, talk_only)
            except Exception as e:  # noqa: BLE001  one bad turn must never wedge the chat
                self.messages, self.said, self.protect = snap[0], deque(snap[1], maxlen=3), snap[2]
                self.cache = None   # appends mutate the cache in place: rebuild it from the restored messages
                print(f"moth: turn rolled back ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                return [{"say": FALLBACK}]

    def _chat(self, user: str, talk_only: bool = False) -> list[dict]:
        user = self.defuse(user)
        msg = {"role": "user", "content": user}
        self.protect = set(self.enc(" " + user)) | set(self.enc(user))
        # the phrase ban: reactions always; chat turns only with --phrase-ban-chat, sparing every user message's words
        quote = self.enc(" " + user) if talk_only else self.enc(" ".join(
            [m["content"] for m in self.messages if m["role"] == "user"] + [user]))
        self._hook = self.phrases.hook(self.word_ids, self.boring_ids, quote) if talk_only or self.ban_chat else None
        self._carry = self.phrases.recent() if talk_only else set()   # reactions: the soft penalty spans "new chat"
        logits = None
        if self.cache is not None:
            logits = self._feed(self.enc("\n" + render([msg], add_generation_prompt=True)))
        self.messages.append(msg)
        if logits is None:
            logits = self._prefill_all()
        events, seen_calls = [], set()
        for rnd in range(self.a.max_tool_rounds + 1):
            last = talk_only or rnd == self.a.max_tool_rounds      # this turn must talk
            m = parse_assistant(self._turn(logits, talk_only=last))
            if last:
                m["tool_calls"] = []   # banned tokens can still be spelled out in plain text
            keys = {json.dumps(c, sort_keys=True) for c in m["tool_calls"]}
            cut = "<tool_call>" in (m["content"] or "")   # a call cut off by the token limit
            if (m["tool_calls"] and keys <= seen_calls) or cut:
                # the same call again (a loop) or a broken call: rewind this turn and make it talk instead
                self.cache = None
                logits = self._prefill_all()
                m = parse_assistant(self._turn(logits, talk_only=True))
                m["tool_calls"], keys = [], set()
            m["content"] = (m["content"] or "").lower()   # moth only speaks lowercase (the odd "I" slips through)
            self.messages.append(m)
            if not m["tool_calls"]:
                events.append({"say": m["content"] or "..."})
                self.said.append(self.enc(" " + (m["content"] or "")))
                break
            if m["content"]:
                events.append({"say": m["content"]})
            seen_calls |= keys
            results = []
            for c in m["tool_calls"]:
                r = c["error"] if "error" in c else self.home.call(c["name"], c["arguments"])
                results.append(r)
                events.append({"tool": c.get("name", "?"), "args": c.get("arguments", {}), "result": r})
            results = self._fit_results([self.defuse(r) for r in results], msg, self.a.max_tool_rounds - rnd - 1)
            for r in results:
                self.protect |= set(self.enc(" " + r))
            tool_msg = {"role": "tool", "results": results}
            self.messages.append(tool_msg)
            logits = self._feed(self.enc("\n" + render([tool_msg], add_generation_prompt=True)))
            if logits is None:
                logits = self._prefill_all()
        return events

    # --- the human ------------------------------------------------------------------------------------------
    def say_to(self, text: str) -> list[dict]:
        """The human talks to moth."""
        self.last_user = self.clock()
        with self.lock:
            self.log.add({"you": text})
            evs = self.chat(text)
            self.log.add(*evs)
        self.last_user = self.clock()   # the quiet window starts after moth answered
        with self.wake:
            self.wake.notify_all()
        return evs

    # --- the coding agent -----------------------------------------------------------------------------------
    @staticmethod
    def card(text: str, source: str | None = None, **extra) -> dict:
        ev = {"agent": agent_turn(strip_ansi(text))[len(AGENT_TAG):].strip()}
        if source:
            ev["source"] = source
        ev.update({k: v for k, v in extra.items() if v is not None})
        return ev

    @staticmethod
    def item(text: str, source: str | None = None, session: str | None = None, label: str | None = None) -> dict:
        """One agent message as hear_agent, the batch and the reactor carry it. key = the session it came from; an
        adapter that sends no session id counts as one session per harness."""
        return {"text": text, "source": source, "session": session, "label": label,
                "key": f"{source or 'agent'}:{session}" if session else f"src:{source or 'agent'}",
                "score": interest(text),
                "urgent": bool(URGENT.search(text)) or len(text.split()) > LONG_AGENT_WORDS}

    def _seen(self, it: dict):
        """Mark the message's session active (under self.wake). No label: a short session id. A label another active
        session already shows (two agents in one folder) gets the end of the session id after it."""
        now = self.clock()
        for k in [k for k, s in self.sessions.items() if now - s["last"] >= ACTIVE_AFTER]:
            del self.sessions[k]
        label, sid = it["label"], it["session"]
        if not label and sid:
            label = sid[:8]
        if label and sid and any(k != it["key"] and s["label"] == label for k, s in self.sessions.items()):
            label = f"{label} ({sid[-4:]})"
        s = self.sessions.setdefault(it["key"], {"source": it["source"], "session": sid, "skips": deque(maxlen=200)})
        s["label"] = it["label"] = label
        s["last"] = now

    def _log_card(self, it: dict, skipped: bool):
        self.log.add(self.card(it["text"], it["source"], session=it["session"], label=it["label"],
                               skipped=True if skipped else None))
        if skipped and it["key"] in self.sessions:
            self.sessions[it["key"]]["skips"].append(self.clock())

    def agents(self) -> dict:
        """The page's status line and session filter: the sessions that posted in the last ACTIVE_AFTER s, and how
        many messages moth let pass in that time. Only in the token-guarded /api/log (labels are folder names)."""
        with self.wake:
            now = self.clock()
            act = [s for s in self.sessions.values() if now - s["last"] < ACTIVE_AFTER]
            return {"active": len(act), "skipped": sum(now - t < ACTIVE_AFTER for s in act for t in s["skips"]),
                    "sessions": [{"source": s["source"], "session": s["session"], "label": s["label"]} for s in act]}

    def agent_said(self, text: str, source: str | None = None, session: str | None = None,
                   label: str | None = None) -> list[dict]:
        """The big agent (Claude Code, Hermes, ...) finished a reply: moth sees it and reacts now (no throttle)."""
        it = self.item(text, source, session, label)
        with self.wake:
            self._seen(it)
            self._log_card(it, skipped=False)
        return self.react(text, item=it)

    def react(self, text: str, hold_for_human: bool = False, item: dict | None = None) -> list[dict]:
        """moth's talk-only reaction to an agent message; afterwards only ~25 words of it stay in moth's history,
        so agent turns don't push the human's conversation out of context. (The cache still holds the full text
        until the next re-read; trimming decides on the short version.) The model sees exactly agent_turn(text):
        session names never go into moth's text. item: the message's session (see _one_turn_per_session)."""
        msg = agent_turn(strip_ansi(text))
        key = item["key"] if item else None
        with self.lock:
            if hold_for_human and self.clock() - self.last_user < self.quiet_after:   # the human got in first
                with self.wake:
                    if self.pending is None:
                        self.pending = item or self.item(text)
                return []
            self.last_react = self.clock()
            self._one_turn_per_session(key)
            evs = self.chat(msg, talk_only=True)
            for e in evs:   # the phrase ban's memory (not the fallback line: that one isn't moth's own)
                if e.get("say") and e["say"] != FALLBACK:
                    self.phrases.add(self.enc(e["say"]))
            want = self.defuse(msg)
            for i in range(len(self.messages) - 1, 0, -1):
                m = self.messages[i]
                if m["role"] == "user":
                    if m["content"] == want:
                        words = want[len(AGENT_TAG):].split()
                        short = " ".join(words[:COMPACT_WORDS]) + (" ..." if len(words) > COMPACT_WORDS else "")
                        self.messages[i] = {**m, "content": f"{AGENT_TAG}\n{short}", "session": key}
                    break
            # the session this reaction answers, for /api/says?source=&session= (the token-free view strips it)
            tag = {k: item[k] for k in ("source", "session") if item and item.get(k)}
            self.log.add(*({**e, "re": tag} if tag and "say" in e else e for e in evs))
        return evs

    def _one_turn_per_session(self, key: str | None):
        """Sessions don't mix: before moth reacts to a message from session `key`, its history keeps only the newest
        earlier agent turn of every OTHER session (already cut to COMPACT_WORDS) and none of `key`'s own (the new
        message, in full, replaces it). Agent turns carry a "session" field in self.messages, the marker; render()
        never shows it to the model. A dropped agent turn goes together with moth's reply to it, so the chat stays
        the user/assistant turns moth trained on. When anything was dropped the cache is re-read from the trimmed
        history, so the dropped text is really gone."""
        seen, drop = {key}, set()
        for i in range(len(self.messages) - 1, 0, -1):
            m = self.messages[i]
            if m["role"] == "user" and "session" in m:
                if m["session"] in seen:
                    drop.add(i)
                seen.add(m["session"])
        if not drop:
            return
        kept, dropping = [], False
        for i, m in enumerate(self.messages):
            if m["role"] == "user":
                dropping = i in drop
            if not dropping:
                kept.append(m)
        self.messages, self.cache = kept, None

    def hear_agent(self, text: str, source: str | None = None, session: str | None = None,
                   label: str | None = None) -> str:
        """POST /api/agent: react later on the reactor thread, at most once per react_every s unless it sounds bad
        (sorry, failed, error...) or is very long, and only once the human has been quiet for quiet_after s.

        Several agent sessions at once: a message moth may react to opens a batch_window s window, and every message
        that arrives in it joins. When the window closes, moth takes the most interesting one (interest(): failure >
        question > long > routine, the newest on a tie) and the rest are skipped. Batched cards are logged when the
        window closes, with their final skipped flag (displays draw each event once), the chosen one last so moth's
        reply sits right under it. A message already waiting for the human to go quiet stays unless the batch has
        one at least as interesting. batch_window 0 or react_every 0: no window, the card is logged at once and
        (react_every 0) every message is queued (the judge tooling's "react to everything right away").
        Returns "queued", "batched" (in a window: decided when it closes) or "skipped"."""
        it = self.item(text, source, session, label)
        with self.wake:
            self._seen(it)
            if self.batch is not None:
                self.batch.append(it)
                return "batched"
            p = self.pending
            if p and p["urgent"] and not it["urgent"]:
                status = "skipped"
            elif not it["urgent"] and not p and self.clock() - self.last_react < self.react_every:
                status = "skipped"
            elif self.batch_window > 0 and self.react_every > 0:
                self.batch, self.batch_until = [it], self.clock() + self.batch_window
                self.wake.notify_all()
                return "batched"
            else:
                status = "queued"
            self._log_card(it, skipped=status == "skipped")
            if status == "queued":   # after the card is logged, so moth's reply always comes after it
                self.pending = it
                self.wake.notify_all()
        return status

    def _close_batch(self):
        """The window is over (under self.wake): log the cards, the winner last, and queue it."""
        batch, self.batch = self.batch, None
        win = max(reversed(batch), key=lambda it: it["score"])   # max keeps the first best: reversed = the newest
        if self.pending is not None and self.pending["score"] > win["score"]:
            win = None   # the message already waiting is more interesting: the whole batch passes
        for it in batch:
            if it is not win:
                self._log_card(it, skipped=True)
        if win is not None:
            self._log_card(win, skipped=False)
            self.pending = win

    def due(self) -> tuple[dict | None, float]:
        """(the agent message (an item()) to react to now, or None; seconds to wait before asking again)."""
        with self.wake:
            if self.batch is not None:
                wait = self.batch_until - self.clock()
                if wait > 0:
                    return None, wait
                self._close_batch()
            if self.pending is None:
                return None, 3600.0
            wait = self.last_user + self.quiet_after - self.clock()
            if wait > 0:
                return None, wait
            it, self.pending = self.pending, None
            return it, 0.0

    def start_reactor(self):
        def loop():
            while True:
                with self.wake:   # check and wait under one hold of the condition: no lost wakeups
                    it, wait = self.due()
                    if it is None:
                        self.wake.wait(min(wait, 3600.0))
                        continue
                try:
                    self.react(it["text"], hold_for_human=True, item=it)
                except Exception as e:  # noqa: BLE001  keep the reactor alive
                    print(f"moth: reaction failed ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        threading.Thread(target=loop, daemon=True, name="moth-reactor").start()


# --- http ---------------------------------------------------------------------------------------------------
class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer minus the reverse-DNS lookup HTTPServer.server_bind does (socket.getfqdn of the bind
    address, used only for CGI's SERVER_NAME). Where reverse DNS is slow that lookup sits between bind() and listen(),
    so the port refuses connections for that long (35 s on GitHub's macOS runners) and the pet gives up waiting."""
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]


def make_handler(moth: Moth, token: str, hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]")):
    page = os.path.join(HERE, "static", "index.html")

    class H(BaseHTTPRequestHandler):
        server_version = "moth"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, cors: bool = True):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                if cors:
                    self._cors()
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass   # agent adapters hang up after 1 s on purpose; moth still keeps the message

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _cors(self) -> bool:
            """Reflect the Origin of a local page; desktop-shell origins only for /api/says; others get nothing."""
            origin = self.headers.get("Origin", "")
            pet = urlsplit(self.path).path == "/api/says"
            if not (LOCAL_ORIGIN.fullmatch(origin) or (pet and PET_ORIGIN.fullmatch(origin))):
                return False
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            return True

        def _host_ok(self) -> bool:
            """DNS rebinding: a page on evil.example resolving to 127.0.0.1 still sends Host: evil.example."""
            port = self.server.server_address[1]
            return (self.headers.get("Host") or "").lower() in {f"{h}:{port}" for h in hosts}

        def _authed(self) -> bool:
            return hmac.compare_digest(self.headers.get(TOKEN_HEADER, "").encode(), token.encode())

        def do_OPTIONS(self):   # CORS preflight for a JSON POST / a token header from a local page
            if not self._host_ok():
                return self._json({"error": "bad host"}, 403)
            self.send_response(204)
            if self._cors():
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", f"Content-Type, {TOKEN_HEADER}")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if not self._host_ok():
                return self._json({"error": "bad host"}, 403)
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if url.path == "/api/health":   # no token (Host still checked): which install runs here, for moth update
                health = {"ok": True, "pid": os.getpid(), "install_root": INSTALL_ROOT, "version": version()}
                return self._send(200, json.dumps(health).encode(), "application/json", cors=False)
            if url.path == "/":
                with open(page, encoding="utf-8") as f:
                    html = f.read().replace("__MOTH_TOKEN__", token)
                return self._send(200, html.encode(), "text/html; charset=utf-8", cors=False)   # holds the token
            if url.path not in ("/api/says", "/api/log", "/api/home"):
                return self._json({"error": "not found"}, 404)
            if url.path != "/api/says" and not self._authed():
                return self._json({"error": "missing or wrong token (see ~/.moth/token)"}, 401)
            if url.path == "/api/home":
                return self._json({"tree": moth.home.tree()})
            since = (q.get("since") or ["0"])[0]
            if not re.fullmatch(r"\d{1,18}", since):
                return self._json({"error": "since must be a number"}, 400)
            epoch = (q.get("epoch") or [None])[0]
            # /api/says?source=&session=: only moth's reactions to that harness / session (an in-harness display's
            # own), never chat or another session's; filtered on the tagged log, then stripped to {"say"}
            want = {}   # cleaned the way /api/agent cleans them; an id that cleans to nothing matches nothing
            if q.get("source"):
                want["source"] = q["source"][0].lower()
            if q.get("session"):
                want["session"] = clean_session(q["session"][0]) or "-"
            if url.path != "/api/says" or not want:
                d = moth.log.since(int(since), epoch, says_only=url.path == "/api/says")
            else:
                d = moth.log.since(int(since), epoch)
                d["events"] = [{"say": e["say"]} for e in d["events"] if "say" in e
                               and all((e.get("re") or {}).get(k) == v for k, v in want.items())]
            if url.path == "/api/log":
                d["tree"] = moth.home.tree()
                d["agents"] = moth.agents()   # session labels are folder names: never in the token-free /api/says
            self._json(d)

        def _body(self) -> dict | None:
            """The JSON object of a POST, or None after answering 4xx."""
            if self.headers.get_content_type() != "application/json":
                return self._json({"error": "Content-Type must be application/json"}, 415)
            n = self.headers.get("Content-Length", "")
            if not (n.isascii() and n.isdigit()):
                return self._json({"error": "Content-Length required"}, 411)
            if int(n) > MAX_BODY:
                return self._json({"error": "too big"}, 413)
            try:
                body = json.loads(self.rfile.read(int(n)) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return self._json({"error": "bad JSON"}, 400)
            if not isinstance(body, dict):
                return self._json({"error": "expected a JSON object"}, 400)
            return body

        def do_POST(self):
            if not self._host_ok():
                return self._json({"error": "bad host"}, 403)
            origin = self.headers.get("Origin")
            if origin is not None and not LOCAL_ORIGIN.fullmatch(origin):
                return self._json({"error": "foreign origin"}, 403)
            path = urlsplit(self.path).path
            if path not in ("/api/chat", "/api/agent", "/api/reset"):
                return self._json({"error": "not found"}, 404)
            if not self._authed():
                return self._json({"error": "missing or wrong token (see ~/.moth/token)"}, 401)
            body = self._body()
            if body is None:
                return
            if path == "/api/chat":
                text = body.get("text")
                text = text.strip()[:500] if isinstance(text, str) else ""
                if not text:
                    return self._json({"error": "empty"}, 400)
                self._json({"events": moth.say_to(text), "tree": moth.home.tree()})
            elif path == "/api/agent":   # from moth/integrations/* : one-way, never replies to the agent
                text = body.get("text")
                text = text.strip()[:20000] if isinstance(text, str) else ""
                if not text:
                    return self._json({"error": "empty"}, 400)
                source = body.get("source")
                source = source.lower() if isinstance(source, str) and SOURCE.fullmatch(source.lower()) else None
                status = moth.hear_agent(text, source, session=clean_session(body.get("session")),
                                         label=clean_label(body.get("label")))
                self._json({"ok": True, "status": status})
            else:   # /api/reset
                with moth.lock:
                    if body.get("new_home") is True and not moth.home.owned:
                        return self._json({"error": f"moth didn't make {moth.home.root} itself, so it won't delete "
                                                    f"it. move it away by hand for a fresh home.",
                                           "tree": moth.home.tree()}, 409)
                    if body.get("everything") is True:   # moth forget: also the phrase memory
                        moth.forget(everything=True)
                    else:
                        moth.forget()
                    if body.get("new_home") is True:
                        shutil.rmtree(moth.home.root, ignore_errors=True)
                        moth.home = Home(moth.a.home, random.randrange(10**6), moth.a.pool_file)
                self._json({"ok": True, "tree": moth.home.tree(), "epoch": moth.log.epoch})

    return H


CPU_QUANT = "int8"   # --quant auto on the CPU (tiny_agent.generate.quantize_cpu; see release/README.md "Speed")


def resolve_quant(a) -> str:
    """--quant auto -> CPU_QUANT on the CPU, none (unquantized) on a GPU. A namespace without quant: none."""
    q = getattr(a, "quant", None) or "none"
    if q == "auto":
        return CPU_QUANT if getattr(a, "device", "cpu") == "cpu" else "none"
    return q


def default_threads() -> int:
    """At most 8. On a 16-core 5950X with free cores, 16 threads are 10-30% faster for both int8 decode and long
    prefill (so no separate prefill count), but more threads than free cores (a busy desktop, a CPU quota) cost a lot:
    under a 4-CPU quota 8 threads halve int8 decode vs 4. 8 is the safe middle (tests/bench_speed.py --threads)."""
    return max(1, min(8, os.cpu_count() or 1))


def build_parser() -> argparse.ArgumentParser:
    d = default_paths()
    ap = argparse.ArgumentParser(description="moth's local chat server")
    ap.add_argument("--model", default=d["model"])
    ap.add_argument("--tokenizer", default=d["tokenizer"])
    ap.add_argument("--home", default=d["home"], help="moth's home folder (kept between runs)")
    ap.add_argument("--pool-file", default=d["pool"])
    ap.add_argument("--seed", type=int, default=3, help="which home moth starts with")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.3, help="on words from moth's own recent replies")
    ap.add_argument("--phrase-memory", type=int, default=16,
                    help="a new reaction can't repeat a 3-token phrase from moth's last N reactions, and the "
                         "repetition penalty covers the last 3 of them, also after a new chat (0: off)")
    ap.add_argument("--phrase-n", type=int, default=3, help=argparse.SUPPRESS)   # the phrase ban's n-gram length
    ap.add_argument("--phrase-ban-chat", action="store_true",
                    help="use the phrase ban in chat replies too (off: chat recall is never affected)")
    ap.add_argument("--ctx-budget", type=int, default=None, help="max tokens of chat moth keeps in mind")
    ap.add_argument("--max-turn-tokens", type=int, default=200)
    ap.add_argument("--max-tool-rounds", type=int, default=3)
    ap.add_argument("--quant", default="auto", choices=["auto", "none", "int8", "bf16"],
                    help=f"Linear weights on the CPU: auto (default) = {CPU_QUANT} on the CPU, none on a GPU; "
                         "int8 (weight-only, per-channel) and bf16 are the fast modes; none = exact fp32 (slowest)")
    ap.add_argument("--fast", action="store_const", const=CPU_QUANT, dest="quant",
                    help=f"same as --quant {CPU_QUANT} (already the CPU default)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: min(8, CPUs))")
    ap.add_argument("--no-graph", action="store_true", help="GPU: decode without device graphs (debugging)")
    ap.add_argument("--compile", action="store_true",
                    help="GPU: torch.compile the decode step (~1.4x faster replies on the B70; ~80 s at startup)")
    ap.add_argument("--react-every", type=float, default=REACT_EVERY,
                    help="seconds between reactions to the coding agent (sorry/failed/long messages always count)")
    ap.add_argument("--quiet-after", type=float, default=QUIET_AFTER_CHAT,
                    help="hold reactions until you haven't chatted for this many seconds")
    ap.add_argument("--batch-window", type=float, default=BATCH_WINDOW,
                    help="when moth may react, wait this many seconds for other agent sessions to finish too, then "
                         "react to the most interesting message (0: no waiting; --react-every 0 never waits either)")
    ap.add_argument("--token-file", default=None, help="default ~/.moth/token (or $MOTH_TOKEN_FILE)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4747)
    ap.add_argument("--exit-with-parent", action="store_true",
                    help="for launchers (the desktop pet): stop when stdin closes, i.e. when the launcher that holds "
                         "the other end of the pipe dies")
    return ap


def trim_heap():
    """Give the memory freed while loading (the fp32 weights --quant replaces, load buffers) back to the system: glibc
    keeps freed heap otherwise (~15 MB less RSS for an all-day server). Not glibc: nothing."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (OSError, AttributeError):
            pass


def main(argv=None, on_ready=None):
    a = build_parser().parse_args(argv)
    from moth import weights
    live = {}
    if a.exit_with_parent:   # first, so a pet killed while the model loads takes this server too (moth/launch.py)
        import _thread
        from moth.launch import exit_with_parent
        exit_with_parent(lambda: live["srv"].shutdown() if "srv" in live else _thread.interrupt_main())
    weights.ensure_default(a.model)   # pip install, first run: download the model files
    token = load_token(a.token_file)
    torch.set_num_threads(a.threads or default_threads())
    moth = Moth(a)
    trim_heap()
    hosts = ("127.0.0.1", "localhost", "[::1]") + ((a.host,) if a.host not in ("0.0.0.0", "::", "") else ())
    srv = Server((a.host, a.port), make_handler(moth, token, hosts))
    moth.start_reactor()
    url = f"http://127.0.0.1:{a.port}" if a.host in ("0.0.0.0", "::", "") else f"http://{a.host}:{a.port}"
    print(f"moth is awake at {url}  (home: {moth.home.root})", flush=True)
    if on_ready:
        on_ready(url)
    live["srv"] = srv   # from here the lifeline stops serve_forever (and the finally below removes the pidfile)
    pidfile = write_pidfile(srv.server_address[1])
    try:
        srv.serve_forever()
    finally:
        remove_pidfile(pidfile)


if __name__ == "__main__":
    main()
