"""moth's bash tool at serve time: a tiny pure-Python shell, no subprocess, no bubblewrap.

The model only ever learned a handful of commands (results/r3: ls / cat cover ~93% of its bash calls). This
implements exactly that grammar, scoped to the home folder, and answers byte for byte like the bwrap'd GNU bash
it was trained against (tiny_agent/tools.py: HOME=/work, cwd /work, LANG=C.UTF-8):

  ls [-1aAR] [path ...]       one name per line (not a tty), C.UTF-8 = code point order, dir headers when >1 arg
  cat path ...
  head [-n N | -N] path ...   wc [-lwcm] path ...
  echo [-n] words ...         bash's builtin: arguments joined by spaces (no -e escapes)
  ~ and ~/x expand to /work; /work/x and relative paths are inside the home; ~name stays literal (no passwd)

Output is built like tools.Workspace.tool_bash: stdout, then stderr, then "\\n[exit code N]", stripped,
"(no output)" when empty, truncated at MAX_RESULT_CHARS. Error lines use coreutils' own wording and quoting.
Paths that leave the home (/etc, .., a symlink out) answer "No such file or directory". Anything else
(pipes, redirects, globs, $vars, other programs) gets REFUSE. Works the same on Linux, macOS and Windows.
"""
from __future__ import annotations

import errno
import os
import re

from tiny_agent.tools import MAX_RESULT_CHARS, truncate

WORK = "/work"                   # what the model believes its home is called
MARKER = ".moth-home"            # written by serve.Home when it creates a home; invisible to the shell
REFUSE = "moth can only use ls, cat, head, wc and echo here."
HINT = ", narrow the command (e.g. | head, grep)"

_UNQUOTED_META = set("|&;<>()$`*?[{}\n\r")
_REASON = {errno.ENOENT: "No such file or directory", errno.ENOTDIR: "Not a directory",
           errno.EISDIR: "Is a directory", errno.EACCES: "Permission denied"}


class Unsupported(Exception):
    pass


# --- words --------------------------------------------------------------------------------------------------
def split(command: str) -> list[str]:
    """bash word splitting for the plain subset: quotes, backslashes and tilde expansion; anything that would
    need a real shell (pipes, redirects, globs, $, `, braces, a second line) raises Unsupported."""
    words, cur, started, tilde_ok = [], [], False, False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c in " \t":
            if started:
                words.append(_tilde("".join(cur), tilde_ok))
                cur, started = [], False
            i += 1
            continue
        if not started:
            started, tilde_ok = True, c == "~"
            if c == "#":
                raise Unsupported("comment")
        if c == "'":
            j = command.find("'", i + 1)
            if j < 0:
                raise Unsupported("open quote")
            cur.append(command[i + 1:j])
            i = j + 1
        elif c == '"':
            i += 1
            while True:
                if i >= n:
                    raise Unsupported("open quote")
                c = command[i]
                if c == '"':
                    i += 1
                    break
                if c in "$`":
                    raise Unsupported("expansion")
                if c == "\\" and i + 1 < n and command[i + 1] in '"\\$`':
                    cur.append(command[i + 1])
                    i += 2
                    continue
                cur.append(c)
                i += 1
        elif c == "\\":
            if i + 1 >= n or command[i + 1] == "\n":
                raise Unsupported("line continuation")
            cur.append(command[i + 1])
            i += 2
        elif c in _UNQUOTED_META:
            raise Unsupported(c)
        else:
            cur.append(c)
            i += 1
    if started:
        words.append(_tilde("".join(cur), tilde_ok))
    return words


def _tilde(word: str, ok: bool) -> str:
    """~ and ~/x -> /work...; ~+ is $PWD (also /work); ~name and ~- have nothing to expand to in the sandbox."""
    if not ok:
        return word
    head, sep, rest = word.partition("/")
    if head in ("~", "~+"):
        return WORK + sep + rest
    return word


# --- coreutils' quoting of file names in messages (gnulib quotearg, shell-escape styles) ---------------------
_SPECIAL = set(" \t\n!\"$&'()*;<=>?[\\^`|")   # plus # ~ first, { } alone, : in quotef


def quote(name: str, always: bool) -> str:
    """always=True: quote() (ls "cannot access", head "cannot open"); False: quotef() (cat, wc)."""
    if name == "":
        return "''"
    if any(ord(c) < 32 or ord(c) == 127 for c in name):   # 'tab'$'\t''x'
        s = "'"
        for c in name:
            if ord(c) < 32 or ord(c) == 127:
                s += "'$'\\" + _CESC.get(c, f"{ord(c):03o}") + "''"
            else:
                s += "'\\''" if c == "'" else c
        s += "'"
        return s[:-2] if s.endswith("''") else s
    needs = (always or name[0] in "#~" or name in ("{", "}") or any(c in _SPECIAL for c in name)
             or ":" in name)
    if not needs:
        return name
    if "'" in name:
        if not any(c in '"$`\\!' for c in name):
            return f'"{name}"'
        return "'" + name.replace("'", "'\\''") + "'"
    return f"'{name}'"


_CESC = {"\a": "a", "\b": "b", "\t": "t", "\n": "n", "\v": "v", "\f": "f", "\r": "r"}


# --- the shell ----------------------------------------------------------------------------------------------
class Shell:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)

    def run(self, command: str) -> str:
        """The tool result for a `bash` call, formatted like tools.Workspace.tool_bash."""
        try:
            words = split(command)
        except Unsupported:
            return REFUSE
        if not words:
            return "(no output)"
        fn = {"ls": self.ls, "cat": self.cat, "head": self.head, "wc": self.wc, "echo": self.echo}.get(words[0])
        if fn is None:
            return REFUSE
        out, err = [], []
        try:
            code = fn(words[1:], out, err)
        except Unsupported:
            return REFUSE
        text = (b"".join(out) + "".join(err).encode()).decode("utf-8", "replace")
        if code:
            text += f"\n[exit code {code}]"
        return truncate(text.strip() or "(no output)", MAX_RESULT_CHARS, hint=HINT)

    # paths ------------------------------------------------------------------------------------------------
    def locate(self, path: str) -> str:
        """The real path for a path as the model wrote it (after ~ expansion). Raises OSError like the kernel
        would, and ENOENT for anything outside the home (or the marker file)."""
        if not path:
            raise OSError(errno.ENOENT, "empty")
        if path.startswith("/"):
            if path != WORK and not path.startswith(WORK + "/"):
                raise OSError(errno.ENOENT, "outside")
            path = path[len(WORK):].lstrip("/") or "."
        if os.name == "nt" and _windows_unsafe(path):
            raise OSError(errno.ENOENT, "outside")
        full = os.path.join(self.root, path)
        real = os.path.realpath(full)
        if real != self.root and not real.startswith(self.root + os.sep):
            raise OSError(errno.ENOENT, "outside")
        if real.casefold() == os.path.join(self.root, MARKER).casefold():   # macOS / Windows: .MOTH-HOME is it too
            raise OSError(errno.ENOENT, "marker")
        os.stat(full)   # ENOENT / ENOTDIR exactly as the kernel says it, trailing slash included
        return full

    def entries(self, full: str, hidden: str) -> list[str]:
        """hidden: "" (ls), "A" (-A) or "a" (-a). Code point order is C.UTF-8's collation."""
        names = os.listdir(full)
        if os.path.realpath(full) == self.root:
            names = [x for x in names if x != MARKER]
        if not hidden:
            names = [x for x in names if not x.startswith(".")]
        names.sort()
        return [".", ".."] + names if hidden == "a" else names

    @staticmethod
    def flags(args: list[str], allowed: str) -> tuple[set[str], list[str]]:
        """GNU-style: options anywhere before `--`, bundled (-1a). Unknown option -> Unsupported."""
        opts, paths, done = set(), [], False
        for a in args:
            if not done and a == "--":
                done = True
            elif not done and a.startswith("-") and len(a) > 1:
                if a.startswith("--") or any(c not in allowed for c in a[1:]):
                    raise Unsupported(a)
                opts |= set(a[1:])
            else:
                paths.append(a)
        return opts, paths

    # ls ---------------------------------------------------------------------------------------------------
    def ls(self, args, out, err) -> int:
        opts, paths = self.flags(args, "1aAR")
        hidden = "a" if "a" in opts else "A" if "A" in opts else ""
        code, files, dirs = 0, [], []
        for p in paths or ["."]:
            try:
                full = self.locate(p)
            except OSError as e:
                err.append(f"ls: cannot access {quote(p, True)}: {_REASON.get(e.errno, e.strerror)}\n")
                code = 2
                continue
            (dirs if os.path.isdir(full) else files).append((p, full))
        files.sort()
        dirs.sort()
        for p, _ in files:
            out.append(p.encode() + b"\n")
        headers = len(paths) > 1 or "R" in opts
        first = not files
        queue = list(dirs)
        while queue:
            p, full = queue.pop(0)
            if headers:
                out.append(("" if first else "\n").encode() + p.encode() + b":\n")
            first = False
            try:
                names = self.entries(full, hidden)
            except OSError as e:
                err.append(f"ls: cannot open directory {quote(p, True)}: {_REASON.get(e.errno, e.strerror)}\n")
                code = max(code, 1)
                continue
            out.extend(x.encode("utf-8", "surrogateescape") + b"\n" for x in names)
            if "R" in opts:
                subs = [(p.rstrip("/") + "/" + x if p != "/" else "/" + x, os.path.join(full, x)) for x in names
                        if x not in (".", "..") and os.path.isdir(os.path.join(full, x))
                        and not os.path.islink(os.path.join(full, x))]
                queue[:0] = subs
        return code

    # echo -------------------------------------------------------------------------------------------------
    def echo(self, args, out, err) -> int:
        newline, words = _echo_opts(args)
        out.append((" ".join(words) + ("\n" if newline else "")).encode("utf-8", "surrogateescape"))
        return 0

    # cat / head / wc --------------------------------------------------------------------------------------
    def read(self, p: str) -> bytes:
        full = self.locate(p)
        if os.path.isdir(full):
            raise OSError(errno.EISDIR, "dir")
        with open(full, "rb") as f:
            return f.read(4 << 20)

    def cat(self, args, out, err) -> int:
        _, paths = self.flags(args, "")
        code = 0
        for p in paths:
            try:
                out.append(self.read(p))
            except OSError as e:
                err.append(f"cat: {quote(p, False)}: {_REASON.get(e.errno, e.strerror)}\n")
                code = 1
        return code

    def head(self, args, out, err) -> int:
        n, rest, i = 10, [], 0
        while i < len(args):   # -n N, -nN, -N, --lines=N
            a = args[i]
            if a == "-n" and i + 1 < len(args):
                n, i = _int(args[i + 1]), i + 2
                continue
            m = re.fullmatch(r"-n(-?\d+)|-(\d+)|--lines=(-?\d+)", a)
            if m:
                n = int(next(g for g in m.groups() if g is not None))
            elif a.startswith("-") and a != "-":
                raise Unsupported(a)
            else:
                rest.append(a)
            i += 1
        code = 0
        for k, p in enumerate(rest):
            try:
                data = self.read(p)
            except OSError as e:
                verb = "error reading" if e.errno == errno.EISDIR else "cannot open"
                tail = "" if e.errno == errno.EISDIR else " for reading"
                err.append(f"head: {verb} {quote(p, True)}{tail}: {_REASON.get(e.errno, e.strerror)}\n")
                code = 1
                if e.errno == errno.EISDIR and len(rest) > 1:
                    out.append(("\n" if k else "").encode() + f"==> {p} <==\n".encode())
                continue
            if len(rest) > 1:
                out.append(("\n" if k else "").encode() + f"==> {p} <==\n".encode())
            lines = data.splitlines(keepends=True)
            out.append(b"".join(lines[:n] if n >= 0 else lines[:max(0, len(lines) + n)]))
        return code

    def wc(self, args, out, err) -> int:
        opts, paths = self.flags(args, "lwcm")
        if not paths:
            raise Unsupported("stdin")
        kinds = [k for k in "lwmc" if k in opts] or ["l", "w", "c"]
        rows, code, total, size, regular = [], 0, dict.fromkeys(kinds, 0), 0, True
        for p in paths:
            try:
                full = self.locate(p)
                if os.path.isdir(full):
                    rows.append((p, dict.fromkeys(kinds, 0)))
                    regular = False
                    raise OSError(errno.EISDIR, "dir")
                data = self.read(p)
            except OSError as e:
                rows.append((p, e))
                continue
            size += len(data)
            text = data.decode("utf-8", "replace")
            c = {"l": data.count(b"\n"), "w": len(text.split()), "m": len(text), "c": len(data)}
            rows.append((p, {k: c[k] for k in kinds}))
        # width: one file + one count -> 1; else digits of the regular files' total size (7 for a non-file)
        if len(paths) == 1 and len(kinds) == 1:
            width = 1
        else:
            ok = [r for r in rows if isinstance(r[1], dict)]
            width = len(str(size)) if ok else 1
            if not regular:
                width = max(width, 7)
        for p, r in rows:
            if isinstance(r, OSError):
                err.append(f"wc: {quote(p, False)}: {_REASON.get(r.errno, r.strerror)}\n")
                code = 1
                continue
            out.append((" ".join(str(r[k]).rjust(width) for k in kinds) + f" {p}\n").encode())
            for k in kinds:
                total[k] += r[k]
        if len(paths) > 1:
            out.append((" ".join(str(total[k]).rjust(width) for k in kinds) + " total\n").encode())
        return code


_WIN_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {f"{d}{i}" for d in ("com", "lpt")
                                                                      for i in "123456789\u00b9\u00b2\u00b3"}


def _windows_unsafe(path: str) -> bool:
    """On Windows "\\" is a separator, ":" a drive or a stream (C:x, \\\\host\\share, x::$DATA), device names (CON,
    nul.txt) open devices, and a trailing dot or space is dropped (".moth-home." is the marker). In bash on Linux
    all of these are plain file names; here none of them may reach the file system."""
    if "\\" in path or ":" in path:
        return True
    for part in path.split("/"):
        if part in ("", ".", ".."):
            continue
        if part[-1] in ". " or part.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES:
            return True
    return False


def _echo_opts(args: list[str]) -> tuple[bool, list[str]]:
    """bash's echo: leading words made only of n/e/E after a dash are options; -e (escapes) is not supported."""
    newline, i = True, 0
    while i < len(args) and re.fullmatch(r"-[neE]+", args[i]):
        if "e" in args[i]:
            raise Unsupported("echo -e")
        newline = newline and "n" not in args[i]
        i += 1
    return newline, args[i:]


def _int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        raise Unsupported(s) from None
