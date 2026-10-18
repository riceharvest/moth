"""moth/shell.py against the real thing.

  python3 tests/test_shell.py        (from ~/moth; a few seconds, no model)

Replay: every in-grammar bash call in results/r3/b.jsonl (bwrap'd GNU bash, recorded while making the training
data) is re-run through moth.shell from the chat's starting home (+ the chat's writes so far). A chat stops being
compared after an out-of-grammar call that can change files (echo > x, rm, mv, heredocs ...).
Differential: tricky names and error cases run through moth.shell and through tiny_agent.tools' own bwrap'd bash
(skipped without bwrap). Unit: parsing, escapes and refusals.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moth.shell import MARKER, REFUSE, Shell, split, Unsupported  # noqa: E402
from tiny_agent.tools import Workspace  # noqa: E402

DATA = os.path.join(ROOT, "results", "r3", "b.jsonl")
MUTATES = re.compile(r">|<<|\b(rm|mv|cp|mkdir|touch|tee|sed|truncate|python3?|cd)\b")


def materialize(root: str, files: dict[str, str]):
    for rel, content in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:   # "\n" stays "\n" on Windows too
            f.write(content)


def replay(path: str = DATA, limit: int | None = None):
    """-> (Counter of outcomes, list of mismatches (cmd, expected, got), Counter of commands by first word)"""
    stats, bad, refused = collections.Counter(), [], collections.Counter()
    base = tempfile.mkdtemp()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                break
            d = json.loads(line)
            root = tempfile.mkdtemp(dir=base)
            materialize(root, d["home"])
            sh, ws = Shell(root), Workspace.__new__(Workspace)
            ws.root, ws.read_paths, ws.files = os.path.realpath(root), set(), {}
            msgs, alive = d["messages"], True
            for i, m in enumerate(msgs):
                if not alive:
                    break
                for k, c in enumerate(m.get("tool_calls") or []):
                    if "error" in c or i + 1 >= len(msgs) or msgs[i + 1].get("role") != "tool":
                        continue
                    want = msgs[i + 1]["results"][k] if k < len(msgs[i + 1]["results"]) else None
                    args = c.get("arguments") or {}
                    if c["name"] == "write":
                        ws.call("write", args)
                    elif c["name"] == "bash" and set(args) - {"command", "timeout"}:
                        stats["bad_args"] += 1   # tool_bash(limit=...): a TypeError result, not a shell output
                    elif c["name"] == "bash" and isinstance(args.get("command"), str):
                        cmd = args["command"]
                        got = sh.run(cmd)
                        if got == REFUSE:
                            refused[(cmd.split() or [""])[0]] += 1
                            if MUTATES.search(cmd):
                                alive = False
                                stats["stopped"] += 1
                                break
                            continue
                        if "/dev/shm/tiny-agent-ws" in (want or ""):
                            stats["unsandboxed"] += 1   # a MOTH_NO_BWRAP box: real paths, not /work
                            continue
                        stats["compared"] += 1
                        stats["compared_" + cmd.split()[0]] += 1
                        if got == want:
                            stats["match"] += 1
                            stats["match_" + cmd.split()[0]] += 1
                        else:
                            bad.append((cmd, want, got))
            shutil.rmtree(root)
    shutil.rmtree(base, ignore_errors=True)
    return stats, bad, refused


@unittest.skipUnless(os.path.exists(DATA), "needs results/r3/b.jsonl (local training data)")
class Replay(unittest.TestCase):
    def test_training_outputs(self):
        stats, bad, refused = replay()
        rate = stats["match"] / max(1, stats["compared"])
        print(f"\nshell vs training bash: {stats['match']}/{stats['compared']} identical ({rate:.2%}); "
              f"{stats['unsandboxed']} unsandboxed results and {stats['bad_args']} bad-argument calls skipped; {stats['stopped']} chats stopped at a "
              f"file-changing command; refused by first word: {dict(refused.most_common(8))}; per command: "
              + ", ".join(f"{c} {stats['match_' + c]}/{stats['compared_' + c]}" for c in ("ls", "cat", "head", "wc", "echo")),
              file=sys.stderr)
        for cmd, want, got in bad[:10]:
            print(f"--- {cmd!r}\nwant {want!r}\ngot  {got!r}", file=sys.stderr)
        self.assertGreater(stats["compared"], 3000)
        self.assertGreaterEqual(rate, 0.99)


HOME = {
    "diary/tuesday.txt": "the fan was gentle today.\ni ate a crumb. good day.\n",
    "diary/no_newline.txt": "one\ntwo",
    "diary/it's mine.txt": "mine\n",
    "lamps/B.txt": "big\n", "lamps/a.txt": "a\n", "lamps/_u.txt": "u\n", "lamps/Z-1.txt": "z\n",
    "lamps/é.txt": "accent\n", "lamps/sub/deep.txt": "deep\n",
    "saved/.keep": "", "snacks/empty.txt": "", "snacks/words.txt": "a b c\n\nlast line with words\n" * 3,
}
COMMANDS = [
    "ls", "ls -a", "ls -A", "ls -1", "ls -1a", "ls ~", "ls ~/", "ls ~/lamps", "ls /work", "ls /work/lamps/",
    "ls lamps diary", "ls lamps/a.txt diary", "ls missing lamps", "ls missing", "ls missing nope", "ls saved",
    "ls -a saved", "ls ~ -R", "ls -R lamps", "ls -aR lamps", "ls ~snacks", "ls ~lamps/", "ls ~ lamps/", "ls ..",
    "ls /etc", "ls lamps/a.txt/", "ls ''", "ls \"diary/it's mine.txt\"", "ls 'a  b'", "ls lamps/../diary",
    "cat diary/tuesday.txt", "cat ~/diary/tuesday.txt", "cat /work/diary/tuesday.txt", "cat diary",
    "cat diary/tuesday.txt missing diary/no_newline.txt", "cat diary/no_newline.txt lamps/a.txt",
    "cat \"diary/it's mine.txt\"", "cat 'diary/it'\\''s x'", "cat 'a  b'", "cat ''", "cat lamps/a.txt/",
    "cat ../etc/passwd", "cat /etc/hostname", "cat snacks/empty.txt", "cat diary/tuesday.txt diary/tuesday.txt",
    "head diary/tuesday.txt", "head -n 1 diary/tuesday.txt", "head -1 snacks/words.txt", "head -n -2 snacks/words.txt",
    "head missing", "head diary", "head diary/tuesday.txt lamps/a.txt", "head diary lamps/a.txt",
    "wc diary/tuesday.txt", "wc -l diary/tuesday.txt", "wc -w snacks/words.txt", "wc -c diary/no_newline.txt",
    "wc -lc snacks/words.txt", "wc -m lamps/é.txt", "wc diary", "wc -l diary", "wc missing",
    "wc diary/tuesday.txt lamps/a.txt", "wc -l ~/diary/tuesday.txt",
    "echo", "echo hi", "echo \"hello  moth\"", "echo 'it''s' \"a 'b'\" c\\ d", "echo -n hi", "echo -nE x", "echo -- -n",
    "echo -x y", "echo ~ ~/diary '~'", "echo \"i'm a moth\" — ok", "echo   spaced    out  ",
]


def bwrap_works() -> bool:
    """bwrap installed AND allowed to make namespaces (Ubuntu 24.04 / CI runners may block unprivileged userns)."""
    if not (sys.platform.startswith("linux") and shutil.which("bwrap")):
        return False
    import subprocess
    try:
        return subprocess.run(["bwrap", "--ro-bind", "/", "/", "--unshare-all", "true"], capture_output=True,
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(bwrap_works(), "needs a working bubblewrap (Linux)")
class Differential(unittest.TestCase):
    def test_against_bwrapped_bash(self):
        ws = Workspace(dict(HOME))
        try:
            sh = Shell(ws.root)
            open(os.path.join(ws.root, MARKER), "w").close()   # invisible to moth's shell, so hide it from bash
            for cmd in COMMANDS:
                with self.subTest(cmd=cmd):
                    os.rename(os.path.join(ws.root, MARKER), os.path.join(ws.root, "..", "m" + os.path.basename(ws.root)))
                    try:
                        want = ws.tool_bash(cmd)
                    finally:
                        os.rename(os.path.join(ws.root, "..", "m" + os.path.basename(ws.root)), os.path.join(ws.root, MARKER))
                    got = sh.run(cmd)
                    if cmd in ("ls ..", "ls /etc", "cat /etc/hostname"):   # outside the home: refused on purpose
                        self.assertRegex(got, r"No such file or directory\n\n\[exit code [12]\]$")
                        continue
                    self.assertEqual(got, want)
        finally:
            ws.close()

    def test_name_quoting(self):
        """coreutils quotes a file name in its messages only when needed, and differently per tool."""
        import shlex
        names = [n for c in " !\"#$%&'()*+,.:;<=>?@[\\]^_`{|}~" for n in (f"a{c}b", f"{c}ab", f"ab{c}")]
        names += ["it's a", "it's$", "a'b'c", "~", "#", "{", "}", "é'x", "x\ty", "\tx", "x\t", "it's:"]
        ws = Workspace({"d/x.txt": "x\n"})
        try:
            sh = Shell(ws.root)
            for name in names:
                for tool in ("cat", "ls", "head", "wc"):
                    cmd = f"{tool} {shlex.quote(name)}"
                    with self.subTest(cmd=cmd):
                        self.assertEqual(sh.run(cmd), ws.tool_bash(cmd))
        finally:
            ws.close()


class Unit(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        materialize(self.d, HOME)
        open(os.path.join(self.d, MARKER), "w").close()
        self.sh = Shell(self.d)

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_split(self):
        self.assertEqual(split("ls ~ ~/a '~/b' \"x y\" a\\ b ~x ~+/c"),
                         ["ls", "/work", "/work/a", "~/b", "x y", "a b", "~x", "/work/c"])
        for bad in ("ls | head", "cat a > b", "ls *", "echo $HOME", "ls; rm x", "cat `x`", "ls\nls", "cat <<EOF",
                    "ls {a,b}", "ls 'open", "cat \"$x\"", "# hi"):
            with self.subTest(bad=bad), self.assertRaises(Unsupported):
                split(bad)

    def test_refusals(self):
        for cmd in ("echo -e 'a\\tb'", "echo hi > x", "echo $HOME", "rm diary/tuesday.txt", "ls -l", "grep -r x .",
                    "cat a | head", "pwd", "head -c 3 x",
                    "/bin/ls", "wc", "ls > x"):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.sh.run(cmd), REFUSE)
        self.assertEqual(self.sh.run(""), "(no output)")

    def test_stays_home(self):
        outside = tempfile.mkdtemp()
        try:
            with open(os.path.join(outside, "secret.txt"), "w") as f:
                f.write("secret\n")
            try:
                os.symlink(outside, os.path.join(self.d, "saved", "door"), target_is_directory=True)
                os.symlink(os.path.join(outside, "secret.txt"), os.path.join(self.d, "saved", "s.txt"))
            except OSError as e:   # Windows without symlink rights (no admin / developer mode)
                self.skipTest(f"can't make symlinks here: {e}")
            for cmd in ("cat saved/door/secret.txt", "cat saved/s.txt", "cat ../" + os.path.basename(outside)
                        + "/secret.txt", f"cat {outside}/secret.txt", "ls saved/door", "head saved/s.txt",
                        "wc saved/s.txt", "cat /work/../" + outside.lstrip("/") + "/secret.txt"):
                with self.subTest(cmd=cmd):
                    self.assertNotIn("secret", self.sh.run(cmd).replace("secret.txt", ""))
                    self.assertIn("No such file or directory", self.sh.run(cmd))
        finally:
            shutil.rmtree(outside)

    def test_marker_invisible(self):
        self.assertNotIn(MARKER, self.sh.run("ls -a"))
        self.assertNotIn(MARKER, self.sh.run("ls -aR ~"))
        self.assertIn("No such file", self.sh.run(f"cat {MARKER}"))

    def test_windows_names_and_marker_case(self):
        from moth.shell import _windows_unsafe
        for p in ("diary\\tuesday.txt", "C:x", "x::$DATA", "CON", "nul.txt", "diary/Com1.log", ".moth-home.", "x ",
                  "lpt\u00b9", "\\\\host\\share\\x"):
            self.assertTrue(_windows_unsafe(p), p)
        for p in ("diary/tuesday.txt", "../x", ".", "it's mine.txt", "console.txt", "a.b.c", "lamps/"):
            self.assertFalse(_windows_unsafe(p), p)
        for cmd in ("cat .MOTH-HOME", "cat /work/.Moth-Home", "head .moth-HOME"):   # one file on macOS / Windows
            self.assertIn("No such file", self.sh.run(cmd))
        if os.name == "nt":
            for cmd in ("cat 'diary\\tuesday.txt'", "cat CON", "cat 'C:\\Windows\\win.ini'", "cat .moth-home."):
                self.assertIn("No such file", self.sh.run(cmd))

    def test_shapes(self):
        self.assertEqual(self.sh.run("ls"), "diary\nlamps\nsaved\nsnacks")
        self.assertEqual(self.sh.run("ls saved"), "(no output)")
        self.assertEqual(self.sh.run("ls ~/lamp"),
                         "ls: cannot access '/work/lamp': No such file or directory\n\n[exit code 2]")
        self.assertEqual(self.sh.run("cat diary/x.txt diary/tuesday.txt"),
                         "the fan was gentle today.\ni ate a crumb. good day.\n"
                         "cat: diary/x.txt: No such file or directory\n\n[exit code 1]")
        with open(os.path.join(self.d, "saved", "big.txt"), "w") as f:
            f.write("x" * 9000)
        self.assertIn("[... truncated", self.sh.run("cat saved/big.txt"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
