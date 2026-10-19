"""End-to-end check of moth's coding-agent hooks against the REAL harness CLIs, with a fake model and a fake moth.

  python tests/e2e/run_e2e.py [--harness claude-code codex pi opencode] [--soft pi opencode] [--spaced-python]
                              [--out DIR] [--keep]

For each run it:
  1. makes a scratch root "moth e2e XXXX/" (a space on purpose) with a scratch home, a copy of the moth package in
     "moth repo/" and a project folder "proj e2e/"; with --spaced-python also a venv in "py env/", so the hook's
     interpreter path has a space too (the case Windows shells quote differently);
  2. starts tests/e2e/fake_moth.py (serve.py's real handler, a recording engine; token in <home>/.moth/token) and
     tests/e2e/fake_llm.py (Anthropic Messages + OpenAI Responses / Chat, one fixed reply, no tool calls);
  3. installs moth's hooks the way a person does: `python -m moth hook <harness...> --yes`, run from the copy, with
     HOME / USERPROFILE / CLAUDE_CONFIG_DIR / CODEX_HOME / XDG_CONFIG_HOME / PI_CODING_AGENT_DIR all in the scratch
     home (never the real one), and checks the configs landed there and point at the copy;
  4. runs each harness one-shot in "proj e2e/" against the fake model:
       claude -p hi --output-format json                    (session id: the JSON's session_id)
       codex exec --json --skip-git-repo-check hi            (twice: without --dangerously-bypass-hook-trust the
                                                              unapproved hook must NOT post; with it, it must)
                                                             (session id: the thread.started event's thread_id)
       pi -p --session-id <uuid> --provider moth-e2e ...     (session id: the one we set)
       opencode run --format json -m moth-e2e/fake-model hi  (session id: the events' sessionID)
     and asserts moth got exactly the fake model's reply, with the right source, session and label ("proj e2e"),
     and (Claude Code, Codex) that it arrived BEFORE the CLI exited: the hooks run in the foreground for that;
  5. Codex with show_in_harness on: two turns of one thread (exec + exec resume); the Stop hook fetches moth's line
     for that thread (its {"systemMessage"}), and that line must appear in no request body the fake model gets.
Results go to <out>/results.json and a table on stdout; logs (fake model requests, moth's record, each CLI's output,
the installed configs) to <out>/. Exit 1 if a harness not listed in --soft failed; a harness whose CLI isn't on PATH
is skipped (and fails unless --soft lists it, so CI can't go green on a missing install).
Stdlib only. Never touches the real ~/.claude, ~/.codex, ~/.pi or moth's port 4747: everything is scratch, ports are
free ones picked by the OS.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPLY = "Fixed the login bug. All tests pass."
WIN = os.name == "nt"
HARNESSES = ("claude-code", "codex", "pi", "opencode")
BIN = {"claude-code": "claude", "codex": "codex", "pi": "pi", "opencode": "opencode"}
# environment that would point a CLI at a real account or the real config: never passed on
SCRUB = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY",
         "OPENAI_BASE_URL", "CODEX_API_KEY", "CLAUDE_CONFIG_DIR", "CODEX_HOME", "PI_CODING_AGENT_DIR", "XDG_CONFIG_HOME",
         "MOTH_URL", "MOTH_TOKEN_FILE", "MOTH_SHOW_IN_HARNESS", "MOTH_CONFIG_FILE", "CLAUDECODE",
         "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT", "CLAUDE_PLUGIN_ROOT")


def log(*a):
    print(*a, flush=True)


def wait_file(path: Path, timeout: float = 30.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.1)
    raise SystemExit(f"timed out waiting for {path}")


def records(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


class Env:
    def __init__(self, a):
        self.a = a
        base = Path(a.work).resolve() if a.work else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="moth e2e ", dir=base))
        self.home = self.root / "home"
        self.repo = self.root / "moth repo"
        self.proj = self.root / "proj e2e"
        self.out = Path(a.out).resolve() if a.out else self.root / "out"
        for d in (self.home, self.proj, self.out):
            d.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "moth", self.repo / "moth",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"))
        self.python = sys.executable
        if a.spaced_python:   # a venv whose path has a space: the hook command's first token must be quoted
            venv = self.root / "py env"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
            self.python = str(venv / ("Scripts/python.exe" if WIN else "bin/python"))
        self.procs: list[subprocess.Popen] = []

    def env(self, **extra) -> dict:
        e = {k: v for k, v in os.environ.items() if k not in SCRUB}
        e.update(HOME=str(self.home), CLAUDE_CONFIG_DIR=str(self.home / ".claude"), CODEX_HOME=str(self.home / ".codex"),
                 XDG_CONFIG_HOME=str(self.home / ".config"), PI_CODING_AGENT_DIR=str(self.home / ".pi" / "agent"),
                 MOTH_URL=self.moth_url, PYTHONIOENCODING="utf-8", NO_COLOR="1")
        if WIN:
            e["USERPROFILE"] = str(self.home)
        e.update(extra)
        return e

    def start(self, name: str, args: list[str]) -> dict:
        info = self.out / f"{name}.json"
        info.unlink(missing_ok=True)
        f = open(self.out / f"{name}.log", "w")
        p = subprocess.Popen([sys.executable, str(HERE / f"{name}.py"), "--info", str(info), *args],
                             stdout=f, stderr=subprocess.STDOUT)
        self.procs.append(p)
        return wait_file(info)

    def stop(self):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def run(cmd: list[str], env: dict, cwd: Path, name: str, out: Path, timeout: float = 120) -> tuple[int, str, str, float]:
    """(exit code, stdout, stderr, time.time() at exit)."""
    log(f"$ {subprocess.list2cmdline(cmd) if WIN else ' '.join(cmd)}   (in {cwd.name})")
    t0 = time.time()
    try:
        # PWD as a shell would set it: OpenCode 1.19 takes its directory from $PWD, not the process's cwd
        p = subprocess.run(cmd, env={**env, "PWD": str(cwd)}, cwd=str(cwd), capture_output=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        code, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code, so, se = -999, e.stdout or b"", (e.stderr or b"") + b"\n[run_e2e: timed out]"
    t1 = time.time()
    so, se = so.decode("utf-8", "replace"), se.decode("utf-8", "replace")
    (out / f"{name}.stdout.txt").write_text(so, encoding="utf-8")
    (out / f"{name}.stderr.txt").write_text(se, encoding="utf-8")
    log(f"  exit {code} after {t1 - t0:.1f}s; stdout {len(so)} chars, stderr {len(se)} chars")
    return code, so, se, t1


class Runner:
    def __init__(self, a):
        self.a = a
        self.results: dict[str, dict] = {}

    def setup(self):
        e = self.e = Env(self.a)
        self.record = e.out / "moth_record.jsonl"
        self.record.unlink(missing_ok=True)
        (e.out / "llm_requests.jsonl").unlink(missing_ok=True)
        self.bodies = e.out / "llm_bodies.jsonl"   # every request body in full: what the model was sent
        self.bodies.unlink(missing_ok=True)
        m = e.start("fake_moth", ["--token-file", str(e.home / ".moth" / "token"), "--record", str(self.record)])
        e.moth_url = f"http://127.0.0.1:{m['port']}"
        llm = e.start("fake_llm", ["--reply", REPLY, "--log", str(e.out / "llm_requests.jsonl"),
                                   "--bodies", str(self.bodies)])
        self.llm = f"http://127.0.0.1:{llm['port']}"
        log(f"scratch root: {e.root}\nhook python: {e.python}\nfake moth: {e.moth_url}   fake model: {self.llm}")

    # --- harness configs (written before `moth hook`, like a person's existing setup) -------------------------
    def configure(self, todo: list[str]):
        e = self.e
        if "codex" in todo:
            (e.home / ".codex").mkdir(parents=True, exist_ok=True)
            (e.home / ".codex" / "config.toml").write_text(
                'model = "fake-model"\nmodel_provider = "moth-e2e"\ncheck_for_update_on_startup = false\n\n'
                '[model_providers.moth-e2e]\nname = "moth e2e fake"\n'
                f'base_url = "{self.llm}/v1"\nwire_api = "responses"\nenv_key = "MOTH_E2E_KEY"\n', encoding="utf-8")
        if "pi" in todo:
            (e.home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
            (e.home / ".pi" / "agent" / "models.json").write_text(json.dumps({"providers": {"moth-e2e": {
                "baseUrl": f"{self.llm}/v1", "api": "openai-completions", "apiKey": "dummy",
                "models": [{"id": "fake-model"}]}}}, indent=2), encoding="utf-8")
        if "opencode" in todo:
            (e.home / ".config" / "opencode").mkdir(parents=True, exist_ok=True)
            (e.home / ".config" / "opencode" / "opencode.json").write_text(json.dumps({
                "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
                "model": "moth-e2e/fake-model",
                "provider": {"moth-e2e": {"npm": "@ai-sdk/openai-compatible", "name": "moth e2e fake",
                                          "options": {"baseURL": f"{self.llm}/v1", "apiKey": "dummy"},
                                          "models": {"fake-model": {"name": "fake"}}}}}, indent=2), encoding="utf-8")

    def install(self, todo: list[str]) -> bool:
        e = self.e
        code, so, se, _ = run([e.python, "-m", "moth", "hook", *todo, "--yes"], e.env(), e.repo, "moth_hook", e.out)
        log(so.strip() + ("\n" + se.strip() if se.strip() else ""))
        _, so2, _, _ = run([e.python, "-m", "moth", "hooks"], e.env(), e.repo, "moth_hooks", e.out)
        log(so2.strip())
        want = {"claude-code": e.home / ".claude" / "settings.json", "codex": e.home / ".codex" / "hooks.json",
                "pi": e.home / ".pi" / "agent" / "extensions" / "moth.ts",
                "opencode": e.home / ".config" / "opencode" / "plugins" / "moth.js"}
        ok = code == 0
        for h in todo:
            f = want[h]
            if not f.is_file():
                log(f"INSTALL FAIL {h}: {f} missing")
                ok = False
                continue
            shutil.copy(f, e.out / f"installed_{h}_{f.name}")
            if h in ("claude-code", "codex"):
                cmd = json.loads(f.read_text(encoding="utf-8"))["hooks"]["Stop"][-1]["hooks"][0]["command"]
                log(f"{h} hook command: {cmd}")
                if "moth repo" not in cmd or ("py env" not in cmd if self.a.spaced_python else False):
                    log(f"INSTALL FAIL {h}: the hook doesn't run the copy in 'moth repo' / the spaced python")
                    ok = False
        return ok

    # --- one harness -----------------------------------------------------------------------------------------
    def check(self, h: str, before: int, session: str | None, t_exit: float, must_be_before_exit: bool,
              grace: float = 3.0) -> dict:
        end = time.time() + grace
        new = records(self.record)[before:]
        while not new and time.time() < end:
            time.sleep(0.1)
            new = records(self.record)[before:]
        res = {"posts": new, "session_expected": session}
        problems = []
        if len(new) != 1:
            problems.append(f"expected 1 post, got {len(new)}")
        if new:
            r = new[0]
            if r["text"].strip() != REPLY:
                problems.append(f"text {r['text']!r}")
            if r["source"] != h:
                problems.append(f"source {r['source']!r}")
            if not session or r["session"] != session:
                problems.append(f"session {r['session']!r} != harness's {session!r}")
            if r["label"] != e_label(self.e):
                problems.append(f"label {r['label']!r} != {e_label(self.e)!r}")
            res["posted_before_exit"] = r["t"] <= t_exit
            res["post_minus_exit_s"] = round(r["t"] - t_exit, 3)
            if must_be_before_exit and r["t"] > t_exit:
                problems.append(f"posted {r['t'] - t_exit:.2f}s AFTER the CLI exited")
        res["problems"] = problems
        res["status"] = "pass" if not problems else "fail"
        return res

    def claude(self) -> dict:
        e, exe = self.e, shutil.which("claude")
        env = e.env(ANTHROPIC_BASE_URL=self.llm, ANTHROPIC_API_KEY="sk-ant-moth-e2e-dummy",
                    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1", DISABLE_TELEMETRY="1",
                    DISABLE_ERROR_REPORTING="1", CLAUDE_CODE_EFFORT_LEVEL="medium")
        before = len(records(self.record))
        code, so, se, t = run([exe, "-p", "hi", "--output-format", "json"], env, e.proj, "claude", e.out)
        session = None
        try:
            d = json.loads(so.strip().splitlines()[-1])
            session = d.get("session_id")
            log(f"  claude result: {d.get('result')!r} session {session}")
        except (ValueError, IndexError):
            log("  claude printed no JSON")
        res = self.check("claude-code", before, session, t, True)
        res["exit"] = code
        if code != 0:
            res["problems"].insert(0, f"claude exited {code}")
            res["status"] = "fail"
        return res

    def codex(self) -> dict:
        e, exe = self.e, shutil.which("codex")
        env = e.env(MOTH_E2E_KEY="dummy", RUST_LOG="warn")
        base = [exe, "exec", "--json", "--skip-git-repo-check", "--color", "never"]
        # control: the hook isn't approved (no trusted_hash in config.toml), so Codex must skip it
        before = len(records(self.record))
        code0, so0, _, _ = run(base + ["hi"], env, e.proj, "codex_unapproved", e.out)
        time.sleep(1.5)
        unapproved = records(self.record)[before:]
        before = len(records(self.record))
        code, so, se, t = run(base[:2] + ["--dangerously-bypass-hook-trust"] + base[2:] + ["hi"], env, e.proj,
                              "codex", e.out)
        session = None
        for line in so.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "thread.started":
                session = ev.get("thread_id")
        log(f"  codex thread {session}")
        res = self.check("codex", before, session, t, True)
        res["exit"], res["unapproved_exit"], res["unapproved_posts"] = code, code0, unapproved
        if code != 0:
            res["problems"].insert(0, f"codex exited {code}")
        if code0 != 0:
            res["problems"].append(f"the unapproved control run exited {code0}")
        if unapproved:
            res["problems"].append(f"the UNAPPROVED hook posted {len(unapproved)}x (Codex should skip it)")
        res["display"] = self.codex_display(base, env)
        res["problems"] += res["display"]["problems"]
        res["status"] = "pass" if not res["problems"] else "fail"
        return res

    def codex_display(self, base: list[str], env: dict) -> dict:
        """show_in_harness on: the Stop hook prints moth's reaction as {"systemMessage"} (the TUI draws it under the
        turn). Two turns of one thread (exec, then exec resume): the hook must have fetched moth's line for that
        thread both times, and that line must be in NO request body the model got, while turn 2's request does
        carry turn 1 (so the check isn't vacuous)."""
        e, bypass = self.e, base[:2] + ["--dangerously-bypass-hook-trust"] + base[2:]
        env = {**env, "MOTH_SHOW_IN_HARNESS": "1"}
        access = Path(str(self.record) + ".access")
        n_acc = len(access.read_text(encoding="utf-8").splitlines()) if access.exists() else 0
        n_body = len(self.bodies.read_text(encoding="utf-8").splitlines()) if self.bodies.exists() else 0
        res: dict = {"problems": []}
        code, so, _, _ = run(bypass + ["hi"], env, e.proj, "codex_display1", e.out)
        thread = next((ev.get("thread_id") for ev in map(_json_or_none, so.splitlines())
                       if ev and ev.get("type") == "thread.started"), None)
        if code != 0 or not thread:
            res["problems"].append(f"display turn 1: exit {code}, thread {thread!r}")
            return res
        code, _, _, _ = run(bypass + ["resume", thread, "and now?"], env, e.proj, "codex_display2", e.out)
        if code != 0:
            res["problems"].append(f"display turn 2 (resume): exit {code}")
        polls = [r["path"] for r in map(_json_or_none, access.read_text(encoding="utf-8").splitlines()[n_acc:])
                 if r and r["method"] == "GET" and r["path"].startswith("/api/says?source=codex&session=" + thread)]
        res["says_polls"] = len(polls)
        if len(polls) < 2:   # turn 1: a baseline + the queued reaction; turn 2: at least the late-lines check
            res["problems"].append(f"the hook fetched moth's line {len(polls)}x for thread {thread} (expected >= 2)")
        line = f"heard {e_label(e)}"   # fake_moth's reaction: "heard <label>: <first words>"
        bodies = [json.loads(l)["body"] for l in self.bodies.read_text(encoding="utf-8").splitlines()[n_body:]]
        res["model_requests"] = len(bodies)
        leaked = [i for i, b in enumerate(bodies) if line in b or "systemMessage" in b]
        if leaked:
            res["problems"].append(f"moth's line REACHED THE MODEL in request(s) {leaked}")
        if not any("and now?" in b and REPLY in b for b in bodies):
            res["problems"].append("turn 2's request doesn't carry turn 1 (the leak check would prove nothing)")
        log(f"  display: {len(polls)} polls, {len(bodies)} model requests, moth's line in none: {not leaked}")
        return res

    def pi(self) -> dict:
        e, exe = self.e, shutil.which("pi")
        sid = str(uuid.uuid4())
        env = e.env(PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1")
        before = len(records(self.record))
        code, so, se, t = run([exe, "-p", "--session-id", sid, "--provider", "moth-e2e", "--model", "fake-model",
                               "--no-tools", "hi"], env, e.proj, "pi", e.out)
        res = self.check("pi", before, sid, t, False)
        res["exit"] = code
        if code != 0:
            res["problems"].insert(0, f"pi exited {code}")
            res["status"] = "fail"
        return res

    def opencode(self) -> dict:
        e, exe = self.e, shutil.which("opencode")
        env = e.env(OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_SHARE="1")
        before = len(records(self.record))
        code, so, se, t = run([exe, "run", "--format", "json", "-m", "moth-e2e/fake-model", "hi"], env, e.proj,
                              "opencode", e.out, timeout=240)
        session = None
        for line in so.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            session = session or ev.get("sessionID") or (ev.get("part") or {}).get("sessionID")
        res = self.check("opencode", before, session, t, False)
        res["exit"] = code
        if code != 0:
            res["problems"].insert(0, f"opencode exited {code}")
            res["status"] = "fail"
        return res

    def main(self) -> int:
        a = self.a
        todo = [h for h in a.harness if h in HARNESSES]
        found = [h for h in todo if shutil.which(BIN[h])]
        for h in todo:
            if h not in found:
                self.results[h] = {"status": "skip", "problems": [f"`{BIN[h]}` not on PATH"]}
        self.setup()
        try:
            self.configure(found)
            installed = self.install(found)
            for h in found:
                if not installed:
                    self.results[h] = {"status": "fail", "problems": ["moth hook didn't install cleanly"]}
                    continue
                log(f"\n=== {h} ===")
                self.results[h] = {"claude-code": self.claude, "codex": self.codex, "pi": self.pi,
                                   "opencode": self.opencode}[h]()
                log(f"  -> {self.results[h]['status']} {self.results[h]['problems']}")
        finally:
            self.e.stop()
        (self.e.out / "results.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        log(f"\n{'harness':<12} {'status':<6} notes   (python: {'spaced venv' if a.spaced_python else 'plain'})")
        bad = False
        for h, r in self.results.items():
            soft = h in a.soft
            log(f"{h:<12} {r['status']:<6} {'(soft) ' if soft else ''}{'; '.join(r['problems'])}")
            bad |= r["status"] != "pass" and not soft
        if not a.keep and not a.out:
            shutil.rmtree(self.e.root, ignore_errors=True)
        else:
            log(f"logs in {self.e.out}")
        return 1 if bad else 0


def _json_or_none(line: str):
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def e_label(e: Env) -> str:
    return e.proj.name


def main():
    try:   # a Windows console / CI log may be cp1252: never die printing moth's check marks
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", nargs="+", default=list(HARNESSES))
    ap.add_argument("--soft", nargs="*", default=[], help="harnesses whose failure or absence doesn't fail the run")
    ap.add_argument("--spaced-python", action="store_true", help="run the hooks with a venv python in 'py env/'")
    ap.add_argument("--out", help="where logs and results.json go (kept)")
    ap.add_argument("--work", help="parent folder for the scratch root (default: the system temp folder)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch root")
    sys.exit(Runner(ap.parse_args()).main())


if __name__ == "__main__":
    main()
