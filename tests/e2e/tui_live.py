"""Live checks in the real interactive TUIs, local: the Claude Code band, and Codex's hook approval.

  python tests/e2e/tui_live.py start --out DIR [--tui claude|codex]
        scratch home, fake model, fake moth, the Stop hook installed, and in a detached tmux session on its own socket
        (-L moth-band-e2e, window "band"): `claude --plugin-dir <copy>/claude_code_mod` (default), or `codex` (its
        startup hook review, to approve moth's hook the way a person does); prints DIR/state.json
  python tests/e2e/tui_live.py stop --out DIR      kill that tmux server and the two fakes

Codex: after approving in the TUI, `python tests/e2e/tui_live.py exec --out DIR` runs `codex exec` (no bypass flag)
in the same scratch home and prints what moth received.

In between, drive it with tmux (`tmux -L moth-band-e2e send-keys -t band ...`, `capture-pane -p`). What to check:
the band's GETs in DIR/moth_record.jsonl.access carry `session=<id>`, the Stop hook's POST in DIR/moth_record.jsonl
carries `session` = the same id, and the band shows this session's line, still after another session's reaction is
posted last (POST /api/agent with another session id, token in DIR/home/.moth/token). Needs tmux and `claude` on
PATH. Never touches the real ~/.claude or moth's port 4747: home, config and ports are scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e2e  # noqa: E402

SOCK, NAME = "moth-band-e2e", "band"


def start(out: Path, tui: str):
    a = types.SimpleNamespace(work=str(out), out=str(out), spaced_python=False, keep=True)
    r = run_e2e.Runner(a)
    r.setup()
    e = r.e
    harness = "claude-code" if tui == "claude" else "codex"
    r.configure([harness])
    if not r.install([harness]):
        raise SystemExit("hook install failed")
    key = "sk-ant-moth-e2e-dummy-key-000000000000"
    (e.home / ".claude").mkdir(parents=True, exist_ok=True)
    # first-run screens answered ahead: onboarding done, the custom key approved, the project folder trusted
    (e.home / ".claude" / ".claude.json").write_text(json.dumps({
        "hasCompletedOnboarding": True, "theme": "dark", "numStartups": 5,
        "customApiKeyResponses": {"approved": [key[-20:]], "rejected": []},
        "projects": {str(e.proj): {"hasTrustDialogAccepted": True, "hasCompletedProjectOnboarding": True}}}),
        encoding="utf-8")
    env = e.env(ANTHROPIC_BASE_URL=r.llm, ANTHROPIC_API_KEY=key, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                DISABLE_AUTOUPDATER="1", DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", TERM="xterm-256color")
    env.pop("NO_COLOR", None)
    env.pop("TMUX", None)
    mod = e.repo / "moth" / "integrations" / "claude_code_mod"
    cmd = ["tmux", "-L", SOCK, "-f", os.devnull, "new-session", "-d", "-s", NAME, "-x", "150", "-y", "40", "-c",
           str(e.proj)]
    for k in ("HOME", "CLAUDE_CONFIG_DIR", "MOTH_URL", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "XDG_CONFIG_HOME",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY",
              "DISABLE_ERROR_REPORTING", "TERM"):
        cmd += ["-e", f"{k}={env[k]}"]
    cmd += ["-e", "MOTH_E2E_KEY=dummy", "-e", f"CODEX_HOME={env['CODEX_HOME']}"]
    cmd += ["claude", "--plugin-dir", str(mod)] if tui == "claude" else [shutil.which("codex")]
    subprocess.run(cmd, env=env, check=True)
    state = {"tui": tui, "llm_port": r.llm.rsplit(":", 1)[1], "pids": [p.pid for p in e.procs], "moth_url": e.moth_url, "llm": r.llm, "home": str(e.home),
             "proj": str(e.proj), "record": str(r.record), "token": (e.home / ".moth" / "token").read_text().strip(),
             "tmux": f"tmux -L {SOCK}", "target": NAME}
    (out / "state.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


def codex_exec(out: Path):
    st = json.loads((out / "state.json").read_text())
    home, proj = Path(st["home"]), Path(st["proj"])
    env = {k: v for k, v in os.environ.items() if k not in run_e2e.SCRUB}
    env.update(HOME=str(home), CODEX_HOME=str(home / ".codex"), MOTH_URL=st["moth_url"], MOTH_E2E_KEY="dummy",
               PWD=str(proj))
    before = len(run_e2e.records(Path(st["record"])))
    p = subprocess.run([shutil.which("codex"), "exec", "--json", "--skip-git-repo-check", "hi"], env=env, cwd=proj,
                       capture_output=True, text=True, timeout=120)
    print("exit", p.returncode, [l for l in p.stdout.splitlines() if "thread.started" in l])
    print("moth got:", run_e2e.records(Path(st["record"]))[before:])


def stop(out: Path):
    subprocess.run(["tmux", "-L", SOCK, "kill-server"])
    for pid in json.loads((out / "state.json").read_text())["pids"]:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("start", "stop", "exec"))
    ap.add_argument("--tui", choices=("claude", "codex"), default="claude")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "start":
        start(out, a.tui)
    else:
        (stop if a.what == "stop" else codex_exec)(out)


if __name__ == "__main__":
    main()
