"""python -m moth : wake moth up and print how to hook up your coding agents.

All moth.serve options work here too (python -m moth --help). The chat page is served at the printed URL but only
opened in the browser with --open (or --browser); --no-browser is accepted and does nothing (the old opt-out).
  python -m moth setup [--yes] [--dry-run]       start here: install to a running moth, step by step (moth/setup.py)
  python -m moth config [KEY VALUE]              moth's settings in ~/.moth/config.json: show_in_harness, pet_skin,
                                                 update_check, hide_in_fullscreen, chirp, muted_sources, pet_screen,
                                                 quiet_hours, model (moth/config.py)
  python -m moth update [--check | --rollback]   update this Hugging Face download in place (moth/update.py)
  python -m moth --version                       version and Hub revision
  python -m moth hooks | hook | unhook           see / install / remove the coding-agent hooks (moth/hooks.py)
  python -m moth doctor                          one-screen health check (moth/doctor.py)
  python -m moth autostart on|off|status         start the desktop pet at login (moth/autostart.py)
  python -m moth forget [--all [--token]]        wipe moth's logs (and home folder) (moth/forget.py)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
INTEGRATIONS = os.path.join(HERE, "integrations")
CODE = os.path.dirname(HERE)


def command(*parts: str) -> str:
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


def run_here(*args: str, module: str = "moth") -> str:
    """How to run `python -m <module> <args>` on this install: anywhere for a pip install and the installers' app
    (the `moth` command, or their Python / executable with -m), from code/ for a download or the checkout."""
    from moth.backends import compat   # stdlib only
    if compat.install_kind() in ("pip", "app"):
        return command("moth", *args) if module == "moth" else command(sys.executable, "-m", module, *args)
    return f"cd {command(CODE)} && {command(sys.executable, '-m', module, *args)}"


def snippets() -> str:
    """Ready-to-paste setup for the six harnesses, with this install's real paths."""
    py = sys.executable
    from .hooks import harness_command, hook_entry   # stdlib only; the entry `moth hook` writes (see HOOK_TIMEOUT)
    hook = lambda name: json.dumps({"hooks": {"Stop": [{"hooks": [   # noqa: E731
        hook_entry(harness_command(name, py))]}]}})
    home = os.path.expanduser("~")
    p = lambda *x: os.path.join(INTEGRATIONS, *x)   # noqa: E731
    return f"""
moth can watch your coding agent and react to its replies (one way: the agent never hears you or moth).
Let moth set it up:  {run_here('hook', '--all')}   (asks first; backs up and merges configs;
`{run_here('hooks').split(' && ')[-1]}` shows what's installed). Or paste what fits your harness
(details: {p('README.md')}):

  Claude Code   merge into {os.path.join(home, '.claude', 'settings.json')}
{_indent(hook('claude_code.py'), 4)}
                moth's line above the prompt, with a field to talk to it:
                  claude --plugin-dir {command(p('claude_code_mod'))}

  Codex CLI     merge into {os.path.join(home, '.codex', 'hooks.json')}, then approve it once in `codex` (/hooks;
                again after any edit of hooks.json: until then even `codex exec` skips it)
{_indent(hook('codex.py'), 4)}

  Hermes        {command('cp', '-r', p('hermes', 'moth'), os.path.join(home, '.hermes', 'plugins', 'moth'))}
                hermes plugins enable moth

  Pi            {command('cp', p('pi', 'moth.ts'), os.path.join(home, '.pi', 'agent', 'extensions', 'moth.ts'))}

  OpenCode      {command('cp', p('opencode', 'moth.js'), os.path.join(home, '.config', 'opencode', 'plugins', 'moth.js'))}

  dsh           add to {os.path.join(home, '.dsh', 'profiles', '<profile>', 'cordis.patch.yml')}
                  - insert: [{{ id: moth, name: {json.dumps(p('dsh', 'moth.mjs'))} }}]

  any harness   watch and talk to moth in a side terminal:
                  {run_here(module='moth.tail')}

  The adapters only tell moth about each turn; the desktop pet shows its reaction. To also see moth's reply
  inside Pi (widget) and OpenCode (toast) too, from the next turn on:
                  {run_here('config', 'show_in_harness', 'on')}
"""


USAGE = """usage: python -m moth [serve options] [--open]
                                        wake moth up (the default): start the local server on 127.0.0.1:4747
                                        and print the agent hook snippets; --open (or --browser) also opens
                                        the chat page in your browser
       python -m moth setup [--yes] [--dry-run] [--url URL]
                                        start here: from install to a running moth (packages, weights, hooks,
                                        autostart, the pet); asks first, safe to run again
       python -m moth config [show_in_harness on|off | pet_skin <id|folder|moth> | update_check on|off]
                                        print or change moth's settings (~/.moth/config.json); show_in_harness
                                        shows moth's replies inside Pi / OpenCode (default off), pet_skin is
                                        the desktop pet's look (a Petdex pet id, default moth), update_check
                                        is the pet's daily "new moth out" bubble (default on)
       python -m moth config [hide_in_fullscreen on|off | chirp on|off | muted_sources codex,pi|none |
                              pet_screen <name>|any | quiet_hours 23:00-08:00|off]
                                        the pet: hide while an app is fullscreen (default on), a soft sound when
                                        an agent is done (default off), harnesses that pop no bubble (default
                                        none), stay on one screen (default any), quiet hours (default off)
       python -m moth config model <folder | HF repo id | openai:URL#model | default>
                                        the model behind moth (default: its own); see
                                        moth/docs/MODEL_BACKENDS.md. --model beats it for one run
       python -m moth update [--check | --rollback] [--yes] [--repo ID] [--revision REV]
                                        update this Hugging Face download in place (python -m moth update --help)
       python -m moth hooks             which coding agents are on this machine and whether moth is hooked in
       python -m moth hook <harness ...> | --all [--project DIR] [--yes] [--dry-run]
                                        hook moth into Claude Code, Codex, Hermes, Pi, OpenCode or dsh (backs up
                                        and merges their configs; python -m moth hook --help)
       python -m moth unhook <harness ...> | --all [--yes] [--dry-run]
                                        remove exactly what `hook` added, nothing else
       python -m moth doctor            one-screen health check with a fix hint per problem (no model load)
       python -m moth autostart on|off|status
                                        start the desktop pet (which wakes the server) when you log in
       python -m moth forget [--all [--token]] [--yes]
                                        wipe moth's logs; --all also resets its home folder (files, diary);
                                        --all --token also the token and settings. Stop moth first
       python -m moth --version         print moth's version and the Hub revision this folder is on
       python -m moth --help            this help, then the serve options below
"""


def _indent(text: str, n: int) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["config"]:   # first: never falls through to serve (torch, the model, a server)
        from moth import config
        sys.exit(config.main(argv[1:]))
    if argv[:1] == ["update"]:
        from moth import update
        sys.exit(update.main(argv[1:]))
    if argv[:1] in (["hooks"], ["hook"], ["unhook"]):   # before --help / --version: `moth hook --help` is hook's
        from moth import hooks
        sys.exit(hooks.main(argv[0], argv[1:]))
    if argv[:1] in (["setup"], ["doctor"], ["autostart"], ["forget"]):
        import importlib
        sys.exit(importlib.import_module(f"moth.{argv[0]}").main(argv[1:]))
    if "--version" in argv:
        from moth import update
        print(update.version_line())
        return
    if any(x in ("-h", "--help") for x in argv):
        print(USAGE + "\nserve options (plus --open / --browser: open the chat page in your browser):", flush=True)
        try:
            from moth import serve   # noqa: F401  (needs torch for its options)
        except ImportError as e:
            print(f"  (can't list them: {e}; pip install torch and -r requirements.txt)")
            return
    browser = any(x in ("--open", "--browser") for x in argv)
    argv = [x for x in argv if x not in ("--open", "--browser", "--no-browser")]   # --no-browser: old opt-out, no-op
    if not any(x in ("-h", "--help") for x in argv):
        print(snippets(), flush=True)
    from moth import serve   # torch loads here, after the snippets are on screen

    def ready(url: str):
        if browser:
            threading.Timer(0.3, webbrowser.open, [url]).start()
        else:
            print(f"chat page: {url}  (not opened; python -m moth --open opens it)", flush=True)

    serve.main(argv, on_ready=ready)


if __name__ == "__main__":
    main()
