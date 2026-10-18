"""moth-hook: the installers' coding-agent hook runner, a separate small executable (packaging/moth.spec).

  moth-hook hook-run <claude_code.py | codex.py | .../integrations/x.py> [args]

Runs this app's own moth/integrations/<name> (always the bundled copy, whatever folder the command names: an
AppImage's mount moves, an old install's path may be gone). Stdlib only, with its own tiny module archive: a hook runs
on every agent turn and Claude Code gives it 3 s, and the big moth executable (torch and Qt in its archive) starts
several times slower. The AppImage's AppRun sends `hook-run` here too; `moth hook-run` (moth_app.py) does the same.
"""
import os
import sys

# what the hook scripts import (they run from source, so PyInstaller can't see it): moth.spec's hiddenimports
HOOK_IMPORTS = ("json", "socket", "time", "urllib.request", "urllib.error", "urllib.parse", "http.client")


def bundle() -> str:
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hook_run(argv: list) -> int:
    import runpy
    if not argv:
        print("usage: moth-hook hook-run <claude_code.py | codex.py> [args]", file=sys.stderr)
        return 2
    name = os.path.basename(argv[0].replace("\\", "/"))
    path = os.path.join(bundle(), "moth", "integrations", name)
    if not os.path.isfile(path):
        print(f"moth hook-run: no hook {name!r} in this app", file=sys.stderr)
        return 0   # never fail the harness's turn
    sys.argv = [path] + list(argv[1:])
    sys.path.insert(0, os.path.dirname(path))
    runpy.run_path(path, run_name="__main__")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    return hook_run(argv[1:] if argv[:1] == ["hook-run"] else argv)


if __name__ == "__main__":
    sys.exit(main())
