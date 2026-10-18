"""Screenshot the real desktop pet with a reaction in its bubble, for a by-eye check on a CI runner's desktop.

  python tests/e2e/pet_shot.py --out DIR

Starts tests/e2e/fake_moth.py (serve.py's real handler; token in a scratch home), launches the real
`python -m moth.pet --no-server --no-wander` against it (a real window: no QT_QPA_PLATFORM=offscreen), posts one
Claude Code turn to /api/agent as the Stop hook would, and takes full-screen screenshots before and after (Windows:
.NET CopyFromScreen via PowerShell; macOS: screencapture; elsewhere: Qt's screen grab). The pet sits in the
bottom-right corner, so the bubble ("moth> heard moth-repo: ...", captioned "claude · moth-repo") and the tray area
should both be in the shot. Writes DIR/pet-*.png, DIR/pet.log and DIR/fake_moth.log; exit 1 if the pet died.
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
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PS_SHOT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save($env:SHOT_PATH, [System.Drawing.Imaging.ImageFormat]::Png)
"""
QT_SHOT = ("import sys; from PySide6.QtGui import QGuiApplication; app = QGuiApplication(sys.argv); "
           "s = app.primaryScreen(); print(s.geometry()); s.grabWindow(0).save(sys.argv[1])")


def shot(path: Path, env: dict) -> str:
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_SHOT], env={**env, "SHOT_PATH": str(path)},
                           capture_output=True, text=True, timeout=60)
    elif sys.platform == "darwin":
        r = subprocess.run(["screencapture", "-x", str(path)], capture_output=True, text=True, timeout=60)
    else:
        xcb = {"QT_QPA_PLATFORM": "xcb"} if env.get("DISPLAY") else {}   # the X screen the pet is on (it asks xcb)
        r = subprocess.run([sys.executable, "-c", QT_SHOT, str(path)], env={**env, **xcb}, capture_output=True,
                           text=True, timeout=60)
    msg = f"{path.name}: exit {r.returncode} {r.stdout.strip()} {r.stderr.strip()}"[:400]
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=float, default=8.0, help="seconds for the pet to start before the post")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="moth pet home "))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOTH_") and k != "QT_QPA_PLATFORM"}
    env.update(HOME=str(home), USERPROFILE=str(home) if os.name == "nt" else env.get("USERPROFILE", ""),
               PYTHONIOENCODING="utf-8")
    token_file = home / ".moth" / "token"
    info = out / "fake_moth.json"
    info.unlink(missing_ok=True)
    procs = []
    try:
        procs.append(subprocess.Popen([sys.executable, str(HERE / "fake_moth.py"), "--info", str(info), "--token-file",
                                       str(token_file), "--record", str(out / "pet_record.jsonl")],
                                      stdout=open(out / "fake_moth.log", "w"), stderr=subprocess.STDOUT))
        for _ in range(300):
            if info.exists():
                break
            time.sleep(0.1)
        url = f"http://127.0.0.1:{json.loads(info.read_text())['port']}"
        pet = subprocess.Popen([sys.executable, "-m", "moth.pet", "--no-server", "--no-wander", "--size", "96",
                                "--url", url, "--token-file", str(token_file)], cwd=str(ROOT), env=env,
                               stdout=open(out / "pet.log", "w"), stderr=subprocess.STDOUT)
        procs.append(pet)
        time.sleep(a.wait)
        shot(out / "pet-0-before.png", env)
        body = json.dumps({"text": "Fixed the login bug. All tests pass.", "source": "claude-code",
                           "session": "e2e-pet-1", "label": "moth-repo"}).encode()
        req = urllib.request.Request(url + "/api/agent", data=body, headers={
            "Content-Type": "application/json", "X-Moth-Token": token_file.read_text().strip()})
        print("POST /api/agent ->", urllib.request.urlopen(req, timeout=5).read().decode(), flush=True)
        for i, wait in enumerate((2.5, 2.5, 4.0), 1):
            time.sleep(wait)
            shot(out / f"pet-{i}-after.png", env)
        alive = pet.poll() is None
        print("pet alive at the end:", alive, flush=True)
        return 0 if alive else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
