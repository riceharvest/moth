"""Chat with moth in the terminal:  python -m moth.cli --model runs/sft1/model

Same engine as moth.serve (persistent home, the same little home-only shell). Commands: /reset, /home, /quit.
Tool output is printed with terminal escape codes stripped (a file in moth's home can't move your cursor).
"""
from __future__ import annotations

import argparse

from moth.serve import Moth, default_paths, strip_ansi

DIM, LAMP, OFF = "\033[2m", "\033[33m", "\033[0m"


def main():
    d = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d["model"])
    ap.add_argument("--tokenizer", default=d["tokenizer"])
    ap.add_argument("--home", default=d["home"], help="moth's home folder (kept between runs)")
    ap.add_argument("--pool-file", default=d["pool"])
    ap.add_argument("--seed", type=int, default=3, help="which home moth starts with")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--ctx-budget", type=int, default=None, help="max tokens of chat moth keeps in mind")
    ap.add_argument("--max-turn-tokens", type=int, default=200)
    ap.add_argument("--max-tool-rounds", type=int, default=3)
    a = ap.parse_args()
    from moth import weights
    weights.ensure_default(a.model)   # pip install, first run: download the model files
    moth = Moth(a)
    print(f"{DIM}moth is awake. home: {moth.home.root}  (/reset, /home, /quit){OFF}")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            moth.reset()
            print(f"{DIM}(new chat){OFF}")
            continue
        if text == "/home":
            for d in moth.home.tree():
                print(f"{LAMP}{d['dir']}/{OFF} " + "  ".join(d["files"]))
            continue
        for ev in moth.chat(text):
            if "say" in ev:
                print(f"{LAMP}moth>{OFF} {strip_ansi(ev['say'])}")
            else:
                a, t = ev["args"], ev["tool"]
                cmd = {"bash": f"$ {a.get('command', '')}", "read": f"$ cat {a.get('path', '')}",
                       "write": f"✎ {a.get('path', '')}"}.get(t, t)
                body = str(a.get("content", "") if t == "write" else ev["result"])
                print(f"{DIM}  {strip_ansi(str(cmd))}\n  " + strip_ansi(body).replace("\n", "\n  ")[:800] + OFF)


if __name__ == "__main__":
    main()
