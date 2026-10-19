"""Chat with moth in the terminal:  python -m moth.cli  [serve's model options]

Same engine as moth.serve (the default model, its card, the phrase memory); pure chat. Commands: /reset (new chat),
/card (who moth is), /quit. Replies are printed with terminal escape codes stripped.
"""
from __future__ import annotations

from moth.serve import default_threads, make_engine, parse_args, strip_ansi

DIM, LAMP, OFF = "\033[2m", "\033[33m", "\033[0m"


def main(argv=None):
    a = parse_args(argv)
    from moth import backends, weights
    if a.manifest.backend in ("tiny-moe", "tiny-agent"):
        weights.ensure_default(a.model, legacy=a.legacy)   # pip install, first run: download the model files
        backends.prepare(a.manifest)
    import torch
    torch.set_num_threads(a.threads or default_threads())
    moth = make_engine(a)
    who = moth.identity_view()
    name = who["name"]
    print(f"{DIM}{name} is awake ({who['text'] or 'no card'}).  (/reset, /card, /quit){OFF}")
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
            moth.forget()
            print(f"{DIM}(new chat){OFF}")
            continue
        if text == "/card":
            print(f"{DIM}{moth.identity_view()['text'] or 'no card'}{OFF}")
            continue
        for ev in moth.chat(text):
            if "say" in ev:
                print(f"{LAMP}{name}>{OFF} {strip_ansi(ev['say'])}")


if __name__ == "__main__":
    main()
