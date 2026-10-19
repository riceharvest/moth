"""moth card: who moth is. The identity comes only from this card (the system prompt); the character underneath stays.

  python -m moth card                          show the card (and where it comes from)
  python -m moth card set "you are a cat named bo"   any one-line identity
  python -m moth card set --file card.json     a card file: {"name": "pip", "kind": "a houseplant",
                                               "traits": ["you love puns"]} or {"lines": ["you are pip, ..."]}
  python -m moth card set moth                 back to the default moth card (also: moth card reset)
  python -m moth card set blank                no identity at all, just a little chat buddy
  [--port N]                                   the running moth to switch (default 4747)

Saved in ~/.moth/config.json ("card"); a running moth switches at once (POST /api/card, which starts a new chat: the
system prompt changed) and the chat page and the pet show the new name. serve --card beats the setting.
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from moth import cards, config


def prog() -> str:
    return "moth" if os.path.basename(sys.argv[0]) in ("moth", "moth.exe") else "python -m moth"


def token_path() -> str:   # = serve.token_path(), without importing serve (torch)
    return os.environ.get("MOTH_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".moth", "token")


def describe(value) -> str:
    if value in (None, "moth"):
        return "moth (the default card: " + " ".join(cards._ident(cards.MOTH_CARD)) + ")"
    if value == "blank":
        return "blank (no identity: just a little chat buddy)"
    if isinstance(value, dict):
        return "a card file: " + " ".join(cards._ident(value))
    return value


def push(value, port: int, timeout: float = 5.0) -> str:
    """Switch a running moth to value. Returns what happened, for the user."""
    try:
        with open(token_path(), encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        return "moth isn't running (no token yet): it uses the card from its next start"
    body = json.dumps({"card": value if value is not None else "moth"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/card", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-Moth-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            got = json.loads(r.read() or b"{}")
        return f"the running moth switched (a new chat): it's {got.get('card', {}).get('name') or 'the card'} now"
    except urllib.error.HTTPError as e:
        try:
            why = json.loads(e.read() or b"{}").get("error") or e.reason
        except ValueError:
            why = e.reason
        return f"the running moth on port {port} said no ({e.code}: {why}); it uses the card from its next start"
    except (OSError, ValueError):
        return "moth isn't running: it uses the card from its next start"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"{prog()} card", description="who moth is (its identity card)")
    ap.add_argument("action", nargs="?", choices=["show", "set", "reset"], default="show")
    ap.add_argument("text", nargs="*", help='a one-line identity, "moth" or "blank"')
    ap.add_argument("--file", default=None, help="a JSON card file")
    ap.add_argument("--port", type=int, default=4747)
    a = ap.parse_args(argv)
    if a.action == "show":
        value, source = config.lookup("card")
        print(f"card: {describe(value)}   ({'from ' + config.path() if source == 'file' else 'default'})")
        return 0
    if a.action == "reset":
        value = "moth"
    elif a.file:
        if a.text:
            print("give a one-line identity or --file, not both", file=sys.stderr)
            return 2
        try:
            value = cards.card_from_file(a.file)
        except (OSError, ValueError) as e:
            print(f"can't use {a.file}: {e}", file=sys.stderr)
            return 2
        value = {k: v for k, v in value.items() if k in ("id", "name", "kind", "traits", "lines", "refs", "about")}
    else:
        value = config.parse_card(" ".join(a.text))
        if value is None:
            print(f'usage: {prog()} card set "you are a cat named bo" | --file card.json | moth | blank',
                  file=sys.stderr)
            return 2
        if isinstance(value, str) and value not in ("moth", "blank"):
            try:
                cards.validate(cards.text_card(value))
            except ValueError as e:
                print(f"can't use that: {e}", file=sys.stderr)
                return 2
    try:
        config.write("card", value)
    except OSError as e:
        print(f"couldn't write {config.path()}: {e}", file=sys.stderr)
        return 1
    print(f"card = {describe(value)}   (saved in {config.path()})")
    print(push(None if value == "moth" else value, a.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
