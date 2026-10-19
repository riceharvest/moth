"""A card registry that grows, and versioned frozen eval sets.

  python -m moth.rl.registry list
  python -m moth.rl.registry add --line "you are a cat named bo"          # one-line identity, as a user writes it
  python -m moth.rl.registry add --file my_cards.json                     # a card dict or a list of them
  python -m moth.rl.registry add --teacher 40 --base-url http://HOST:8000/v1   # teacher-written one-liners
  python -m moth.rl.registry add-phrasing "you go by {name}; you're {kind}."
  python -m moth.rl.registry freeze v2                                    # held-out snapshot -> pool/eval_v2.json

Cards and phrasings = the built-ins of moth/cards.py + the append-only files pool/cards.ndjson and
pool/phrasings.ndjson (one JSON object per line, "#" lines are comments; never edit or delete a line, only append).

Split (train / heldout), per entry:
  * pinned built-ins: build_pool(), MOTH_CARD, BLANK_CARD and PHRASINGS are train (the gist-spec SFT data used them);
    heldout_cards() and HELDOUT_PHRASINGS are heldout. So the SFT data and the existing eval stay valid.
  * appended entries: heldout if sha256("card:<id>") (or "phrasing:<id>") falls in the lowest HELDOUT_PCT of 100
    buckets, else train. Adding cards therefore auto-assigns them, and the assignment never changes later.
  * an appended card whose kind noun or name is one of the held-out kinds/names (HELDOUT_KINDS, HELDOUT_HUMAN) is
    forced heldout, so a new "hedgehog named spike" can't leak a held-out identity into training. A card named like
    one of the eval's friends (EVAL_FRIEND_NAMES) is refused: recall probes would get muddy.
Ids: a one-line card is "u" + sha256(line)[:10]; a phrasing is "p" + sha256(template)[:10]; a JSON card keeps its id
(or gets "u" + the hash of its identity fields).

Frozen eval sets: freeze(version) writes pool/eval_<version>.json with fully resolved episode specs (card dict,
rendered system prompt, persona, goal, topic, seed, cleaned agent messages, scripted lines), plus their sha256.
load_eval() checks the hash; an existing version is never overwritten. Scores on eval_v1 stay comparable across
releases however much the pool grows; freeze a new version when the held-out pool has grown enough to matter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from moth import cards as C  # noqa: E402
from moth.agent import AGENT_SCENARIOS, AGENT_USER_PERSONAS, clean_agent_message  # noqa: E402
from moth.persona import TALK_SCENARIOS, TOPICS, USER_PERSONAS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool")
EVAL_AGENT_MSGS = os.path.join(ROOT, "data", "eval_agent_msgs.json")
HELDOUT_PCT = 10
SCENARIOS = ("talk", "agent", "recall")   # a pure chatbot: no home/tools scenario (eval_v1/v2 still have one)
SCENARIO_MIX = {"talk": 0.6, "agent": 0.28, "recall": 0.12}
LEGACY_EVALS = {"v1", "v2"}   # frozen before the no-tools spec: stale system prompts and "home" specs
TURN_CHOICES = [3, 4, 5, 6, 6, 8]
RECALL_FACTS = [("your name (make one up)", "your name"), ("your pet's name and what animal it is (make it up)",
                "your pet"), ("your favourite food", "your favourite food"), ("your favourite colour",
                "your favourite colour"), ("where you're going this weekend", "your weekend plan"),
                ("what your job is", "your job")]   # = gen_dialogs.RECALL_FACTS (subset; that module needs openai)
P_FREE = 0.4   # share of sampled cards phrased with a free-form phrasing


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def bucket(key: str) -> int:
    """Stable 0..99 bucket (sha256, never the salted hash())."""
    return int(_h(key)[:8], 16) % 100


def phrasing_id(tmpl: str) -> str:
    return "p" + _h(tmpl)[:10]


def line_card_id(line: str) -> str:
    return "u" + _h(" ".join(line.split()).lower())[:10]


ADJ = ("small", "little", "tiny")


def kind_core(kind: str) -> str:
    """'a lighthouse keeper's cat' -> "lighthouse keeper's cat", 'a small cat' -> 'cat'."""
    return " ".join(w for w in kind.split()[1:] if w not in ADJ)


HELD_PHRASES = [kind_core(k) for k, *_ in C.HELDOUT_KINDS]
HELD_NAMES = {n for _, ns, _ in C.HELDOUT_KINDS for n in ns} | set(C.HELDOUT_HUMAN)


def _has(phrase: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"(?:s|es)?\b", text) is not None


def collides_heldout(card: dict) -> bool:
    """The card names a held-out kind (the whole phrase: 'weather vane', not just 'vane') or a held-out name."""
    text = " ".join([card.get("kind") or ""] + list(card.get("lines") or [])).lower()
    names = {(card.get("name") or "").lower()}
    names |= set(re.findall(r"\b(?:named|called|name is|name:)\s+([a-z]+)", text))
    return any(_has(p, text) for p in HELD_PHRASES) or bool(names & HELD_NAMES)


def card_split(card: dict) -> str:
    if card.get("split") in ("train", "heldout"):
        return card["split"]
    if collides_heldout(card):
        return "heldout"
    return "heldout" if bucket("card:" + card["id"]) < HELDOUT_PCT else "train"


def phrasing_split(p: dict) -> str:
    if p.get("split") in ("train", "heldout"):
        return p["split"]
    return "heldout" if bucket("phrasing:" + p["id"]) < HELDOUT_PCT else "train"


END_NP = r"(?!\s+(?!named|called|who|that|with|and|living|from|in|on|at|but)[a-z])"


def kind_of(line: str) -> str | None:
    """A known kind the line names as what it is: 'you are a grumpy old kettle who...' -> 'a kettle'. Longest kinds
    first ('a rubber duck' before 'a duck'); the noun must end its phrase ('a lighthouse keeper' is no lighthouse)."""
    t = line.lower()
    kinds = sorted([k for k, *_ in C.KINDS + C.HELDOUT_KINDS] + ["a moth"], key=lambda k: -len(kind_core(k)))
    for k in kinds:
        if re.search(r"\b(?:a|an)\s+(?:[a-z']+\s+){0,2}?" + re.escape(kind_core(k)) + r"s?\b" + END_NP, t):
            return k
    return None


def card_from_line(line: str) -> dict:
    """A one-line identity -> a card. The line goes into the system prompt verbatim (cards.text_card); kind is filled
    from a known kind it names (so describe(), which the judge's label prompt uses, says what the card is), else
    cards.describe quotes the line itself."""
    line = " ".join(line.split())
    c = C.text_card(line)
    c["id"] = line_card_id(line)
    c["kind"] = kind_of(line)
    if not c.get("name") and not c["kind"]:   # else describe() says "no particular identity" and skips the line
        c["about"] = "a chat buddy"
    return c


def _norm_card(card: dict) -> dict:
    card = dict(card)
    if not card.get("id"):
        ident = json.dumps({k: card.get(k) for k in ("name", "kind", "traits", "lines")}, sort_keys=True)
        card["id"] = "u" + _h(ident)[:10]
    card.setdefault("traits", [])
    card.setdefault("name", None)
    card.setdefault("kind", None)
    if "refs" not in card:
        card["refs"] = [w for w in re.findall(r"[a-z]{3,}", (card.get("kind") or "").lower())
                        if w not in ("small", "little", "tiny")]
    return card


FIELDS = {"kind", "Kind", "noun", "name", "Name"}   # phrasing fields (cards.phrase)
STALE = ("split", "folders", "home")   # card fields of the tools/home era: ignored when an old line is loaded


def _current(c: dict) -> dict:
    """An appended card line read under the current schema (lines are never rewritten; ids don't depend on these)."""
    c = {k: v for k, v in c.items() if k not in STALE}
    if str(c.get("about", "")).startswith("whatever its system prompt says"):   # cards.describe quotes lines now
        c.pop("about")
        if not c.get("name") and not c.get("kind"):
            c["about"] = "a chat buddy"
    return c


def _read_ndjson(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        s = line.strip()
        if s and not s.startswith("#"):
            try:
                out.append(json.loads(s))
            except ValueError as e:
                raise SystemExit(f"{path}:{i + 1}: not JSON ({e})") from e
    return out


class Registry:
    def __init__(self, root: str = POOL_DIR):
        self.root = root
        self.cards_path = os.path.join(root, "cards.ndjson")
        self.phrasings_path = os.path.join(root, "phrasings.ndjson")
        self._cards: list[dict] = []
        self._phr: list[dict] = []
        self.reload()

    # --- loading -----------------------------------------------------------------------------------------------
    def reload(self):
        builtin = ([{**C.MOTH_CARD, "split": "train", "source": "builtin"},
                    {**C.BLANK_CARD, "split": "train", "source": "builtin"}]
                   + [{**c, "split": "train", "source": "builtin_pool"} for c in C.build_pool()]
                   + [{**c, "split": "heldout", "source": "builtin_heldout"} for c in C.heldout_cards()])
        seen = {c["id"] for c in builtin}
        added = []
        for c in _read_ndjson(self.cards_path):
            if c["id"] not in seen:
                seen.add(c["id"])
                added.append(_current(c))   # appended: split by hash only
        self._cards = [{**c, "split": card_split(c)} for c in builtin + added]
        phr = ([{"id": phrasing_id(t), "tmpl": t, "split": "train", "source": "builtin"} for t in C.PHRASINGS]
               + [{"id": phrasing_id(t), "tmpl": t, "split": "heldout", "source": "builtin"}
                  for t in C.HELDOUT_PHRASINGS])
        pseen = {p["id"] for p in phr}
        for p in _read_ndjson(self.phrasings_path):
            if set(re.findall(r"{(\w+)}", p["tmpl"])) - FIELDS:   # e.g. {home}: retired with the no-home spec
                continue
            if p["id"] not in pseen:
                pseen.add(p["id"])
                phr.append({k: v for k, v in p.items() if k != "split"})
        self._phr = [{**p, "split": phrasing_split(p)} for p in phr]
        self._by_id = {c["id"]: c for c in self._cards}

    def cards(self, split: str | None = None) -> list[dict]:
        return [c for c in self._cards if split is None or c["split"] == split]

    def phrasings(self, split: str | None = None) -> list[dict]:
        return [p for p in self._phr if split is None or p["split"] == split]

    def card(self, cid: str) -> dict:
        return self._by_id[cid]

    # --- growing -----------------------------------------------------------------------------------------------
    def _append(self, path: str, obj: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

    def add_card(self, card: dict, source: str = "manual") -> dict | None:
        """Append one card (None if its id is already known). Raises ValueError for an eval friend's name."""
        card = _norm_card(card)
        if (card.get("name") or "").lower() in C.EVAL_FRIEND_NAMES:
            raise ValueError(f"card name {card['name']!r} is one of the eval friends' names (cards.EVAL_FRIEND_NAMES)")
        if card["id"] in self._by_id:
            return None
        card = {k: v for k, v in card.items() if k != "split"}
        card["source"] = card.get("source") or source
        card["added"] = card.get("added") or time.strftime("%Y-%m-%d")
        self._append(self.cards_path, card)
        self.reload()
        return self._by_id[card["id"]]

    def add_line(self, line: str, source: str = "line") -> dict | None:
        return self.add_card(card_from_line(line), source)

    def add_phrasing(self, tmpl: str, source: str = "manual") -> dict | None:
        bad = set(re.findall(r"{(\w+)}", tmpl)) - FIELDS
        if bad:
            raise ValueError(f"unknown fields {sorted(bad)}; use kind, Kind, noun, name, Name")
        pid = phrasing_id(tmpl)
        if any(p["id"] == pid for p in self._phr):
            return None
        self._append(self.phrasings_path, {"id": pid, "tmpl": tmpl, "source": source,
                                           "added": time.strftime("%Y-%m-%d")})
        self.reload()
        return next(p for p in self._phr if p["id"] == pid)

    # --- sampling ----------------------------------------------------------------------------------------------
    def phrase(self, card: dict, p: dict) -> dict | None:
        c = C.as_free(card, p["tmpl"])
        if c is None:
            return None
        return {**c, "id": f"{card['id']}~{p['id']}", "base_id": card["id"], "phrasing_id": p["id"],
                "split": "heldout" if "heldout" in (card["split"], p["split"]) else "train"}

    def sample_card(self, rng: random.Random, split: str = "train", p_free: float = P_FREE) -> dict:
        """A card for one episode: train -> moth ~8% / blank ~10% / a train card, ~40% phrased free-form with a
        train phrasing. heldout -> a held-out card, ~40% phrased with any phrasing (the pair counts as held-out)."""
        if split == "train":
            x = rng.random()
            if x < C.SHARES["moth"]:
                card = self.card("moth")
            elif x < C.SHARES["moth"] + C.SHARES["blank"]:
                return dict(self.card("blank"))
            else:
                card = rng.choice([c for c in self.cards("train") if c["id"] not in ("moth", "blank")])
        else:
            card = rng.choice(self.cards("heldout"))
        if (card.get("lines") is None or card["id"] == "moth") and rng.random() < p_free:
            src = {**card, "kind": card.get("kind") or ("a moth" if card["id"] == "moth" else None)}
            opts = [x for x in (self.phrase({**src, "split": card["split"]}, p) for p in self.phrasings(
                    "train" if split == "train" else None)) if x]
            if opts:
                c = rng.choice(opts)
                if card["id"] == "moth":
                    c["refs"] = C.MOTH_REFS
                return c
        return dict(card)


# --- episode specs ----------------------------------------------------------------------------------------------
def load_agent_msgs(path: str | None) -> list[str]:
    """Raw agent final messages: a JSON list of strings or of {"text": ...} (moth/agent_pool.json shape)."""
    if not path or not os.path.exists(path):
        return []
    d = json.load(open(path, encoding="utf-8"))
    return [x["text"] if isinstance(x, dict) else str(x) for x in d]


def eval_agent_texts() -> set[str]:
    return set(load_agent_msgs(EVAL_AGENT_MSGS))


MOCK_AGENT_MSGS = [   # tiny built-in stand-ins for the training pool (mock runs, tests)
    "Fixed the login redirect bug in `auth/session.py`; all 42 tests pass.",
    "The deploy failed: the build step ran out of memory. I can retry with a smaller worker count, want me to?",
    "Should I also migrate the old records, or leave them as they are?",
    "I couldn't reproduce the crash locally. Can you share the logs from the failing run?",
    "Sorry, my earlier change broke the CSV export. I reverted it and added a test; everything is green again.",
    "The box is still booting. I'll check again in a minute.",
    "Done: dark mode is in. One caveat: the charts still use the light palette.",
    "I found the cause of the memory leak (an unbounded cache). Fixing it needs a schema change; OK to proceed?",
]


_WARNED: list = []


def make_spec(card: dict, scenario: str, rng: random.Random, seed: int, agent_msgs: list[str] | None = None,
              sid: str | None = None, turns: int | None = None) -> dict:
    """A fully resolved episode spec (JSON-serialisable): everything the simulator and policy need."""
    spec = {"id": sid or f"{card['id']}:{scenario}:{seed}", "seed": seed, "scenario": scenario, "card": card,
            "card_id": card["id"], "system": C.student_system(card), "topic": rng.choice(TOPICS),
            "turns": turns or rng.choice(TURN_CHOICES), "script": None, "agent_msgs": [], "recall": None}
    if scenario == "agent":
        spec["persona"], spec["goal"] = rng.choice(AGENT_USER_PERSONAS), rng.choice(AGENT_SCENARIOS)
        if not agent_msgs and not _WARNED:
            _WARNED.append(1)
            print("moth.rl: no agent messages given: agent episodes use the 8 built-in stand-ins (MOCK_AGENT_MSGS); "
                  "pass agent_msgs=train_agent_msgs(<agent_pool.json>) for training", file=sys.stderr, flush=True)
        pool = agent_msgs or MOCK_AGENT_MSGS
        k = rng.randint(1, 3)
        spec["agent_msgs"] = [clean_agent_message(t) for t in rng.sample(pool, min(k, len(pool)))]
        spec["follow"] = [rng.choice([0, 1, 1, 2]) for _ in spec["agent_msgs"]]   # sim turns after each agent msg
        spec["turns"] = len(spec["agent_msgs"]) + sum(spec["follow"])
    else:
        spec["persona"] = rng.choice(USER_PERSONAS)
        spec["goal"] = rng.choice(TALK_SCENARIOS)
    if scenario == "recall":
        fact, short = rng.choice(RECALL_FACTS)
        share = rng.choice([1, 2])
        ask = share + rng.randint(3, 5)
        spec["recall"] = {"fact": fact, "short": short, "share_turn": share, "ask_turn": ask}
        spec["turns"] = max(spec["turns"], ask + 1)
    return spec


def script_spec(card: dict, name: str, lines: list[str], check, seed: int) -> dict:
    """A scripted probe (fixed user lines, no simulator), e.g. the who-are-you card probes."""
    return {"id": f"{card['id']}:{name}", "seed": seed, "scenario": "script", "name": name, "card": card,
            "card_id": card["id"], "system": C.student_system(card), "script": list(lines),
            "check": list(check) if isinstance(check, tuple) else check, "turns": len(lines), "agent_msgs": [],
            "recall": None, "persona": "", "goal": "", "topic": ""}


def sample_spec(reg: Registry, rng: random.Random, split: str = "train", mix: dict | None = None,
                agent_msgs: list[str] | None = None) -> dict:
    mix = mix or SCENARIO_MIX
    scenario = rng.choices(list(mix), weights=list(mix.values()))[0]
    card = reg.sample_card(rng, split)
    seed = rng.randrange(1 << 31)
    return make_spec(card, scenario, random.Random(seed), seed, agent_msgs)


def train_agent_msgs(path: str | None) -> list[str]:
    """Training agent messages from a pool file, minus the frozen eval ones (they must stay out of training)."""
    held = eval_agent_texts()
    return [t for t in load_agent_msgs(path) if t not in held]


# --- frozen eval sets -----------------------------------------------------------------------------------------
def eval_path(version: str, root: str = POOL_DIR) -> str:
    return os.path.join(root, f"eval_{version}.json")


def _digest(specs: list[dict]) -> str:
    return _h(json.dumps(specs, sort_keys=True, ensure_ascii=False))


def build_eval_specs(reg: Registry, per_card: int = 4, seed: int = 1947) -> list[dict]:
    """Held-out cards in the template and with each a held-out phrasing, plus moth and blank: one sim episode per
    scenario (cycled, per_card of them) and the scripted who-are-you probes for each."""
    from moth.probes import CARD_PROBES, stable_seed
    agent = load_agent_msgs(EVAL_AGENT_MSGS) or MOCK_AGENT_MSGS
    held_ph = reg.phrasings("heldout")
    cards = []
    for i, c in enumerate(reg.cards("heldout")):
        cards.append(c)
        for k in range(len(held_ph)):
            fc = reg.phrase(c, held_ph[(i + k) % len(held_ph)])
            if fc:
                cards.append(fc)
                break
    cards += [reg.card("moth"), reg.card("blank")]
    specs = []
    for j, c in enumerate(cards):
        for k in range(per_card):
            scen = SCENARIOS[(j + k) % len(SCENARIOS)]
            s = stable_seed(f"rl:{c['id']}:{scen}", k)
            specs.append(make_spec(c, scen, random.Random(s + seed), s, agent))
        for _, name, lines, check in CARD_PROBES:
            specs.append(script_spec(c, name, lines, check, stable_seed(f"rl:{c['id']}:{name}", 0)))
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "eval spec ids must be unique"
    return specs


def freeze(version: str, reg: Registry | None = None, root: str = POOL_DIR, per_card: int = 4) -> str:
    path = eval_path(version, root)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: frozen eval sets are never overwritten; pick a new version")
    reg = reg or Registry(root)
    specs = build_eval_specs(reg, per_card)
    doc = {"version": version, "created": time.strftime("%Y-%m-%d"), "n": len(specs),
           "n_cards": len({s["card_id"] for s in specs}), "heldout_pct": HELDOUT_PCT, "sha256": _digest(specs),
           "specs": specs}
    os.makedirs(root, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=0)
    return path


def load_eval(version: str, root: str = POOL_DIR) -> dict:
    """A frozen eval set (sha256 checked). v1/v2 are legacy (tools/home era): kept for old comparisons only."""
    doc = json.load(open(eval_path(version, root), encoding="utf-8"))
    if version in LEGACY_EVALS and root == POOL_DIR:
        print(f"moth.rl: eval_{version} is legacy (frozen with the old tools/home spec); use eval_v3+",
              file=sys.stderr, flush=True)
    if _digest(doc["specs"]) != doc["sha256"]:
        raise ValueError(f"eval_{version}.json was edited after freezing (sha256 mismatch)")
    return doc


# --- teacher-written cards --------------------------------------------------------------------------------------
TEACHER_CARDS_PROMPT = """Write {n} different one-line system prompts that people would type to give a tiny chat buddy an identity, e.g. "you are a cat named bo." or "Name: Pim. You're a grumpy old kettle who lives in a cafe."
Vary everything: what it is (animals, objects, food, weather, places, made-up creatures, plain people with a job or a hobby), names, where it lives, and the writing style (lowercase, Formal, terse "Name: X.", playful, with a trait or two, some with no name). One line each, under 25 words, second person.
Do not use these names: {avoid}.
Return ONLY a JSON list of strings."""


def teacher_lines(ask, n: int, avoid: set[str]) -> list[str]:
    """ask(prompt) -> text. Parses a JSON list of strings (skips anything else)."""
    t = ask(TEACHER_CARDS_PROMPT.format(n=n, avoid=", ".join(sorted(avoid)) or "none"))
    t = t.split("</think>")[-1]
    try:
        xs = json.loads(t[t.index("["): t.rindex("]") + 1])
    except ValueError:
        return []
    return [" ".join(str(x).split()) for x in xs if isinstance(x, str) and 3 <= len(str(x)) <= 200]


def add_teacher_batch(reg: Registry, ask, n: int, source: str = "teacher") -> list[dict]:
    avoid = C.EVAL_FRIEND_NAMES | HELD_NAMES
    out = []
    for line in teacher_lines(ask, n, avoid):
        try:
            c = reg.add_line(line, source)
        except ValueError:
            continue
        if c:
            out.append(c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m moth.rl.registry")
    ap.add_argument("--root", default=POOL_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a_add = sub.add_parser("add")
    a_add.add_argument("--line", action="append", default=[])
    a_add.add_argument("--file", default=None, help="a JSON card or a list of cards")
    a_add.add_argument("--teacher", type=int, default=0, help="ask the teacher for N one-line cards")
    a_add.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    a_add.add_argument("--model", default=None)
    a_ph = sub.add_parser("add-phrasing")
    a_ph.add_argument("tmpl")
    a_fr = sub.add_parser("freeze")
    a_fr.add_argument("version")
    a_fr.add_argument("--per-card", type=int, default=4)
    a = ap.parse_args(argv)
    reg = Registry(a.root)
    if a.cmd == "list":
        for sp in ("train", "heldout"):
            cs = reg.cards(sp)
            n_add = sum(c.get("source") not in ("builtin", "builtin_pool", "builtin_heldout") for c in cs)
            print(f"{sp}: {len(cs)} cards ({n_add} appended), {len(reg.phrasings(sp))} phrasings")
        evs = sorted(f for f in os.listdir(a.root) if f.startswith("eval_")) if os.path.isdir(a.root) else []
        print("frozen eval sets:", ", ".join(evs) or "none")
        return
    if a.cmd == "add":
        added = []
        for line in a.line:
            c = reg.add_line(line)
            added.append(c)
        if a.file:
            d = json.load(open(a.file, encoding="utf-8"))
            for c in d if isinstance(d, list) else [d]:
                added.append(reg.add_card(c, "file"))
        if a.teacher:
            from moth.rl.llm import OAIClient
            cl = OAIClient(a.base_url, a.model)
            added += add_teacher_batch(reg, lambda p: cl.chat([{"role": "user", "content": p}], temperature=1.0,
                                                              max_tokens=60 * a.teacher + 200,
                                                              chat_template_kwargs={"enable_thinking": False})["text"],
                                       a.teacher)
        for c in added:
            print("skipped (already known)" if c is None else f"{c['id']} {c['split']}: {C.describe(c)}")
        return
    if a.cmd == "add-phrasing":
        p = reg.add_phrasing(a.tmpl)
        print("skipped (already known)" if p is None else f"{p['id']} {p['split']}: {p['tmpl']}")
        return
    if a.cmd == "freeze":
        path = freeze(a.version, reg, a.root, a.per_card)
        d = load_eval(a.version, a.root)
        print(f"{path}: {d['n']} specs, {d['n_cards']} cards, sha256 {d['sha256'][:12]}")


if __name__ == "__main__":
    main()
