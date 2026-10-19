"""Test moth like a user would: scripted multi-turn probes through the real chat engine (moth.serve.Moth).

  python -m moth.eval --model runs/sft1/model --samples 3 --device cuda [--judge-url http://127.0.0.1:8000/v1] [--out DIR]
      [--cards moth,heldout:0,heldout_free:4,text:"you are a cat named bo"]
      (default: moth + held-out identities, in the template and in held-out free-form phrasings)

Probes live in moth/probes.py (scripted chats, "what X" follow-ups built from moth's own reply, recall at 5/10/20
turns, the user's own typing style, held-out real agent final messages). Every (probe, sample) gets a seed from a
stable hash of its name, so two checkpoints see the same homes and sampling streams (compare them with
python -m moth.pairwise A/results.json B/results.json).

Per reply: automatic flags (style, invented files, leaked format tokens, tool errors, repeats) and the deterministic
metrics of moth/metrics.py (identity crutch, bible copy, dodge, agent misreadings, repeats, diversity, motifs).
Per probe: the write/recall/what-X/follows-card checks. --cards runs the probes once per identity card: the moth card,
held-out identities that never appear in training (heldout:<i>), the same in held-out free-form phrasings
(heldout_free:<i>), or any one-liner (text:...). Moth-only probes are skipped for the others.
With --judge-url: the headline is per reply, on_gist % (moth.gist's label prompt, the same one build_sft filters with);
the teacher also scores each chat 1-5 (JUDGE below), which counts for less.
Writes <out>/report.md (tables), <out>/transcripts.md (every chat, for reading by eye) and <out>/results.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moth.agent import AGENT_TAG  # noqa: E402
from moth.build_sft import BANNED, FILE_RE  # noqa: E402
from moth.cards import MOTH_CARD, describe, load_card, student_system, student_system_v1  # noqa: E402
from moth.gist import chat_for_label_turns, follows_card, label_prompt, parse_labels  # noqa: E402
from moth.probes import WHAT_X, all_probes, for_card, on_word, stable_seed, what_word  # noqa: E402

PROBES = all_probes()

LEGACY = {"on": False}   # --legacy-prompt: the tools/home student prompt of the released round-3 weights
LEAK = re.compile(r"<\|?im_|<tool_call>|</?think>|<tool_response>|NOT_FOUND|submit", re.I)


def reply_flags(text: str, home_names: set[str], seen: str) -> list[str]:
    f = []
    words = len(text.split())
    if text != text.lower():
        f.append("uppercase")
    if words > 40:
        f.append("long")
    if words == 0 or text.strip() in ("...",):
        f.append("empty")
    if re.search(r"^\s*([-*#]|\d+\.)\s", text, re.M) or "**" in text:
        f.append("markdown")
    if any(b in text.lower() for b in BANNED):
        f.append("assistanty")
    if LEAK.search(text):
        f.append("leak")
    if re.search(r"\bfans?\b", text.lower()):
        f.append("mentions_fan")   # round-2 obsession: 25% of teacher replies
    if any(name not in seen for name in FILE_RE.findall(text.lower())):
        f.append("file_talk")   # pure chat: it has no files (old tools models: names it never saw)
    return f



def seed_everything(seed: int):
    random.seed(seed)
    try:   # moth samples with torch.multinomial: seed torch too, or "paired" samples still diverge
        import torch
        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)   # also seeds cuda/xpu generators
    except ImportError:
        pass


def run_probe(moth, probe, seed: int, home_root: str, sample: int = 0, home_cls=None, card: dict | None = None):
    """One chat in a fresh home. moth: a moth.serve.Moth (or anything with .a.pool_file, .home, reset(), chat()).
    card: the identity card (system prompt + home); None = whatever moth was started with (the moth card)."""
    if home_cls is None:
        from moth.serve import Home as home_cls
    cat, name, lines, check = probe
    seed_everything(seed)
    shutil.rmtree(home_root, ignore_errors=True)
    moth.system = (student_system_v1 if LEGACY["on"] else student_system)(card if card is not None else MOTH_CARD)
    if card is not None:
        moth.home = home_cls(home_root, seed, moth.a.pool_file, card=card)
    else:
        moth.home = home_cls(home_root, seed, moth.a.pool_file)
    before = {f"{d['dir']}/{x}" for d in moth.home.tree() for x in d["files"]}
    moth.reset()
    turns, flags, seen = [], collections.Counter(), ""
    replies, what_x, last_said = [], [], ""
    for ti, u in enumerate(lines):
        word = None
        if u == WHAT_X:   # the friend asks about a word from moth's last reply
            word = what_word(last_said, random.Random(f"{seed}:{ti}"), avoid={w["word"] for w in what_x})
            u = f"what {word}?" if word else "what?"
        seen += " " + u.lower()   # files the friend or the agent named are not inventions
        t0 = time.time()
        evs = moth.chat(u)
        dt = time.time() - t0
        names = {p.split("/")[-1] for p in before} | {d["dir"] for d in moth.home.tree()}
        said = []
        for e in evs:
            if "tool" in e:
                seen += " " + e["result"].lower() + " " + json.dumps(e["args"]).lower()
                if e["result"].startswith("Error") or e["tool"] not in ("bash", "read", "write"):
                    flags["tool_error"] += 1
                if e["tool"] in ("submit", "edit"):
                    flags["old_tool"] += 1
            else:
                for fl in reply_flags(e["say"], names, seen):
                    flags[fl] += 1
                if e["say"] in replies:
                    flags["repeat"] += 1
                if e["say"] == "i got lost in my folders.":
                    flags["lost"] += 1
                replies.append(e["say"])
                said.append(e["say"])
        last_said = " ".join(said)
        if lines[ti] == WHAT_X:
            what_x.append({"turn": ti, "word": word, "on": on_word(last_said, word) if word else None})
        turns.append({"user": u, "events": evs, "s": round(dt, 2)})
    after = {f"{d['dir']}/{x}" for d in moth.home.tree() for x in d["files"]}
    ok, kind = None, None
    if check == "wrote":
        kind = "wrote"
        ok = bool(after - before) or any(e.get("tool") == "write" for t in turns for e in t["events"])
    elif check == "what_x":
        kind = "what_x"
        oks = [w["on"] for w in what_x if w["on"] is not None]
        ok = all(oks) if oks else None
    elif isinstance(check, tuple) and check[0] == "card":
        kind = "follows_card"
        ok = follows_card(" ".join(e.get("say", "") for e in turns[-1]["events"]), card)
    elif isinstance(check, tuple):
        kind = check[2] if len(check) > 2 else "recall"
        last = " ".join(e.get("say", "") for e in turns[-1]["events"])
        ok = on_word(last, check[1])
    r = {"cat": cat, "name": name, "seed": seed, "sample": sample, "turns": turns, "flags": dict(flags), "check": ok,
         "check_kind": kind, "tools": sum(1 for t in turns for e in t["events"] if "tool" in e), "n_replies": len(replies)}
    if what_x:
        r["what_x"] = what_x
    if card is not None:
        r["card"], r["card_id"] = card, card.get("id")
    return r


JUDGE_KEYS = ("cute", "charm", "character", "coherence", "back_and_forth")   # cute = primary (v4: pure chat, no grounding)
OLD_KEYS = ("charm", "character", "coherence", "grounding")   # judge v1 (rounds <=3): character was saturated at 5
JUDGE = """You are grading a chat with a tiny 112M-parameter chatbot character ({who}). It should be a fun, warm, curious, slightly dim chat buddy who talks in short lowercase lines and talks about what the friend actually said; its identity is light seasoning. It is only a chat: it has no files or tools. Judge it as a fun companion, not as an assistant. Be strict: 5 is rare and means nothing to improve; a typical decent reply earns 3.
Lines starting with "agent:" are messages from a big coding agent to the friend. The bot only watches them; it should react to what HAPPENED (done, broken, still waiting, asking the friend something, sorry, a big win) in plain cute words, never explain, summarize details or give advice. A misreading of one word only counts as good if it still fits what happened.

Score each 1-5:
- cute (the main score): How cute and fun is it to talk with? A silly lil guy, stupid-cute, endearing, curious about the friend? Dumb, wrong or confused is GOOD if adorable and still on the point. Bad: robotic, assistant-like, preachy, repeating itself, grabbing one word and riffing on it instead of answering, or talking about itself (what it is, where it lives, its stock images) all the time. More than about 1 reply in 10 about its own identity counts against it.
- charm: fun, surprising, sweetly wrong. 5 = several replies made you smile; 3 = pleasant but generic; 1 = flat or annoying.
- character: stays in character (lowercase, short, never an assistant) WITHOUT leaning on stock lines. Subtract for each identity crutch instead of a real answer ("i'm just a ..."), each repeated line or image, each unprompted reference to what it is or where it lives beyond the first, each explanation of an agent message. When the friend asks who or what it is, answering from its identity is right, not a crutch; claiming a different identity is wrong. 5 = fresh every turn; 3 = one crutch or repeat; 1 = mostly crutches, self-talk or assistant talk.
- coherence: each reply answers what the friend just said or meant (silly is fine; ignoring it or riffing on one of its words is not). When the friend says "what?", "huh" or "what <word>", it explains its previous reply in plain words.
- back_and_forth: a real conversation. 5 = refers back to earlier turns, keeps the thread, asks the friend things; 3 = answers each line but each stands alone; 1 = disconnected one-liners.

Return ONLY JSON, reason first: {{"reason": "<one line>", "cute": n, "charm": n, "character": n, "coherence": n, "back_and_forth": n}}

Chat:
{chat}"""
JUDGE_V = 4


def chat_text(r) -> str:
    out = []
    for t in r["turns"]:
        if t["user"].startswith(AGENT_TAG):
            out.append(f"agent: {t['user'][len(AGENT_TAG):].strip()}")
        else:
            out.append(f"user: {t['user']}")
        for e in t["events"]:
            if "tool" in e:
                a = e["args"]
                out.append(f"  [moth ran {e['tool']} {a.get('command') or a.get('path') or ''}] -> {e['result'][:300]!r}")
            else:
                out.append(f"moth: {e['say']}")
    return "\n".join(out)


def make_judge(url, think=None, max_tokens=500):
    """ask(prompt) -> the judge's answer text (reasoning stripped), and a description of the judge.

    think=None: auto from the served model (/v1/models root). LFM2.5's template always opens <think>: never send
    enable_thinking=False to it (vLLM's qwen3 reasoning parser then returns the reasoning as content), cap the
    reasoning with thinking_token_budget instead (at 256 it failed 4% of chats and scored grounding 2/5 on chats
    without any files, 1024 fixes both). Qwen (Qwen3.8-27B served as "teacher"): non-thinking call."""
    from openai import OpenAI
    client = OpenAI(base_url=url, api_key="x", timeout=300)
    m = client.models.list().data[0]
    model, root = m.id, str(getattr(m, "root", None) or m.id)
    if think is None:
        think = "lfm" in root.lower()
    kw = (dict(max_tokens=max_tokens + 2000, extra_body={"thinking_token_budget": 1024}) if think else
          dict(max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}}))

    def ask(prompt: str) -> str:
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              temperature=0, **kw)
        return (resp.choices[0].message.content or "").split("</think>")[-1]
    return ask, f"{root} (think={think})"


def parse_json(text: str) -> dict:
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def judge_all(results, url, think=None, ask=None):
    """Score every chat in place (r["judge"]). ask: a prompt -> text callable (tests), else built from url."""
    desc = "custom"
    if ask is None:
        ask, desc = make_judge(url, think)
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        for attempt in range(3):   # server hiccups / a reply without the JSON: ask again
            try:
                j = parse_json(ask(JUDGE.format(chat=chat_text(r), who=describe(r.get("card") or load_card(None)))))
                for k in JUDGE_KEYS:
                    j[k] = float(j[k])
                j["v"] = JUDGE_V
                r["judge"] = j
                return
            except Exception as e:  # noqa: BLE001
                r["judge"] = {"error": str(e)[:100]}
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, results))
    ok = sum("charm" in r["judge"] for r in results)
    print(f"judge {desc}: scored {ok}/{len(results)}", flush=True)


def gist_all(results, url=None, think=None, ask=None):
    """Label every reply ON/OFF gist in place (r["gist"] = [{"on", "why"}, ...]) with moth.gist's prompt."""
    if ask is None:
        ask, _ = make_judge(url, think, max_tokens=1500)
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        chat, n = chat_for_label_turns(r["turns"])
        if n == 0:
            r["gist"] = []
            return
        for _ in range(3):
            try:
                r["gist"] = parse_labels(ask(label_prompt(chat, n, r.get("card"))), n)
                return
            except Exception as e:  # noqa: BLE001
                r["gist"] = {"error": str(e)[:100]}
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, results))
    ok = [x for r in results if isinstance(r.get("gist"), list) for x in r["gist"]]
    print(f"gist labels: {len(ok)} replies, on_gist {sum(x['on'] for x in ok) / max(len(ok), 1):.1%}, unprompted "
          f"identity {sum(bool(x.get('self')) for x in ok) / max(len(ok), 1):.2%}", flush=True)


def on_gist_rows(results) -> list[str]:
    """Markdown rows: on_gist % per category (+ per card when several) and ALL."""
    def rate(rs):
        xs = [x for r in rs if isinstance(r.get("gist"), list) for x in r["gist"]]
        if not xs:
            return "n/a", "n/a", 0
        return (f"{sum(x['on'] for x in xs) / len(xs):.1%}", f"{sum(bool(x.get('self')) for x in xs) / len(xs):.2%}",
                len(xs))
    by = collections.defaultdict(list)
    for r in results:
        by[r["cat"]].append(r)
    head = "| on_gist | unprompted identity (target <= 1%) | replies |"
    L = ["| category " + head, "|---|---|---|---|"]
    for cat, rs in sorted(by.items()) + [("**ALL**", results)]:
        v, me, n = rate(rs)
        L.append(f"| {cat} | {v} | {me} | {n} |")
    cards = sorted({r.get("card_id") or "moth" for r in results})
    if len(cards) > 1:
        L += ["", "| card " + head, "|---|---|---|---|"]
        for c in cards:
            v, me, n = rate([r for r in results if (r.get("card_id") or "moth") == c])
            L.append(f"| {c} | {v} | {me} | {n} |")
    return L


def probe_hash(names) -> str:
    """Short id of a probe set: two runs with different ids are not fully paired (e.g. a box without
    data/eval_agent_msgs.json silently has no agent_real probes)."""
    import hashlib
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()[:8]


def report(results, out, label):
    from moth.metrics import table
    by = collections.defaultdict(list)
    for r in results:
        by[r["cat"]].append(r)
    L = [f"# moth eval: {label}", "", f"{len(results)} chats, {sum(r['n_replies'] for r in results)} replies", ""]
    samplers = {json.dumps(r.get("sampler"), sort_keys=True) for r in results if r.get("sampler")}
    if samplers:
        L += ["sampler: " + " / ".join(sorted(samplers)), ""]
    names = sorted({r["name"] for r in results})
    L += [f"probe set {probe_hash(names)}: {len(names)} probes ("
          + ", ".join(f"{c} {len({r['name'] for r in rs})}" for c, rs in by.items()) + ")", ""]
    if any(isinstance(r.get("gist"), list) for r in results):
        L += ["## Per reply (headline; moth.gist label prompt): on_gist, unprompted identity", ""] + on_gist_rows(results) + [""]
    cards = sorted({r.get("card_id") or "moth" for r in results})
    L += ["## Deterministic metrics (no judge)", "", table({"all": results} | (
        {f"card {c}": [r for r in results if (r.get("card_id") or "moth") == c] for c in cards} if len(cards) > 1 else {}) | (
        {"user style": by["style"]} if by.get("style") else {}) | (
        {"agent chats": [r for r in results if r["cat"].startswith("agent")]} if any(c.startswith("agent") for c in by) else {})), ""]
    tot = collections.Counter()
    for r in results:
        tot.update(r["flags"])
    nrep = max(1, sum(r["n_replies"] for r in results))
    L += ["## Flags (per 100 replies)", "", "| flag | rate |", "|---|---|"]
    L += [f"| {k} | {v / nrep * 100:.1f} |" for k, v in tot.most_common()]
    fails = [r for r in results if r["check"] is False]
    if fails:
        L += ["", "failed checks: " + ", ".join(f"{r['name']}#{r.get('sample', r['seed'])}" for r in fails)]
    J = [r["judge"] for r in results if "judge" in r and "charm" in r["judge"]]
    if J:
        keys = [k for k in JUDGE_KEYS if any(k in j for j in J)]
        ver = sorted({j.get("v", 1) for j in J})
        L += ["", f"## Judge (mean 1-5, judge prompt v{'/'.join(map(str, ver))})", "",
              "| category | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        order = (["style"] if "style" in by else []) + [c for c in by if c != "style"]
        for cat, rs in [(c, by[c]) for c in order] + [("ALL", results)]:
            js = [r["judge"] for r in rs if "judge" in r and "charm" in r["judge"]]
            if js:
                L.append(f"| {'**user style**' if cat == 'style' else cat} | {len(js)} | " + " | ".join(
                    f"{statistics.mean(float(j[k]) for j in js if k in j):.2f}" if any(k in j for j in js) else "n/a"
                    for k in keys) + " |")
    lat = [t["s"] for r in results for t in r["turns"]]
    if lat:
        L += ["", f"latency per user turn: median {statistics.median(lat):.2f}s, max {max(lat):.1f}s",
              f"chats using tools: {sum(1 for r in results if r['tools']) / len(results):.0%}"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    T = [f"# transcripts: {label}", ""]
    style_first = sorted(results, key=lambda r: r["cat"] != "style")
    for r in style_first:
        j = r.get("judge", {})
        wx = f" what_x={[(w['word'], w['on']) for w in r['what_x']]}" if r.get("what_x") else ""
        g = r.get("gist")
        gl = ("\ngist: " + " | ".join(f"R{i + 1} {'ON' if x['on'] else 'OFF'}{' SELF' if x.get('self') else ''}: "
                                       f"{x['why']}" for i, x in enumerate(g))
              if isinstance(g, list) and g else "")
        T += [f"## {r['cat']}/{r['name']} card={r.get('card_id', 'moth')} (sample {r.get('sample', '?')}, seed {r['seed']}) "
              f"flags={r['flags']} check={r['check']}{wx} judge={j}{gl}", "```", chat_text(r), "```", ""]
    open(os.path.join(out, "transcripts.md"), "w").write("\n".join(T))
    json.dump(results, open(os.path.join(out, "results.json"), "w"))
    return "\n".join(L)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(root, "data/tiny-agent-112m/tokenizer.json"))
    ap.add_argument("--pool-file", default=os.path.join(root, "moth/file_pool.json"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)   # = the training runs' eval setting; pairwise warns on a mismatch
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--ctx-budget", type=int, default=None)
    ap.add_argument("--max-turn-tokens", type=int, default=200)
    ap.add_argument("--max-tool-rounds", type=int, default=3)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-think", choices=["auto", "on", "off"], default="auto",
                    help="judge reasons first (LFM2.5) or not (Qwen); auto = from /v1/models")
    ap.add_argument("--only", default=None, help="comma-separated categories")
    ap.add_argument("--cards", default="moth,heldout:0,heldout:8,heldout_free:2,heldout_free:4,heldout_free:9",
                    help="comma-separated identity cards to run the probes with: moth, blank, heldout:<i>, a card "
                         "JSON file (held-out cards never appear in training)")
    ap.add_argument("--no-gist", action="store_true", help="with --judge-url: skip the per-reply on_gist labels")
    ap.add_argument("--legacy-prompt", action="store_true",
                    help="the tools/home student prompt (released round-3 weights) instead of the pure-chat one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    LEGACY["on"] = a.legacy_prompt
    from moth.serve import Moth
    a.out = a.out or os.path.join(root, "evals", os.path.basename(os.path.dirname(a.model.rstrip("/"))) or "model")
    os.makedirs(a.out, exist_ok=True)
    a.home = os.path.join(a.out, "home")
    a.seed = 0
    moth = Moth(a)
    probes = [p for p in PROBES if not a.only or p[0] in a.only.split(",")]
    if not any(p[0] == "agent_real" for p in probes) and not a.only:
        print("note: no data/eval_agent_msgs.json, skipping the held-out agent probes", flush=True)
    sampler = {"temperature": a.temperature, "top_p": a.top_p, "rep_penalty": a.rep_penalty, "ctx_budget": a.ctx_budget,
               "max_turn_tokens": a.max_turn_tokens, "max_tool_rounds": a.max_tool_rounds}
    results = []
    for spec in a.cards.split(","):
        card = load_card(spec)
        ps = for_card(probes, card)
        for i, p in enumerate(ps):
            for s in range(a.samples):
                r = run_probe(moth, p, stable_seed(p[1], s), a.home, sample=s, card=card)
                r["sampler"] = sampler
                results.append(r)
            print(f"[{card['id']}] {i + 1}/{len(ps)} {p[0]}/{p[1]}", flush=True)
    if a.judge_url:
        think = {"auto": None, "on": True, "off": False}[a.judge_think]
        if not a.no_gist:
            gist_all(results, a.judge_url, think)
        judge_all(results, a.judge_url, think)
    print(report(results, a.out, f"{a.model} T={a.temperature}"))


if __name__ == "__main__":
    main()
