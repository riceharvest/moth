"""Teacher bake-off: the same chats, probes and judges for several candidate teachers on one rented 5090.

Run on the box by runbook/job_bakeoff.sh (one teacher at a time behind vLLM on :8000); `report` runs locally.

  python -m moth.bakeoff smoke   --out DIR [--think]                    # parser/thinking checks before any chat
  python -m moth.bakeoff gen     --quality-start S --quality-n 150 [gen_dialogs args ...]
  python -m moth.bakeoff replay  --out DIR [--cards heldout_free:4,pool:3] [--think]
  python -m moth.bakeoff label   --judge TAG --out FILE TEACHER_DIR ...  # gist labels (moth.gist prompt), both sources
  python -m moth.bakeoff decide  --judge TAG --out FILE TEACHER_DIR ...  # JEV/GEV /v1/decide yes/no per reply
  python -m moth.bakeoff pairwise --judge TAG --out DIR --pairs a:b,... [--auto-top LABELS ...] TEACHER_DIR ...
  python -m moth.bakeoff report  results/bakeoff                         # tables (local)

Pure chat (2026-10-19): no tools, no home folder.
gen = moth.gen_dialogs with two changes for the seeds [S, S+N): a fixed chat-type mix (talk/agent/talk/recall/agent/long, every 4th chat in zh/es/fr/hi
by seed) and every 5th chat on a held-out identity card (template or held-out free-form phrasing), so every teacher
writes the same card/persona/scenario per seed. Seeds past S+N are the speed run (natural mix). Each record gets "bk"
(quality flag, forced kind, completion time). The teacher also plays the human, so chats diverge after turn 1: the
controlled comparison is `replay` = moth.probes' scripted user lines, answered through gen_dialogs.moth_turn (same
reminders and resampling), written in moth.eval's results.json shape (moth.pairwise, moth.metrics and
gist.chat_for_label_turns read it unchanged).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KINDS6 = ["talk", "agent", "talk", "recall", "agent", "long"]
DEFAULT_SHARES = {"agent": 0.25, "recall": 0.15, "long": 0.15}
NON_EN = ["zh", "es", "fr", "hi"]   # quality seeds: every 4th chat (i % 4 == 1) is non-English, in turn


# --- judge calls (Qwen non-thinking; gpt-oss always reasons: give it room and a fixed effort) ------------------------
def judge_kwargs(root: str, max_tokens: int, attempt: int = 0) -> dict:
    r = root.lower()
    if "gpt-oss" in r:
        return dict(max_tokens=max_tokens + 3000, temperature=0 if attempt == 0 else 0.3,
                    extra_body={"reasoning_effort": "medium"})
    if "lfm" in r:
        return dict(max_tokens=max_tokens + 1500, temperature=0 if attempt == 0 else 0.3,
                    extra_body={"thinking_token_budget": 1024})
    return dict(max_tokens=max_tokens * (1 + attempt), temperature=0 if attempt == 0 else 0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})


# --- smoke ------------------------------------------------------------------------------------------------------------
def cmd_smoke(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--plain-only", action="store_true", help="judges: one plain call, judge settings")
    a = ap.parse_args(argv)
    from openai import OpenAI

    from moth.gen_dialogs import LEAKS, NO_THINK
    c = OpenAI(base_url=a.base_url, api_key="x", timeout=300)
    model = c.models.list().data[0].id
    extra = {"thinking_token_budget": 256} if a.think else NO_THINK
    res = {}

    def call(name, msgs, **kw):
        t0 = time.time()
        try:
            r = c.chat.completions.create(model=model, messages=msgs, extra_body=extra, temperature=0.7,
                                          max_tokens=600 if a.think else 300, **kw)
            m = r.choices[0].message
            out = {"content": m.content, "tool_calls": [(t.function.name, t.function.arguments) for t in m.tool_calls or []],
                   "finish": r.choices[0].finish_reason, "s": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"[:300]}
        res[name] = out
        return out

    sysm = {"role": "system", "content": "you are a tiny cute chat buddy. short lowercase lines."}
    if a.plain_only:
        m = c.models.list().data[0]
        kw = judge_kwargs(str(getattr(m, "root", None) or m.id), 200)
        try:
            r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": 'Return ONLY JSON: {"ok": true}'}],
                                          **kw)
            txt = r.choices[0].message.content or ""
            res["plain"] = {"content": txt, "ok": '"ok"' in txt and not any(x in txt for x in LEAKS)}
        except Exception as e:  # noqa: BLE001
            res["plain"] = {"error": f"{type(e).__name__}: {e}"[:300], "ok": False}
        res["model"], res["pass"] = model, res["plain"]["ok"]
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps(res, ensure_ascii=False)[:400], flush=True)
        sys.exit(0 if res["pass"] else 1)
    p = call("plain", [sysm, {"role": "user", "content": "hi! how was your day?"}])
    res["plain"]["ok"] = bool(p.get("content")) and not any(x in (p.get("content") or "") for x in LEAKS)
    turns = [sysm, {"role": "user", "content": "hi! how was your day?"}, {"role": "assistant", "content": "good! sleepy."},
             {"role": "user", "content": "what?"}]
    m2 = call("multi_turn", turns)
    res["multi_turn"]["ok"] = bool(m2.get("content")) and not any(x in (m2.get("content") or "") for x in LEAKS) \
        and not m2.get("tool_calls")
    res["model"] = model
    res["pass"] = bool(res["plain"]["ok"] and res["multi_turn"]["ok"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in res.items()}), flush=True)
    for k in ("plain", "multi_turn"):
        if k in res:
            print(f"  {k}: {json.dumps(res[k], ensure_ascii=False)[:300]}", flush=True)
    sys.exit(0 if res["pass"] else 1)


# --- gen ----------------------------------------------------------------------------------------------------------------
def cmd_gen(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--quality-start", type=int, required=True)
    ap.add_argument("--quality-n", type=int, default=150)
    a, rest = ap.parse_known_args(argv)
    import moth.gen_dialogs as G
    from moth.cards import heldout_cards
    held = heldout_cards() + heldout_cards(free=True)
    orig_sample, orig_one = G.sample_card, G.one_dialog
    force = {"card": None, "lang": None}

    def sample_card(rng, pool=None):
        c = orig_sample(rng, pool)   # always draw, so the seed's rng stream is the same with or without forcing
        c = dict(force["card"]) if force["card"] is not None else c
        return {**c, "lang": force["lang"]} if force["lang"] else c

    async def one_dialog(client, model, seed, stats):
        i = seed - a.quality_start
        q = 0 <= i < a.quality_n
        kind = KINDS6[i % 6] if q else None
        force["card"] = held[(i // 5) % len(held)] if q and i % 5 == 2 else None
        force["lang"] = (NON_EN[(i // 4) % len(NON_EN)] if i % 4 == 1 else "en") if q else None
        # read synchronously by one_dialog/agent_dialog before their first await: no other chat runs in between
        G.AGENT_SHARE["p"] = (1.0 if kind == "agent" else 0.0) if q else DEFAULT_SHARES["agent"]
        G.RECALL_SHARE["p"] = (1.0 if kind == "recall" else 0.0) if q else DEFAULT_SHARES["recall"]
        G.LONG_SHARE["p"] = (1.0 if kind == "long" else 0.0) if q else DEFAULT_SHARES["long"]
        d = await orig_one(client, model, seed, stats)   # its sync prefix runs now, with the settings above
        if d:
            d["bk"] = {"quality": q, "kind": kind, "heldout": q and i % 5 == 2, "t": time.time()}
        return d

    G.sample_card, G.one_dialog = sample_card, one_dialog
    sys.argv = ["moth.gen_dialogs"] + rest
    G.main()


# --- replay -------------------------------------------------------------------------------------------------------------
def replay_cards(spec: str) -> list[dict]:
    """moth | blank | pool:<i> (a training card) | heldout:<i> | heldout_free:<i>"""
    from moth.cards import BLANK_CARD, MOTH_CARD, build_pool, heldout_cards
    out = []
    for s in spec.split(","):
        if s == "moth":
            out.append(dict(MOTH_CARD))
        elif s == "blank":
            out.append(dict(BLANK_CARD))
        elif s.startswith("pool:"):
            out.append(build_pool()[int(s.split(":")[1])])
        elif s.startswith("heldout_free:"):
            out.append(heldout_cards(free=True)[int(s.split(":")[1])])
        elif s.startswith("heldout:"):
            out.append(heldout_cards()[int(s.split(":")[1])])
    return out


def to_turns(msgs: list[dict]) -> list[dict]:
    """gen_dialogs canonical messages -> moth.eval turns [{"user", "events": [{"say"} | {"tool", "args", "result"}]}]."""
    turns = []
    for m in msgs:
        if m["role"] == "user":
            turns.append({"user": m["content"], "events": []})
        elif m["role"] == "assistant" and turns:
            pending = [{"tool": c["name"], "args": c["arguments"], "result": ""} for c in m.get("tool_calls") or []]
            turns[-1]["events"] += pending
            if m.get("content"):
                turns[-1]["events"].append({"say": m["content"]})
        elif m["role"] == "tool" and turns:
            tools_ev = [e for e in turns[-1]["events"] if "tool" in e and not e["result"]]
            for e, r in zip(tools_ev, m["results"]):
                e["result"] = r
    return turns


async def replay_one(client, model, probe, card, stats, sem):
    import moth.gen_dialogs as G
    from moth.agent import AGENT_CHAT_REMIND, AGENT_REMIND, AGENT_TAG
    from moth.persona import teacher_system
    from moth.probes import WHAT_X, stable_seed, what_word
    cat, name, lines, check = probe
    seed = stable_seed(f"{card['id']}:{name}", 0)
    agent = any(str(u).startswith(AGENT_TAG) for u in lines)
    msgs: list[dict] = []
    oai = [{"role": "system", "content": teacher_system(card)}]
    what_x, broken, t0 = [], 0, time.time()
    async with sem:
        for ti, u in enumerate(lines):
            word = None
            if u == WHAT_X:
                last = G.last_moth(msgs)
                word = what_word(last, random.Random(f"{seed}:{ti}"), avoid={w["word"] for w in what_x})
                u = f"what {word}?" if word else "what?"
            msgs.append({"role": "user", "content": u})
            if agent:
                view = G.teacher_view(u, AGENT_REMIND if u.startswith(AGENT_TAG) else AGENT_CHAT_REMIND, card=card)
            else:
                extra = G.QBACK if random.Random(f"{seed}:q{ti}").random() < 0.2 else G.NO_QBACK
                view = G.teacher_view(u, extra=extra, card=card)
            oai.append({"role": "user", "content": view})
            ok = await G.moth_turn(client, model, oai, msgs, stats, talk_only=agent, card=card,
                                   meta={"type": "agent" if agent else "talk", "topic": "", "card": card})
            if not ok:
                broken += 1
                oai.append({"role": "assistant", "content": "hm?"})   # keep the teacher's view in turn order
            if lines[ti] == WHAT_X:
                what_x.append({"turn": ti, "word": word, "on": None})
    r = {"cat": cat, "name": name, "seed": seed, "sample": 0, "turns": to_turns(msgs), "flags": {}, "check": None,
         "check_kind": None, "card": card, "card_id": card["id"], "broken": broken, "s": round(time.time() - t0, 1)}
    if what_x:
        r["what_x"] = what_x
    return r


def cmd_replay(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default="heldout_free:4,pool:3", help="moth is held-out and not a teacher's job: "
                    "one held-out free-form card + one training-pool card")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-probes", type=int, default=0, help="0 = all of a card's probes")
    a = ap.parse_args(argv)
    from openai import AsyncOpenAI

    import moth.gen_dialogs as G
    from moth.probes import all_probes, for_card
    G.THINK["on"] = a.think
    jobs = []
    for card in replay_cards(a.cards):
        ps = for_card(all_probes(), card)
        jobs += [(p, card) for p in (ps[:a.max_probes] if a.max_probes else ps)]
    stats = collections.Counter()

    async def run():
        client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=300)
        model = (await client.models.list()).data[0].id
        sem = asyncio.Semaphore(a.concurrency)
        t0 = time.time()
        out = await asyncio.gather(*(replay_one(client, model, p, c, stats, sem) for p, c in jobs),
                                   return_exceptions=True)
        good = [r for r in out if isinstance(r, dict)]
        errs = [repr(r)[:200] for r in out if not isinstance(r, dict)]
        os.makedirs(a.out, exist_ok=True)
        json.dump(good, open(os.path.join(a.out, "results.json"), "w"), ensure_ascii=False)
        json.dump({"chats": len(good), "errors": errs, "stats": dict(stats), "s": round(time.time() - t0, 1)},
                  open(os.path.join(a.out, "replay_stats.json"), "w"), indent=1)
        print(f"[replay] {len(good)} chats, {len(errs)} errors, {dict(stats)} in {time.time() - t0:.0f}s", flush=True)
        if errs:
            print("[replay] first error:", errs[0], flush=True)
    asyncio.run(run())


# --- inputs for the judges ----------------------------------------------------------------------------------------------
def teacher_items(tdir: str) -> list[dict]:
    """[{"teacher", "src": "gen"|"replay", "id", "card", "chat", "n"}] for one teacher's out dir (quality chats only)."""
    from moth.gist import chat_for_label_msgs, chat_for_label_turns
    tag = os.path.basename(tdir.rstrip("/"))
    items = []
    p = os.path.join(tdir, "dialogs.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            if (d.get("bk") or {}).get("quality"):
                chat, n = chat_for_label_msgs(d["messages"])
                items.append({"teacher": tag, "src": "gen", "id": d["seed"], "card": d.get("card"), "chat": chat, "n": n})
    p = os.path.join(tdir, "replay", "results.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            chat, n = chat_for_label_turns(r["turns"])
            items.append({"teacher": tag, "src": "replay", "id": f"{r['card_id']}:{r['name']}", "card": r.get("card"),
                          "chat": chat, "n": n})
    return items


def cmd_label(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args(argv)
    from openai import AsyncOpenAI

    from moth.gist import label_prompt, max_tokens_for, parse_labels
    items = [it for d in a.dirs for it in teacher_items(d)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            x = json.loads(line)
            if x.get("labels") is not None:
                done.add((x["teacher"], x["src"], str(x["id"])))
    todo = [it for it in items if (it["teacher"], it["src"], str(it["id"])) not in done and it["n"] > 0]
    stats = collections.Counter()

    async def run():
        client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=600)
        m = (await client.models.list()).data[0]
        root = str(getattr(m, "root", None) or m.id)
        sem, f, t0 = asyncio.Semaphore(a.concurrency), open(a.out, "a"), time.time()

        async def go(it):
            p = label_prompt(it["chat"], it["n"], it["card"])
            err, labs, tries = None, None, 0
            async with sem:
                for attempt in range(3):
                    tries += 1
                    try:
                        r = await client.chat.completions.create(model=m.id, messages=[{"role": "user", "content": p}],
                                                                 **judge_kwargs(root, max_tokens_for(it["n"]), attempt))
                        labs = parse_labels(r.choices[0].message.content or "", it["n"])
                        break
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"[:200]
            x = {"judge": a.judge, "teacher": it["teacher"], "src": it["src"], "id": it["id"], "n": it["n"], "labels": labs,
                 "tries": tries}
            if labs is None:
                x["error"] = err
                stats["err"] += 1
            else:
                stats["ok"] += 1
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            if (stats["ok"] + stats["err"]) % 200 == 0:
                print(f"[label {a.judge}] {dict(stats)} of {len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        await asyncio.gather(*(go(it) for it in todo))
        el = time.time() - t0
        print(f"[label {a.judge}] done {dict(stats)} of {len(todo)} in {el:.0f}s ({len(done)} before)", flush=True)
        replies = sum(it["n"] for it in todo)
        json.dump({"judge": a.judge, "chats": len(todo), "replies": replies, "seconds": round(el, 1),
                   "reply_labels_per_s": round(replies / el, 2) if el else None, "calls_per_s": round(len(todo) / el, 2)
                   if el else None, "concurrency": a.concurrency}, open(a.out + ".speed.json", "w"), indent=1)
    asyncio.run(run())


def load_label_file(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for line in open(path):
            x = json.loads(line)
            if x.get("labels") is not None:
                out[(x["teacher"], x["src"], str(x["id"]))] = x["labels"]
    return out


# --- pairwise -----------------------------------------------------------------------------------------------------------
def cmd_pairwise(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="", help="a:b,... teacher tags (dir basenames)")
    ap.add_argument("--auto-top", action="append", default=[], help="label file (repeatable): also pair the two best by replay on_gist")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args(argv)
    from openai import OpenAI

    import moth.gen_dialogs  # noqa: F401  installs the torch/tokenizers stubs moth.eval's build_sft import needs
    from moth import pairwise as PW
    res = {os.path.basename(d.rstrip("/")): json.load(open(os.path.join(d, "replay", "results.json")))
           for d in a.dirs if os.path.exists(os.path.join(d, "replay", "results.json"))}
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
    if a.auto_top:
        on, n = collections.Counter(), collections.Counter()
        for lf in a.auto_top:
            for (t, src, _), labs in load_label_file(lf).items():
                if src == "replay" and t in res:
                    on[t] += sum(lb["on"] and not lb["self"] for lb in labs)
                    n[t] += len(labs)
        top = sorted((t for t in n if n[t]), key=lambda t: -on[t] / n[t])[:2]
        print("[pairwise] replay on_gist (both judges): " + ", ".join(f"{t} {on[t] / n[t]:.1%}" for t in n if n[t]))
        if len(top) == 2 and not any(set(p) == set(top) for p in pairs):
            pairs.append(tuple(top))
    c = OpenAI(base_url=a.base_url, api_key="x", timeout=600)
    m = c.models.list().data[0]
    root = str(getattr(m, "root", None) or m.id)

    def ask(prompt: str) -> str:
        last = None
        for attempt in range(2):
            try:
                r = c.chat.completions.create(model=m.id, messages=[{"role": "user", "content": prompt}],
                                              **judge_kwargs(root, 400, attempt))
                return (r.choices[0].message.content or "").split("</think>")[-1]
            except Exception as e:  # noqa: BLE001
                last = e
        raise last
    os.makedirs(a.out, exist_ok=True)
    for x, y in pairs:
        if x not in res or y not in res:
            print(f"[pairwise] skip {x}:{y} (missing replay)", flush=True)
            continue
        t0 = time.time()
        rows, warn = PW.run(res[x], res[y], ask, workers=a.workers)
        md = PW.summarize(rows, x, y) + ("\n\nwarnings:\n" + "\n".join(warn[:20]) if warn else "")
        open(os.path.join(a.out, f"{x}__{y}.md"), "w").write(md + "\n")
        json.dump(rows, open(os.path.join(a.out, f"{x}__{y}.json"), "w"), ensure_ascii=False)
        cc = collections.Counter(r["result"] for r in rows)
        print(f"[pairwise {a.judge}] {x} vs {y}: A {cc['A']} tie {cc['tie']} B {cc['B']} ({time.time() - t0:.0f}s)",
              flush=True)


# --- decision judges (autotrust JEV/GEV: POST /v1/decide, one forward pass per yes/no) -------------------------------
DECIDE_STATE = """{language}A small chat-buddy character ({who}) chats with a friend. It is meant to be warm, curious and a bit dim, and to talk about what the friend actually said. Its replies are numbered [R1], [R2], ... Lines starting with [agent] are a coding agent's messages that the character only watches.

{chat}"""
DECIDE_Q = {
    "on": "Does reply [R{k}] respond to what the message just before it meant (answers the question, reacts to the news "
          "or feeling, plays along, or after an [agent] message reacts to what happened), instead of riffing on one word, "
          "ignoring it, turning to itself, or explaining code like an assistant? Silly or wrong is fine if it responds "
          "to the meaning.",
    "self": "Does reply [R{k}] bring up the character's own identity (what it is, its name, where it lives, or things "
            "that go with what it is) although the friend did not ask about it?",
}


def cmd_decide(argv):
    """Per reply two yes/no System-1 decisions (on gist; unprompted identity) for the same chats the generative judges
    label. The state (the whole numbered chat) comes first in every request, so a chat's 2n decisions share one
    prefix (vLLM prefix cache): about one prefill per chat. Writes the label format plus p_on / p_self."""
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args(argv)
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from moth.cards import LANG_NAMES, MOTH_CARD, describe, lang_of
    items = [it for d in a.dirs for it in teacher_items(d)]
    done = set(load_label_file(a.out))
    todo = [it for it in items if (it["teacher"], it["src"], str(it["id"])) not in done and it["n"] > 0]
    stats = collections.Counter()

    def post(body: dict) -> dict:   # stdlib only (the box venv has just openai)
        req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/decide", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(a.concurrency + 4))
        if True:
            sem, f, t0 = asyncio.Semaphore(a.concurrency), open(a.out, "a"), time.time()

            async def one(state, q):
                async with sem:
                    for attempt in range(3):
                        try:
                            j = await asyncio.to_thread(post, {"kind": "noul", "state": state, "question": q,
                                                               "thinking": "off"})
                            stats["decisions"] += 1
                            stats["prompt_tokens"] += j.get("usage", {}).get("prompt_tokens", 0)
                            return float(j["probabilities"][1])
                        except Exception as e:  # noqa: BLE001
                            err = e
                            await asyncio.sleep(2 * (attempt + 1))
                    stats["errors"] += 1
                    print(f"[decide] error: {err!r}"[:300], flush=True)
                    return None

            async def go(it):
                lang = lang_of(it["card"])
                state = DECIDE_STATE.format(who=describe(it["card"] or MOTH_CARD), chat=it["chat"],
                                            language="" if lang == "en" else f"(The chat is in {LANG_NAMES[lang]}.)\n")
                qs = [(k, w) for k in range(1, it["n"] + 1) for w in ("on", "self")]
                # one request first: its prefill puts the chat (the shared state) in the prefix cache; the other 2n-1
                # then only prefill their question (vLLM doesn't share a prefix that is still being prefilled)
                first = await one(state, DECIDE_Q[qs[0][1]].format(k=qs[0][0]))
                ps = [first] + list(await asyncio.gather(*(one(state, DECIDE_Q[w].format(k=k)) for k, w in qs[1:])))
                if any(p is None for p in ps):
                    labs = None
                else:
                    pm = {(k, w): p for (k, w), p in zip(qs, ps)}
                    labs = [{"on": pm[(k, "on")] >= 0.5, "self": pm[(k, "self")] >= 0.5, "p_on": round(pm[(k, "on")], 4),
                             "p_self": round(pm[(k, "self")], 4), "why": ""} for k in range(1, it["n"] + 1)]
                x = {"judge": a.judge, "teacher": it["teacher"], "src": it["src"], "id": it["id"], "n": it["n"], "labels": labs}
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
                stats["chats"] += 1
                if stats["chats"] % 100 == 0:
                    el = time.time() - t0
                    print(f"[decide {a.judge}] {dict(stats)} {stats['decisions'] / el:.1f} decisions/s", flush=True)
            await asyncio.gather(*(go(it) for it in todo))
            el = time.time() - t0
            res = {"judge": a.judge, "chats": stats["chats"], "decisions": stats["decisions"], "errors": stats["errors"],
                   "seconds": round(el, 1), "decisions_per_s": round(stats["decisions"] / el, 2) if el else None,
                   "reply_labels_per_s": round(stats["decisions"] / 2 / el, 2) if el else None,
                   "prompt_tokens": stats["prompt_tokens"], "concurrency": a.concurrency}
            json.dump(res, open(a.out + ".speed.json", "w"), indent=1)
            print(f"[decide {a.judge}] done {res}", flush=True)
    asyncio.run(run())


def main():
    cmds = {"smoke": cmd_smoke, "gen": cmd_gen, "replay": cmd_replay, "label": cmd_label, "pairwise": cmd_pairwise,
            "decide": cmd_decide}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds and sys.argv[1] != "report":
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "report":
        from moth.bakeoff_report import main as rep
        return rep(sys.argv[2:])
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
