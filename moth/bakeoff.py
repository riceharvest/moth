"""Teacher bake-off: the same chats, probes and judges for several candidate teachers on one rented 5090.

Run on the box by runbook/job_bakeoff.sh (one teacher at a time behind vLLM on :8000); `report` runs locally.

  python -m moth.bakeoff smoke   --out DIR [--think]                    # parser/thinking checks before any chat
  python -m moth.bakeoff gen     --quality-start S --quality-n 150 [gen_dialogs args ...]
  python -m moth.bakeoff replay  --out DIR [--cards moth,heldout_free:4] [--think]
  python -m moth.bakeoff label   --judge TAG --out FILE TEACHER_DIR ...  # gist labels (moth.gist prompt), both sources
  python -m moth.bakeoff pairwise --judge TAG --out DIR --pairs a:b,... [--auto-top LABELS ...] TEACHER_DIR ...
  python -m moth.bakeoff report  results/bakeoff                         # tables (local)

gen = moth.gen_dialogs with two changes for the seeds [S, S+N): a fixed chat-type mix (talk/agent/home/talk/recall/agent
by seed) and every 5th chat on a held-out identity card (template or held-out free-form phrasing), so every teacher
writes the same card/persona/scenario per seed. Seeds past S+N are the speed run (natural mix). Each record gets "bk"
(quality flag, forced kind, completion time). The teacher also plays the human, so chats diverge after turn 1: the
controlled comparison is `replay` = moth.probes' scripted user lines, answered through gen_dialogs.moth_turn (same
reminders, resampling and tools), written in moth.eval's results.json shape (moth.pairwise, moth.metrics and
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

KINDS6 = ["talk", "agent", "home", "talk", "recall", "agent"]
DEFAULT_SHARES = {"agent": 0.25, "home": 0.15, "recall": 0.15}


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
    ap.add_argument("--plain-only", action="store_true", help="judges: one plain call (no tools), judge settings")
    a = ap.parse_args(argv)
    from openai import OpenAI

    from moth.gen_dialogs import LEAKS, NO_THINK, OAI_TOOLS
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
    t = call("tool_auto", [sysm, {"role": "user", "content": "please use the bash tool to run `ls` in your home folder"}],
             tools=OAI_TOOLS, tool_choice="auto")
    res["tool_auto"]["ok"] = bool(t.get("tool_calls"))
    if t.get("tool_calls"):
        name, args = t["tool_calls"][0]
        rt = call("tool_result", [sysm, {"role": "user", "content": "please use the bash tool to run `ls` in your home folder"},
                                  {"role": "assistant", "content": None, "tool_calls": [
                                      {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]},
                                  {"role": "tool", "tool_call_id": "c1", "content": "diary\nfriends\nnotes\nsaved"}],
                  tools=OAI_TOOLS, tool_choice="auto")
        res["tool_result"]["ok"] = bool(rt.get("content")) and not any(x in (rt.get("content") or "") for x in LEAKS)
    n = call("tool_none", [sysm, {"role": "user", "content": "what's your favourite colour?"}], tools=OAI_TOOLS,
             tool_choice="none")
    res["tool_none"]["ok"] = bool(n.get("content")) and not any(x in (n.get("content") or "") for x in LEAKS) \
        and not n.get("tool_calls")
    res["model"] = model
    res["pass"] = bool(res["plain"]["ok"] and res["tool_none"]["ok"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "smoke.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in res.items()}), flush=True)
    for k in ("plain", "tool_auto", "tool_result", "tool_none"):
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
    force = {"card": None}

    def sample_card(rng, pool=None):
        c = orig_sample(rng, pool)   # always draw, so the seed's rng stream is the same with or without forcing
        return dict(force["card"]) if force["card"] is not None else c

    async def one_dialog(client, model, seed, pool, stats):
        i = seed - a.quality_start
        q = 0 <= i < a.quality_n
        kind = KINDS6[i % 6] if q else None
        force["card"] = held[(i // 5) % len(held)] if q and i % 5 == 2 else None
        # read synchronously by one_dialog/agent_dialog before their first await: no other chat runs in between
        G.AGENT_SHARE["p"] = (1.0 if kind == "agent" else 0.0) if q else DEFAULT_SHARES["agent"]
        G.HOME_SHARE["p"] = (1.0 if kind == "home" else 0.0) if q else DEFAULT_SHARES["home"]
        G.RECALL_SHARE["p"] = (1.0 if kind == "recall" else 0.0) if q else DEFAULT_SHARES["recall"]
        d = await orig_one(client, model, seed, pool, stats)   # its sync prefix runs now, with the settings above
        if d:
            d["bk"] = {"quality": q, "kind": kind, "heldout": q and i % 5 == 2, "t": time.time()}
        return d

    G.sample_card, G.one_dialog = sample_card, one_dialog
    sys.argv = ["moth.gen_dialogs"] + rest
    G.main()


# --- replay -------------------------------------------------------------------------------------------------------------
def replay_cards(spec: str) -> list[dict]:
    from moth.cards import MOTH_CARD, heldout_cards
    out = []
    for s in spec.split(","):
        if s == "moth":
            out.append(dict(MOTH_CARD))
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


async def replay_one(client, model, probe, card, pool, card_files, stats, sem):
    import moth.gen_dialogs as G
    from moth import _tools as tools
    from moth.agent import AGENT_CHAT_REMIND, AGENT_REMIND, AGENT_TAG
    from moth.home import sample_home
    from moth.persona import teacher_system
    from moth.probes import WHAT_X, stable_seed, what_word
    cat, name, lines, check = probe
    seed = stable_seed(f"{card['id']}:{name}", 0)
    rng = random.Random(seed)
    files = sample_home(rng, pool, card, card_files)
    agent = any(str(u).startswith(AGENT_TAG) for u in lines)
    msgs: list[dict] = []
    oai = [{"role": "system", "content": teacher_system(card)}]
    what_x, broken, t0 = [], 0, time.time()
    async with sem:
        with tools.Workspace(files) as ws:
            for ti, u in enumerate(lines):
                word = None
                if u == WHAT_X:
                    last = G.last_moth(msgs)
                    word = what_word(last, random.Random(f"{seed}:{ti}"), avoid={w["word"] for w in what_x})
                    u = f"what {word}?" if word else "what?"
                msgs.append({"role": "user", "content": u})
                if agent:
                    view = G.teacher_view(u, AGENT_REMIND if u.startswith(AGENT_TAG) else AGENT_CHAT_REMIND)
                else:
                    extra = G.QBACK if random.Random(f"{seed}:q{ti}").random() < 0.2 else G.NO_QBACK
                    view = G.teacher_view(u, extra=extra, card=card)
                oai.append({"role": "user", "content": view})
                ok = await G.moth_turn(client, model, oai, msgs, ws, stats, talk_only=agent, card=card,
                                       meta={"type": "agent" if agent else "talk", "home": files, "topic": "", "card": card})
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
    ap.add_argument("--cards", default="moth,heldout_free:4")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--pool-file", default=None)
    ap.add_argument("--card-file-path", default=None)
    ap.add_argument("--max-probes", type=int, default=0, help="0 = all of a card's probes")
    a = ap.parse_args(argv)
    from openai import AsyncOpenAI

    import moth.gen_dialogs as G
    from moth.home import load_pool
    from moth.probes import all_probes, for_card
    G.THINK["on"] = a.think
    pool = load_pool(a.pool_file)
    card_files = json.load(open(a.card_file_path)) if a.card_file_path and os.path.exists(a.card_file_path) else {}
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
        out = await asyncio.gather(*(replay_one(client, model, p, c, pool, card_files, stats, sem) for p, c in jobs),
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
        print(f"[label {a.judge}] done {dict(stats)} of {len(todo)} in {time.time() - t0:.0f}s ({len(done)} before)",
              flush=True)
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


def main():
    cmds = {"smoke": cmd_smoke, "gen": cmd_gen, "replay": cmd_replay, "label": cmd_label, "pairwise": cmd_pairwise}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds and sys.argv[1] != "report":
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "report":
        from moth.bakeoff_report import main as rep
        return rep(sys.argv[2:])
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
