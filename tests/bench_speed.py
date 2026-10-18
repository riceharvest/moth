"""Inference speed of moth on the real serving path, old vs new, interleaved in one process (the desktop wobbles 5-10%).

  new: tiny_agent.generate.append (whole prompt in one pass) + Decoder.generate (what Moth._turn calls: exact eager
       loop on CPU, one captured device graph per token on CUDA/XPU with on-device sampling)
  old: append_chunked (64-token pieces, the pre-2026-10-18 path) + a per-token append loop with a host read-back

  prefill:    tok/s for one prompt of N tokens into a fresh cache
  decode b1:  tok/s for --decode greedy tokens after a --ctx-token prompt (new: Decoder.generate with a stop id that
              never comes, so exactly --decode tokens + the closing stop append; old: the old bench loop)
  decode sampled (new only): the same with moth's sampler (T 0.8, top-p 0.95, rep penalty 1.3 on 300 ids)
  serve turn: Moth.chat() end to end (render, tokenize, prefill / feed, sampled reply, tool calls) over a fixed
              conversation with a temp home; old = moth.serve with its pre-2026-10-18 _feed/_turn/_sample patched in
Each measurement: 1 warm-up + --repeats timed runs, median. Prompts are fixed seeded random token ids.

  python tests/bench_speed.py --device cpu --dtype fp32 --low-mem --threads 8,16      # serving default on CPU
  python tests/bench_speed.py --device cpu --dtype fp32 --low-mem --quant int8         # optional speed modes
  python tests/bench_speed.py --device xpu --dtype bf16                                # or --device cuda
  python tests/bench_speed.py --device cpu --dtype fp32 --low-mem --equiv 12           # old vs new outputs
Prints one JSON line (prefix 'RESULT ') at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # moth's tiny_agent, not any other copy on PYTHONPATH

import torch  # noqa: E402

from tiny_agent import generate as G  # noqa: E402
from tiny_agent.checkpoint import load_model  # noqa: E402

MAX_LEN = 4096   # = moth.serve.MAX_LEN
TOKENIZER = os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json")
CHATS = [   # serve-turn and equivalence conversations (3 user turns each)
    ["hi moth! how are you today?", "what's in your home?", "can you read one of your files for me?"],
    ["i had a rough day at work", "my boss yelled at me about a deadline", "thanks for listening"],
    ["do you like lamps?", "which lamp is your favourite?", "why that one?"],
    ["tell me something you learned recently", "where did you read that?", "write it down so you remember"],
    ["what's 12 times 7?", "are you sure?", "ok what about 9 times 9"],
    ["i'm learning to cook pasta", "it always gets sticky", "should i add oil?"],
    ["good morning", "did you sleep?", "what did you dream about?"],
    ["can you list your files?", "how many are there?", "which is the longest?"],
    ["my cat knocked over a plant", "she looks proud of herself", "what should i name her?"],
    ["i'm bored", "suggest something to do", "that sounds fun, tell me more"],
    ["what do moths eat?", "do you eat that too?", "lol"],
    ["i finished my first marathon!", "it took 4 hours", "my legs hurt so much"],
]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "xpu":
        torch.xpu.synchronize()


def med(fn, repeats):
    fn()                                                   # warm-up
    return statistics.median(fn() for _ in range(repeats))


# --- moth.serve before 2026-10-18 (for the A/B) --------------------------------------------------------------------
def legacy_sample(self, logits, talk_only=False):
    logits = logits.float()
    if talk_only:
        logits[0, self.ban] = float("-inf")
    pen = getattr(self.a, "rep_penalty", 1.0)
    if pen != 1.0 and self.said:
        ids = list({i for r in self.said for i in r} - self.common - self.protect)
        if ids:
            ix = torch.tensor(ids, device=logits.device)
            v = logits[0, ix]
            logits[0, ix] = torch.where(v > 0, v / pen, v * pen)
    if self.a.temperature <= 0:            # (the old code had no greedy mode; added for the comparison)
        return int(logits[0].argmax())
    p = torch.softmax(logits / self.a.temperature, dim=-1)
    sp, si = p.sort(dim=-1, descending=True)
    keep = sp.cumsum(-1) - sp < self.a.top_p
    sp = sp * keep
    return int(si[0, torch.multinomial(sp[0] / sp[0].sum(), 1)])


def legacy_turn(self, logits, talk_only=False):
    out = []
    for _ in range(self.a.max_turn_tokens):
        t = legacy_sample(self, logits, talk_only)
        out.append(t)
        logits = G.append_chunked(self.model, self.cache, torch.tensor([[t]], device=self.a.device), [1])
        if t == self.im_end:
            break
    if out[-1] != self.im_end:
        G.append_chunked(self.model, self.cache, torch.tensor([[self.im_end]], device=self.a.device), [1])
    return self.tok.decode(out, skip_special_tokens=False)


def legacy_feed(self, ids):
    if int(self.cache.lens_cpu[0]) + len(ids) + 200 > self.budget:
        return None
    return G.append_chunked(self.model, self.cache, torch.tensor([ids], device=self.a.device), [len(ids)])


def legacy_new_cache(self):
    return G.KVCache(self.model, 1, MAX_LEN, self.a.device, dtype=self.kv_dtype)


def make_moth(a, model, home, temperature):
    """A real moth.serve.Moth around an already loaded model (no second copy of the weights)."""
    from moth import serve
    args = serve.build_parser().parse_args(["--model", a.model, "--tokenizer", TOKENIZER, "--home", home,
                                            "--device", a.device, "--temperature", str(temperature)]
                                           + (["--no-graph"] if a.no_graph else [])
                                           + (["--compile"] if getattr(a, "compile", False) else []))
    m = serve.Moth.__new__(serve.Moth)
    from tokenizers import Tokenizer
    m.tok = Tokenizer.from_file(TOKENIZER)
    m.model = model
    m.kv_dtype = next(model.parameters()).dtype if a.quant == "none" else torch.float32
    m._kv = G.KVCache(model, 1, MAX_LEN, a.device, dtype=m.kv_dtype)
    m.setup(args)
    if a.device != "cpu":           # as Moth.__init__: capture the GPU graphs up front
        m.cache = m._kv
        m._decoder().warmup(args.max_turn_tokens, m.im_end, args.temperature, args.top_p, args.rep_penalty)
        m.cache = None
    return m


def run_chat(m, chat, old, seed):
    """One conversation through Moth.chat(); old=True patches in the pre-2026-10-18 inference methods."""
    if old:
        m._turn, m._feed, m._new_cache = (types.MethodType(f, m) for f in (legacy_turn, legacy_feed, legacy_new_cache))
    else:
        for k in ("_turn", "_feed", "_new_cache"):
            m.__dict__.pop(k, None)
    m.forget()
    torch.manual_seed(seed)
    if m.a.device == "cuda":
        torch.cuda.manual_seed(seed)
    out = []
    for u in chat:
        evs = m.chat(u)
        out.append(evs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "runs/final2/ckpt_100"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--low-mem", action="store_true")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "bf16"], help="CPU: moth.serve --quant")
    ap.add_argument("--threads", default="8", help="CPU threads, comma list (moth.serve default: physical cores)")
    ap.add_argument("--prefill", default="128,512,2048")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--impl", default="both", choices=["both", "new", "old"])
    ap.add_argument("--serve-chats", type=int, default=3, help="conversations for the serve-turn timing (0: skip)")
    ap.add_argument("--equiv", type=int, default=0, help="compare old vs new outputs over this many conversations")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--compile", action="store_true", help="GPU: torch.compile the decode step (moth.serve --compile)")
    a = ap.parse_args()

    dev = torch.device(a.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    t0 = time.perf_counter()
    model = load_model(a.model, device=a.device, dtype=dtype, low_mem=a.low_mem).eval()
    if a.quant != "none":
        G.quantize_cpu(model, a.quant)
    load_s = time.perf_counter() - t0
    kv_dtype = dtype if a.quant == "none" else torch.float32
    V = model.cfg.vocab_size
    never = V - 1     # a stop id greedy decoding of these prompts doesn't produce: decode runs the full length
    lens = [int(x) for x in a.prefill.split(",")]
    prompt = torch.randint(100, V, (1, max(lens + [a.ctx])), generator=torch.Generator().manual_seed(0)).to(dev)
    impls = ["new", "old"] if a.impl == "both" else [a.impl]
    res = {"device": a.device, "dtype": a.dtype, "low_mem": a.low_mem, "quant": a.quant, "torch": torch.__version__,
           "load_s": round(load_s, 2), "runs": {}}
    if dev.type == "xpu":
        res["gpu"] = torch.xpu.get_device_name(0)
    elif dev.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name(0)

    def new_cache():
        return G.KVCache(model, 1, MAX_LEN, dev, dtype=kv_dtype)

    def prefill_once(n, impl):
        cache = new_cache()
        sync(dev)
        t = time.perf_counter()
        logits = (G.append if impl == "new" else G.append_chunked)(model, cache, prompt[:, :n], [n])
        sync(dev)
        return time.perf_counter() - t, cache, logits

    decoders = {}

    def decode_once(impl, sampled=False):
        _, cache, logits = prefill_once(a.ctx, "new")
        if impl == "new":
            # one cache + decoder reused across runs, as in moth.serve (the device graph is tied to the cache)
            key = sampled
            if key not in decoders:
                dc = new_cache()
                decoders[key] = (dc, G.Decoder(model, dc, graph=False if a.no_graph else None, compile=a.compile))
            c, dec = decoders[key]
            for name in ("gk", "gv", "lk", "lv", "lpos", "lens", "lens_cpu", "tokens"):
                getattr(c, name).copy_(getattr(cache, name))
            sync(dev)
            t = time.perf_counter()
            if sampled:
                out = dec.generate(logits, a.decode, never, 0.8, 0.95, 1.3, set(range(1000, 1300)), [never])
            else:
                out = dec.generate(logits, a.decode, never, 0.0, 0.95)
            assert len(out) == a.decode, f"decode stopped early ({len(out)} tokens): pick another stop id"
            res["decode_path"], res["decode_fallback"] = dec.mode, dec.fallback
        else:
            sync(dev)
            t = time.perf_counter()
            for _ in range(a.decode):
                tok = int(logits[0].argmax())
                logits = G.append_chunked(model, cache, torch.tensor([[tok]], device=dev), [1])
        sync(dev)
        return time.perf_counter() - t

    home_root = tempfile.mkdtemp(prefix="moth-bench-")   # never live/home: moth's tools write into its home
    threads = [int(x) for x in a.threads.split(",")] if dev.type == "cpu" else [None]
    try:
        for th in threads:
            if th:
                torch.set_num_threads(th)
            for impl in impls:
                r = {"prefill_tok_s": {}}
                for n in lens:
                    m_ = med(lambda: prefill_once(n, impl)[0], a.repeats)
                    r["prefill_tok_s"][n] = round(n / m_, 1)
                if impl == "new" and dev.type != "cpu" and not a.no_graph:
                    # feed: a short chunk onto a 512-token context (a new turn / tool result), through the cache's
                    # captured-graph feed (what moth.serve uses on a GPU); prefill above is the eager path
                    fc = new_cache()
                    fdec = G.Decoder(model, fc)
                    fdec.warmup(a.decode, never, 0.0, 0.95)
                    r["feed_ms"] = {}
                    for n in (16, 64, 256):
                        def feed_once():
                            fc.reset([0])
                            fc.static_feed, sf = None, fc.static_feed
                            G.append(model, fc, prompt[:, :a.ctx], [a.ctx])
                            fc.static_feed = sf
                            sync(dev)
                            t = time.perf_counter()
                            G.append(model, fc, prompt[:, a.ctx:a.ctx + n] if a.ctx + n <= prompt.size(1)
                                     else prompt[:, :n], [n])
                            sync(dev)
                            return time.perf_counter() - t
                        r["feed_ms"][n] = round(med(feed_once, a.repeats) * 1000, 2)
                    del fc, fdec
                m_ = med(lambda: decode_once(impl), a.repeats)
                r["decode_tok_s"] = round(a.decode / m_, 1)
                if impl == "new":
                    r["decode_sampled_tok_s"] = round(a.decode / med(lambda: decode_once(impl, True), a.repeats), 1)
                if a.serve_chats and os.path.isfile(TOKENIZER):
                    m = make_moth(a, model, os.path.join(home_root, f"{impl}-{th}"), 0.8)
                    ts, ntok = [], 0
                    for ci, chat in enumerate(CHATS[:a.serve_chats]):
                        old = impl == "old"
                        if old:
                            m._turn, m._feed, m._new_cache = (types.MethodType(f, m) for f in
                                                              (legacy_turn, legacy_feed, legacy_new_cache))
                        m.forget()
                        torch.manual_seed(ci)
                        for u in chat:
                            n0 = len(m.messages)
                            sync(dev)
                            t = time.perf_counter()
                            m.chat(u)
                            sync(dev)
                            ts.append(time.perf_counter() - t)
                            ntok += sum(len(m.enc(x.get("content") or "")) for x in m.messages[n0:]
                                        if x["role"] == "assistant")
                    r["serve_turn_ms"] = round(statistics.median(ts[1:] or ts) * 1000, 1)
                    r["serve_turns"] = len(ts)
                    r["serve_reply_tokens"] = ntok
                res["runs"][f"{impl}" + (f"/th{th}" if th else "")] = r
                print(f"{impl:3s}" + (f" th{th:<2d}" if th else "") + " prefill " +
                      " / ".join(f"{n}:{v:.0f}" for n, v in r["prefill_tok_s"].items()) +
                      f" tok/s | decode b1 {r['decode_tok_s']:.1f} tok/s" +
                      (f" (sampled {r['decode_sampled_tok_s']:.1f})" if "decode_sampled_tok_s" in r else "") +
                      (" | feed after 512: " + " / ".join(f"{n}:{v}ms" for n, v in r["feed_ms"].items())
                       if "feed_ms" in r else "") +
                      (f" | serve turn {r['serve_turn_ms']:.0f} ms median over {r['serve_turns']} turns"
                       if "serve_turn_ms" in r else ""), flush=True)

        if a.equiv:
            res["equiv"] = equivalence(a, model, prompt, lens, kv_dtype, home_root, dev)
    finally:
        shutil.rmtree(home_root, ignore_errors=True)
    if dev.type == "cpu":
        res["threads"] = threads
    res["host"] = platform.processor() or platform.machine()
    print("RESULT " + json.dumps(res), flush=True)


def equivalence(a, model, prompt, lens, kv_dtype, home_root, dev):
    """Old vs new: prefill logits (max |diff|, same argmax), then whole conversations through Moth.chat with greedy
    replies and with seeded sampling (T 0.8): identical replies per turn?"""
    out = {"prefill_max_abs_diff": {}, "prefill_same_argmax": {}}
    for n in lens:
        ca, cb = G.KVCache(model, 1, MAX_LEN, dev, dtype=kv_dtype), G.KVCache(model, 1, MAX_LEN, dev, dtype=kv_dtype)
        la = G.append(model, ca, prompt[:, :n], [n])
        lb = G.append_chunked(model, cb, prompt[:, :n], [n])
        out["prefill_max_abs_diff"][n] = float((la - lb).abs().max())
        out["prefill_same_argmax"][n] = bool((la.argmax(-1) == lb.argmax(-1)).all())
    if next(model.parameters()).dtype != torch.float32 or a.quant != "none":
        # bf16 / quantized: how far is each path from an fp32 reference (same device, old path)?
        ref = load_model(a.model, device=a.device, dtype=torch.float32, low_mem=True).eval()   # tables stay bf16: exact
        out["vs_fp32_ref_max_abs"] = {}
        for n in lens:
            cr = G.KVCache(ref, 1, MAX_LEN, dev, dtype=torch.float32)
            lr = G.append_chunked(ref, cr, prompt[:, :n], [n]).float()
            d = {}
            for impl, fn in (("new", G.append), ("old", G.append_chunked)):
                c = G.KVCache(model, 1, MAX_LEN, dev, dtype=kv_dtype)
                lx = fn(model, c, prompt[:, :n], [n]).float()
                d[impl] = round(float((lx - lr).abs().max()), 4)
                d[impl + "_same_argmax"] = bool((lx.argmax(-1) == lr.argmax(-1)).all())
            out["vs_fp32_ref_max_abs"][n] = d
        print("equiv vs fp32 reference " + json.dumps(out["vs_fp32_ref_max_abs"]), flush=True)
        if a.quant == "none":
            # bf16 baseline: how often does each bf16 path give the same greedy replies as the fp32 model?
            aref = argparse.Namespace(**{**vars(a), "quant": "none"})
            mr = make_moth(aref, ref, os.path.join(home_root, "b-ref"), 0.0)
            mb = make_moth(a, model, os.path.join(home_root, "b-bf16"), 0.0)
            agree = {"new": [0, 0], "old": [0, 0]}
            for ci, chat in enumerate(CHATS[:a.equiv]):
                rr = run_chat(mr, chat, False, ci)
                for impl in ("new", "old"):
                    rb = run_chat(mb, chat, impl == "old", ci)
                    for mm in (mr, mb):
                        shutil.rmtree(mm.home.root, ignore_errors=True)
                        mm.home = type(mm.home)(mm.home.root, mm.a.seed, mm.a.pool_file)
                    agree[impl][0] += sum(x == y for x, y in zip(rb, rr))
                    agree[impl][1] += len(rr)
            out["bf16_vs_fp32_greedy_turns"] = {k: f"{v[0]}/{v[1]}" for k, v in agree.items()}
            print("equiv bf16 vs fp32 model, greedy turns identical: " + json.dumps(out["bf16_vs_fp32_greedy_turns"]),
                  flush=True)
        if a.quant != "none":
            # drift of the speed mode: greedy conversations, quantized vs the full-precision model (both new path)
            same_turns = turns = same_chats = 0
            mq = make_moth(a, model, os.path.join(home_root, "q-quant"), 0.0)
            aref = argparse.Namespace(**{**vars(a), "quant": "none"})
            mr = make_moth(aref, ref, os.path.join(home_root, "q-ref"), 0.0)
            for ci, chat in enumerate(CHATS[:a.equiv]):
                rq, rr = run_chat(mq, chat, False, ci), run_chat(mr, chat, False, ci)
                for mm in (mq, mr):
                    shutil.rmtree(mm.home.root, ignore_errors=True)
                    mm.home = type(mm.home)(mm.home.root, mm.a.seed, mm.a.pool_file)
                sm = [x == y for x, y in zip(rq, rr)]
                same_turns, turns, same_chats = same_turns + sum(sm), turns + len(sm), same_chats + all(sm)
            out["quant_vs_fp32_greedy"] = {"chats": a.equiv, "identical_chats": same_chats,
                                           "identical_turns": f"{same_turns}/{turns}"}
            print(f"equiv {a.quant} vs fp32, greedy: {same_chats}/{a.equiv} conversations identical, "
                  f"{same_turns}/{turns} turns", flush=True)
        del ref
    for temp, name in ((0.0, "greedy"), (0.8, "sampled_seeded")):
        m = make_moth(a, model, os.path.join(home_root, f"eq-{name}"), temp)
        same_turns = turns = same_chats = 0
        diffs = []
        for ci, chat in enumerate(CHATS[:a.equiv]):
            new = run_chat(m, chat, False, 1000 + ci)
            shutil.rmtree(m.home.root, ignore_errors=True)     # same fresh home for the old run
            m.home = type(m.home)(m.home.root, m.a.seed, m.a.pool_file)
            old = run_chat(m, chat, True, 1000 + ci)
            shutil.rmtree(m.home.root, ignore_errors=True)
            m.home = type(m.home)(m.home.root, m.a.seed, m.a.pool_file)
            s = [x == y for x, y in zip(new, old)]
            same_turns += sum(s)
            turns += len(s)
            same_chats += all(s)
            if not all(s):
                diffs.append({"chat": ci, "new": new, "old": old})
        out[name] = {"chats": a.equiv, "identical_chats": same_chats, "identical_turns": f"{same_turns}/{turns}"}
        if diffs:
            out[name]["first_diff"] = diffs[0]
        print(f"equiv {name}: {same_chats}/{a.equiv} conversations identical, {same_turns}/{turns} turns", flush=True)
    print("equiv prefill max|diff| " + json.dumps(out["prefill_max_abs_diff"]) + " same argmax " +
          json.dumps(out["prefill_same_argmax"]), flush=True)
    return out


if __name__ == "__main__":
    main()
