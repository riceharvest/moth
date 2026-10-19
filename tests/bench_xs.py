"""CPU speed of moth's XS backend (moth.xs) on a full-size XS with random weights, and of the old dense model for A/B.

  python tests/bench_xs.py --modes int8,bf16 --threads 4                 # XS, random weights (no checkpoint needed)
  python tests/bench_xs.py --model DIR --modes int8                      # XS from a model folder
  python tests/bench_xs.py --legacy ~/moth/runs/final2/ckpt_100 --modes int8,bf16 --threads 4   # final2, same harness

Per weight format: load time, weight bytes, prefill tok/s (one prompt into a fresh cache), and batch-1 decode tok/s
over --decode tokens (a forced count: the stop id never comes, so random weights can't end it early) at each
--ctx context length (global attention grows with context). Each decode timing: 1 warm-up + --repeats runs, median.
Random weights: std 1/sqrt(fan_in) per matrix, built directly in bf16 one tensor at a time (stays far under 6 GB);
routers are random too, so expert choice is spread, which is the realistic case for speed.
Prints one JSON line (prefix 'RESULT ') at the end.

Run it capped, GPU hidden (this desktop's rule for heavy jobs):
  CUDA_VISIBLE_DEVICES= ZE_AFFINITY_MASK= ONEAPI_DEVICE_SELECTOR='cpu:*' flock /tmp/claude-1000/heavy.lock \\
    systemd-run --user --scope -p MemoryMax=6G -p CPUQuota=400% nice -n 10 python tests/bench_xs.py --threads 4
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402


class RandomXS(Mapping):
    """TinyMoE state-dict names -> random bf16 tensors, made on access (XSModel converts one at a time)."""

    def __init__(self, cfg, seed=0):
        from moth.xs.model import hash_table
        c = self.c = cfg
        H, Hk, D, d, E, l, f, sf = (c.n_heads, c.n_kv_heads, c.head_dim, c.d_model, c.n_routed, c.latent_dim,
                                    c.expert_dff, c.shared_ffn)
        s = {"embed": (c.vocab_size, d), "final_norm.norm.weight": (d,)}
        if c.bigram_rows:
            s["bigram"] = (c.bigram_rows, d)
        for i in range(c.n_layers):
            p = f"layers.{i}."
            s[p + "attn_norm.norm.weight"] = s[p + "mlp_norm.norm.weight"] = (d,)
            s[p + "attn.w_qg.weight"] = (H * D * (2 if c.gated_attn else 1), d)
            s[p + "attn.w_kv_local.weight"] = (2 * Hk * D, d)
            if c.owns_global_kv(i):
                s[p + "attn.w_kv_global.weight"] = (2 * Hk * D, d)
                s[p + "attn.kg_norm.weight"] = (D,)
            s[p + "attn.w_o.weight"] = (d, H * D)
            s[p + "attn.q_norm.weight"] = s[p + "attn.kl_norm.weight"] = (D,)
            s[p + "attn.sink"] = (H,)
            if not c.is_hash_layer(i):
                s[p + "moe.router"] = (E, d)
            s[p + "moe.e_bias"] = (E,)
            s[p + "moe.w1"] = (E, l, 2 * f)
            s[p + "moe.w2"] = (E, f, l)
            s[p + "moe.shared.0.w_in.weight"] = (2 * sf, d)
            s[p + "moe.shared.0.w_out.weight"] = (d, sf)
        self.shapes, self.seed = s, seed
        self.hash = {f"layers.{i}.moe.hash_idx": i for i in range(c.n_layers) if c.is_hash_layer(i)}
        self._hash_table = hash_table

    def __getitem__(self, k):
        if k in self.hash:
            c = self.c
            return self._hash_table(c.vocab_size, c.n_routed, c.top_k_eff, self.hash[k]).int()
        shp = self.shapes[k]
        g = torch.Generator().manual_seed(self.seed + hash(k) % 100003)
        if k.endswith(("norm.weight",)):
            return torch.ones(shp, dtype=torch.bfloat16)
        if k.endswith((".sink", "e_bias")):
            return torch.zeros(shp, dtype=torch.bfloat16)
        fan_in = shp[-2] if k.endswith((".w1", ".w2")) else shp[-1]
        std = 1.0 if k in ("embed",) else (0.02 if k == "bigram" else 1 / math.sqrt(fan_in))
        return (torch.randn(shp, generator=g) * std).to(torch.bfloat16)

    def __iter__(self):
        return iter(list(self.shapes) + list(self.hash))

    def __len__(self):
        return len(self.shapes) + len(self.hash)


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def timed(fn, repeats):
    fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def bench_xs(mode, a, cfg):
    from moth.xs import weights as W
    from moth.xs.generate import Decoder
    from moth.xs.model import XSModel
    from moth.xs.quant import resolve_mode
    gc.collect()
    t = time.perf_counter()
    model = W.load(a.model, mode) if a.model else XSModel(cfg, RandomXS(cfg), resolve_mode(mode))
    load_s = time.perf_counter() - t
    cfg = model.cfg
    g = torch.Generator().manual_seed(1)
    out = {"mode": model.mode, "load_s": round(load_s, 2), "weights_MB": round(model.nbytes() / 1e6),
           "rss_MB": round(rss_mb())}
    for ctx in a.ctx:
        prompt = torch.randint(32, cfg.vocab_size, (ctx,), generator=g)
        cache = model.new_cache(ctx + a.decode + 8)

        def pre():
            cache.reset()
            return model.forward(prompt, cache)

        pt = timed(pre, max(1, a.repeats // 2))
        start_len = cache.len
        dec = Decoder(model, cache)
        logits = model.forward(prompt[:1], model.new_cache(8))   # any logits to start from

        def decode():
            cache.len = start_len   # rewind (the ring / global entries past start_len get overwritten)
            cache.lens_cpu[0] = start_len
            dec.generate(logits, a.decode, stop=-1, temperature=0)

        # cache.lpos keeps positions <= start_len + decode from the previous run: the decode step only reads ring
        # slots it has just written or ones inside the window, so rewinding is safe for timing
        dt = timed(decode, a.repeats)
        out[f"prefill_{ctx}_tok_s"] = round(ctx / pt, 1)
        out[f"decode_ctx{ctx}_tok_s"] = round((a.decode + 1) / dt, 1)
        out[f"ms_per_tok_ctx{ctx}"] = round(dt / (a.decode + 1) * 1e3, 2)
    out["rss_MB_after"] = round(rss_mb())
    return out


def bench_legacy(mode, a):
    from tiny_agent import generate as G
    from tiny_agent.checkpoint import load_model
    gc.collect()
    t = time.perf_counter()
    model = load_model(a.legacy, device="cpu", dtype=torch.float32, low_mem=True).eval()
    if mode != "fp32":
        G.quantize_cpu(model, mode)
    load_s = time.perf_counter() - t
    V = model.cfg.vocab_size
    g = torch.Generator().manual_seed(1)
    out = {"mode": mode, "load_s": round(load_s, 2), "rss_MB": round(rss_mb())}
    for ctx in a.ctx:
        prompt = torch.randint(32, V, (1, ctx), generator=g)
        cache = G.KVCache(model, 1, ctx + a.decode + 8, "cpu", dtype=torch.float32)

        def pre():
            cache.reset([0])
            return G.append(model, cache, prompt, [ctx])

        pt = timed(pre, max(1, a.repeats // 2))
        logits = pre()
        L0 = int(cache.lens_cpu[0])
        dec = G.Decoder(model, cache, graph=False)

        def decode():
            cache.lens_cpu[0] = L0
            cache.lens[0] = L0
            dec.generate(logits, a.decode, stop=-1, temperature=0)

        dt = timed(decode, a.repeats)
        out[f"prefill_{ctx}_tok_s"] = round(ctx / pt, 1)
        out[f"decode_ctx{ctx}_tok_s"] = round((a.decode + 1) / dt, 1)
        out[f"ms_per_tok_ctx{ctx}"] = round(dt / (a.decode + 1) * 1e3, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="an XS model folder (default: random full-size XS weights)")
    ap.add_argument("--legacy", default=None, help="an old tiny-agent model folder (final2) instead of XS")
    ap.add_argument("--modes", default="int8,bf16")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", default="300,3000")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--config", default="{}", help="XS config overrides as JSON (random weights only)")
    a = ap.parse_args()
    a.ctx = [int(x) for x in a.ctx.split(",")]
    torch.set_num_threads(a.threads)
    from moth.xs.config import XSConfig
    cfg = XSConfig(**json.loads(a.config))
    res = {"what": "legacy " + a.legacy if a.legacy else (a.model or "XS random weights"), "threads": a.threads,
           "torch": torch.__version__, "cpu": platform.processor() or platform.machine(), "runs": []}
    for mode in a.modes.split(","):
        r = bench_legacy(mode, a) if a.legacy else bench_xs(mode, a, cfg)
        print(json.dumps(r), flush=True)
        res["runs"].append(r)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
