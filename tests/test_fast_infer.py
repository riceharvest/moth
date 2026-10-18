"""The fast inference paths in tiny_agent/generate.py against the reference chunked append (small random model, CPU).

  python tests/test_fast_infer.py         (from ~/moth; a few seconds; needs torch)

  * append (one pass, query-blocked attention) == append_chunked: logits and the whole cache state, for a fresh
    cache, a non-empty cache, chunks shorter/longer than the window, several query blocks, ragged batches with
    n_new 0, and rows= sub-batches. float64: equal to ~1e-12 (only the reduction order differs); fp32: rounding level.
  * Decoder.step == append_chunked for one token: bit-identical logits and cache in fp32.
  * Decoder.generate: the eager loop == the old per-token loop token for token (seeded sampling, rep penalty, ban);
    the device-loop step (graph="static", what CUDA/XPU capture) leaves the same tokens and cache.
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:   # pragma: no cover
    torch = None

if torch is not None:
    from tiny_agent import generate as G
    from tiny_agent.model import ModelConfig, TinyAgentLM


# The attention softmax runs in float32 in every path (as in the old code), so even a float64 model agrees only to
# ~1e-7 when the key set of a softmax row changes length; a logic error shows up as O(1).
F64_TOL = 1e-5


def tiny_model(dtype, seed=0, engram=True):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=300, d_model=64, n_layers=6, n_heads=4, head_dim=16, n_kv_heads=2, kv_group=4,
                      swa_window=8, max_seq_len=512, engram_layers=(1,) if engram else (),
                      engram_rows_per_head=97, engram_heads=2, engram_head_dim=8)
    m = TinyAgentLM(cfg)
    with torch.no_grad():
        for name, p in m.named_parameters():   # the zero-initialised projections would make every logit 0
            if p.dim() == 2 and "embed" not in name:
                p.normal_(0, 0.3)
    return m.to(dtype).eval()


def cache_state(c):
    """Everything later steps can read: global K/V below each row's length, ring slots in use, history, lengths."""
    out = {"lens": c.lens.clone(), "lens_cpu": c.lens_cpu.clone(), "lpos": c.lpos[:, :-1].clone()}
    for b in range(c.B):
        L = int(c.lens_cpu[b])
        out[f"gk{b}"], out[f"gv{b}"] = c.gk[:, b, :, :L].clone(), c.gv[:, b, :, :L].clone()
        out[f"tok{b}"] = c.tokens[b, :L].clone()
        used = c.lpos[b, :-1] >= 0
        out[f"lk{b}"], out[f"lv{b}"] = c.lk[:, b, :, :-1][:, :, used].clone(), c.lv[:, b, :, :-1][:, :, used].clone()
    return out


@unittest.skipIf(torch is None, "needs torch")
class FastPrefill(unittest.TestCase):
    def assertStateClose(self, a, b, atol):
        sa, sb = cache_state(a), cache_state(b)
        self.assertEqual(sa.keys(), sb.keys())
        for k in sa:
            if sa[k].is_floating_point():
                torch.testing.assert_close(sa[k], sb[k], atol=atol, rtol=0, msg=lambda s, k=k: f"{k}: {s}")
            else:
                self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def run_case(self, dtype, B, steps, block, atol):
        """steps: list of (n_new list, rows or None). Feeds the same random tokens to both paths."""
        m = tiny_model(dtype)
        V = m.cfg.vocab_size
        ca = G.KVCache(m, B, 200, "cpu", dtype=dtype)
        cb = G.KVCache(m, B, 200, "cpu", dtype=dtype)
        g = torch.Generator().manual_seed(1)
        for n_new, rows in steps:
            C = max(n_new) if max(n_new) else 3
            idx = torch.randint(0, V, (len(n_new), C), generator=g)
            la = G.append(m, ca, idx, n_new, rows=rows, block=block)
            lb = G.append_chunked(m, cb, idx, n_new, rows=rows)
            if max(n_new) == 0:
                self.assertIsNone(la)
                self.assertIsNone(lb)
                continue
            real = [i for i, n in enumerate(n_new) if n > 0]      # rows with n_new == 0 return garbage by contract
            # logits come back float32 (model.forward does .float() too): compare at that rounding
            torch.testing.assert_close(la[real], lb[real], atol=max(atol, 4e-6), rtol=0)
            self.assertStateClose(ca, cb, atol)
        return m, ca, cb

    def test_fresh_and_nonempty_float64(self):
        old = G.CHUNK
        try:
            for chunk in (64, 5):                    # reference chunking: one piece, and many pieces
                G.CHUNK = chunk
                for block in (None, 1, 3, 7, 16):     # query blocks: smaller than the window, odd, several
                    self.run_case(torch.float64, 1, [([37], None), ([13], None), ([1], None), ([3], None),
                                                     ([20], None)], block, F64_TOL)
        finally:
            G.CHUNK = old

    def test_short_chunks_and_window_edge_float64(self):
        steps = [([2], None), ([1], None), ([8], None), ([7], None), ([9], None), ([1], None)]
        for block in (None, 2, 4):
            self.run_case(torch.float64, 1, steps, block, F64_TOL)

    def test_ragged_batch_and_rows_float64(self):
        steps = [([5, 0, 11], None), ([0, 0, 0], None), ([3, 9, 1], None), ([4, 6], [2, 0]), ([1], [1]),
                 ([12, 1], [1, 2]), ([1, 1, 1], None)]
        for block in (None, 3):
            self.run_case(torch.float64, 3, steps, block, F64_TOL)

    def test_padded_shapes_float64(self):
        """The GPU shape bucketing (chunk padded to a power of two, global keys rounded up), forced on the CPU."""
        old_pad, old_round = G.PAD_SHAPES, G.KEY_ROUND
        G.PAD_SHAPES, G.KEY_ROUND = True, 16
        try:
            self.run_case(torch.float64, 1, [([37], None), ([13], None), ([1], None), ([20], None)], None, F64_TOL)
            self.run_case(torch.float64, 3, [([5, 0, 11], None), ([4, 6], [2, 0]), ([1, 1, 1], None)], 3, F64_TOL)
        finally:
            G.PAD_SHAPES, G.KEY_ROUND = old_pad, old_round

    def test_explicit_attention_matches_sdpa(self):
        torch.manual_seed(0)
        q, K, V = torch.randn(2, 4, 5, 16), torch.randn(2, 2, 9, 16), torch.randn(2, 2, 9, 16)
        m = torch.rand(2, 1, 5, 9) > 0.4
        m[..., 0] = True
        ref = torch.nn.functional.scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True)
        torch.testing.assert_close(G._attn_explicit(q, K, V, m, 2), ref, atol=1e-5, rtol=0)

    def test_float32_rounding_level(self):
        self.run_case(torch.float32, 1, [([150], None), ([40], None), ([1], None)], None, 5e-5)
        self.run_case(torch.float32, 1, [([150], None), ([40], None)], 32, 5e-5)

    def test_long_prompt_default_blocks_split(self):
        old = G.QUERY_BLOCK["cpu"]
        G.QUERY_BLOCK["cpu"] = 64                    # the default block size, forced small: 190 tokens = 3 blocks
        try:
            self.assertEqual(G._block_size("cpu", 190), 64)
            self.run_case(torch.float64, 1, [([190], None), ([5], None)], None, F64_TOL)
        finally:
            G.QUERY_BLOCK["cpu"] = old


@unittest.skipIf(torch is None, "needs torch")
class FastDecode(unittest.TestCase):
    def prefilled(self, dtype=torch.float32, n=30, max_len=120, seed=0):
        m = tiny_model(dtype, seed)
        c = G.KVCache(m, 1, max_len, "cpu", dtype=dtype)
        idx = torch.randint(0, m.cfg.vocab_size, (1, n), generator=torch.Generator().manual_seed(seed + 5))
        logits = G.append(m, c, idx, [n])
        return m, c, logits

    def test_step_bit_identical_to_append(self):
        for n in (1, 5, 30):                          # short history (Engram padding, ring not full) and long
            m, c, _ = self.prefilled(n=n)
            ref = copy.deepcopy(c)
            dec = G.Decoder(m, c, graph=False)
            g = torch.Generator().manual_seed(3)
            for _ in range(25):
                t = int(torch.randint(0, m.cfg.vocab_size, (1,), generator=g))
                la = dec.step(t)
                lb = G.append_chunked(m, ref, torch.tensor([[t]]), [1])
                self.assertTrue(torch.equal(la, lb))
                sa, sb = cache_state(c), cache_state(ref)
                for k in sa:
                    self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def old_turn(self, m, c, logits, max_new, stop, temp, top_p, pen, pen_ids, ban):
        """moth.serve's per-token loop before 2026-10-18 (host sampling + append_chunked)."""
        out = []
        for _ in range(max_new):
            logits = logits.float()
            if ban:
                logits[0, ban] = float("-inf")
            if pen != 1.0 and pen_ids:
                ix = torch.tensor(list(pen_ids))
                v = logits[0, ix]
                logits[0, ix] = torch.where(v > 0, v / pen, v * pen)
            p = torch.softmax(logits / temp, dim=-1)
            sp, si = p.sort(dim=-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            t = int(si[0, torch.multinomial(sp[0] / sp[0].sum(), 1)])
            out.append(t)
            logits = G.append_chunked(m, c, torch.tensor([[t]]), [1])
            if t == stop:
                break
        if out[-1] != stop:
            G.append_chunked(m, c, torch.tensor([[stop]]), [1])
        return out

    def test_generate_eager_matches_old_loop(self):
        stop = 7
        for seed, (max_new, temp, pen, ban) in enumerate([(40, 0.8, 1.3, [11, 12]), (15, 1.5, 1.0, []),
                                                          (60, 2.0, 1.3, [])]):
            m, c, logits = self.prefilled(seed=seed)
            ref = copy.deepcopy(c)
            pen_ids = set(range(20, 80, 3))
            torch.manual_seed(100 + seed)
            a = G.Decoder(m, c, graph=False).generate(logits.clone(), max_new, stop, temp, 0.95, pen, pen_ids, ban)
            torch.manual_seed(100 + seed)
            b = self.old_turn(m, ref, logits.clone(), max_new, stop, temp, 0.95, pen, pen_ids, ban)
            self.assertEqual(a, b)
            sa, sb = cache_state(c), cache_state(ref)
            for k in sa:
                self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def test_static_device_loop_greedy(self):
        """The captured-graph step, run uncaptured: same greedy tokens and cache as the eager loop, both when the
        turn stops early (steps after <|im_end|> are no-ops on the cache) and when it is cut off."""
        for stop_pick, max_new, sync in ((True, 30, 4), (False, 9, 4), (True, 30, 1), (False, 12, 5)):
            m, c, logits = self.prefilled(dtype=torch.float64, n=12)
            ref = copy.deepcopy(c)
            never = m.cfg.vocab_size - 1
            eager = G.Decoder(m, ref, graph=False).generate(logits.clone(), max_new, never, 0.0, 0.95)
            stop = eager[min(6, len(eager) - 1)] if stop_pick else never
            ref = copy.deepcopy(c)
            eager = G.Decoder(m, ref, graph=False).generate(logits.clone(), max_new, stop, 0.0, 0.95)
            dec = G.Decoder(m, c, graph="static", sync_every=sync)
            got = dec.generate(logits.clone(), max_new, stop, 0.0, 0.95)
            self.assertEqual(got, eager)
            sa, sb = cache_state(c), cache_state(ref)
            for k in sa:
                if sa[k].is_floating_point():
                    torch.testing.assert_close(sa[k], sb[k], atol=F64_TOL, rtol=0, msg=k)
                else:
                    self.assertTrue(torch.equal(sa[k], sb[k]), k)
            got2 = dec.generate(dec.step(3), 5, stop, 0.0, 0.95)        # a second turn on the same decoder works
            self.assertGreater(len(got2), 0)

    def test_static_feed_matches_append(self):
        """Short chunks through the captured-feed step (graph="static" attaches it, uncaptured here) == append."""
        m, c, logits = self.prefilled(dtype=torch.float64, n=12)
        ref = copy.deepcopy(c)
        G.Decoder(m, c, graph="static")
        self.assertIsNotNone(c.static_feed)
        g = torch.Generator().manual_seed(9)
        for w in (1, 5, 16, 17, 40, 3):                    # bucket edges, > window, several buckets
            idx = torch.randint(0, m.cfg.vocab_size, (1, w), generator=g)
            la = G.append(m, c, idx, [w])
            ref.static_feed = None
            lb = G.append(m, ref, idx, [w])
            torch.testing.assert_close(la, lb, atol=1e-5, rtol=0)
            sa, sb = cache_state(c), cache_state(ref)
            for k in sa:
                if sa[k].is_floating_point():
                    torch.testing.assert_close(sa[k], sb[k], atol=F64_TOL, rtol=0, msg=lambda s, k=k: f"{k}: {s}")
                else:
                    self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def test_static_sampler_distribution(self):
        """Inverse-CDF draw (device loop) vs multinomial (eager): same nucleus, same frequencies."""
        torch.manual_seed(0)
        logits = torch.randn(1, 50) * 3
        pm = torch.zeros(1, 50, dtype=torch.bool)
        pm[0, :10] = True
        bm = torch.zeros(1, 50, dtype=torch.bool)
        bm[0, 3] = True
        n = 20000
        u = torch.rand(n)
        a = torch.stack([G._sample_icdf(logits, 0.8, 0.9, 1.3, pm, bm, u[i:i + 1]) for i in range(n)]).view(-1)
        b = torch.stack([G.sample_top_p(logits, 0.8, 0.9, 1.3, pm[0], bm[0]) for _ in range(n)]).view(-1)
        fa, fb = torch.bincount(a, minlength=50) / n, torch.bincount(b, minlength=50) / n
        self.assertEqual(set(a.tolist()), set(b.tolist()))
        self.assertLess((fa - fb).abs().max().item(), 0.02)
        self.assertEqual(int(fa[3]), 0)

    def test_quant_modes_run(self):
        for mode in ("int8", "bf16"):
            m, c, _ = self.prefilled()
            G.quantize_cpu(m, mode)
            c = G.KVCache(m, 1, 100, "cpu", dtype=torch.float32)
            idx = torch.randint(0, m.cfg.vocab_size, (1, 20), generator=torch.Generator().manual_seed(0))
            lg = G.append(m, c, idx, [20])
            toks = G.Decoder(m, c).generate(lg, 10, m.cfg.vocab_size - 1, 0.0, 0.95)
            self.assertGreater(len(toks), 0)
            self.assertTrue(torch.isfinite(lg).all())


class _Exact64(torch.nn.Module if torch is not None else object):
    """A "quant" Linear that keeps float64 weights and takes float64 activations: runs _decode_fast's logic with no
    rounding, so it can be held to _decode_one at float64 tolerance."""
    ACT_DTYPE = torch.float64 if torch is not None else None

    def __init__(self, lin):
        super().__init__()
        self.weight = torch.nn.Parameter(lin.weight.detach().double(), requires_grad=False)
        self.bias = None

    def mm(self, xb):
        return torch.nn.functional.linear(xb, self.weight)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


@unittest.skipIf(torch is None, "needs torch")
class CpuSpeedModes(unittest.TestCase):
    """The CPU speed modes (quantize_cpu): the lean decode step (_decode_fast) == _decode_one, the int8 weights, the
    topk nucleus sampler."""

    def steps(self, m, n, dtype, steps=40):
        """Prefill n tokens, then `steps` decode steps through _decode_fast and _decode_one on two copies of the
        cache. 40 steps wrap the tiny model's 8-slot window several times; layers 0 and 4 are Full."""
        c = G.KVCache(m, 1, 120, "cpu", dtype=dtype)
        G.append(m, c, torch.randint(0, m.cfg.vocab_size, (1, n), generator=torch.Generator().manual_seed(5)), [n])
        ref = copy.deepcopy(c)
        fast, slow = G.Decoder(m, c, graph=False), G.Decoder(m, ref, graph=False)
        slow.fast = False
        g = torch.Generator().manual_seed(3)
        worst = 0.0
        for _ in range(steps):
            t = int(torch.randint(0, m.cfg.vocab_size, (1,), generator=g))
            worst = max(worst, float((fast.step(t) - slow.step(t)).abs().max()))
        return c, ref, worst

    def test_fast_step_matches_decode_one_float64(self):
        G.QUANT_CLASSES["exact64"] = _Exact64
        try:
            for n in (1, 2, 5, 30):     # Engram padding (L < 2), ring not yet full, full and wrapped
                m = G.quantize_cpu(tiny_model(torch.float64), "exact64")
                c, ref, worst = self.steps(m, n, torch.float64)
                self.assertLess(worst, 1e-5, n)          # logits are float32 + the fp32 softmax: ~1e-6
                sa, sb = cache_state(c), cache_state(ref)
                for k in sa:
                    if sa[k].is_floating_point():
                        torch.testing.assert_close(sa[k], sb[k], atol=F64_TOL, rtol=0, msg=lambda s, k=k: f"{k}: {s}")
                    else:
                        self.assertTrue(torch.equal(sa[k], sb[k]), k)
        finally:
            G.QUANT_CLASSES.pop("exact64")

    def test_fast_step_close_for_bf16_int8(self):
        """Real kernels: the two paths round differently (bf16 activations), so only close."""
        for mode in ("bf16", "int8"):
            m = G.quantize_cpu(tiny_model(torch.float32), mode)
            self.assertIsNotNone(m._cpu_fast)
            for n in (1, 30):
                c, ref, worst = self.steps(m, n, torch.float32)
                self.assertLess(worst, 0.5, (mode, n))
                self.assertTrue(torch.equal(c.lpos, ref.lpos))
                self.assertTrue(torch.equal(c.lens_cpu, ref.lens_cpu))

    def test_int8_weights(self):
        torch.manual_seed(0)
        lin = torch.nn.Linear(64, 48, bias=False)
        q = G.Int8Linear(lin)
        self.assertEqual(q.qweight.dtype, torch.int8)
        self.assertEqual(q.scale.dtype, torch.bfloat16)
        w = lin.weight.detach()
        # rounded against the stored bf16 scale: off by at most half a step (+ a hair where the rounded scale clips)
        err = (q.weight - w).abs().amax(1)
        self.assertTrue(bool((err <= q.scale.float() * 0.51 + w.abs().amax(1) * 2 ** -8).all()))
        x = torch.randn(3, 64)
        ref = x.bfloat16().float() @ q.weight.t()
        torch.testing.assert_close(q(x), ref, atol=0.02, rtol=0.02)            # kernel: bf16 activations / output
        xl = torch.randn(G.INT8_GEMV_ROWS + 5, 64)
        torch.testing.assert_close(q(xl), xl @ q.weight.t(), atol=1e-5, rtol=0)  # long prefill: dequant + fp32 GEMM

    def test_topk_sampler(self):
        """CPU vocabularies above TOPK_SAMPLE_MIN: nucleus via topk. Same support and frequencies as the sort path,
        bans hold, the penalty matches the where form, the caller's logits stay untouched."""
        V = 8192
        torch.manual_seed(1)
        logits = torch.randn(1, V) * 1.5
        logits[0, :30] += 6
        pm = torch.zeros(V, dtype=torch.bool)
        pm[torch.randperm(V)[:200]] = True
        pm[:10] = True
        bm = torch.zeros(V, dtype=torch.bool)
        bm[3] = True
        keep = logits.clone()
        ref = torch.where(pm, torch.where(logits > 0, logits / 1.3, logits * 1.3), logits).masked_fill(bm, -float("inf"))
        self.assertEqual(int(G.sample_top_p(logits, 0.0, 0.95, 1.3, pm, bm)), int(ref.argmax(-1)))
        n = 6000
        g = torch.Generator().manual_seed(2)
        a = torch.stack([G.sample_top_p(logits, 0.8, 0.9, 1.3, pm, bm, generator=g) for _ in range(n)]).view(-1)
        old = G.TOPK_SAMPLE_MIN
        G.TOPK_SAMPLE_MIN = 10 ** 9
        try:
            b = torch.stack([G.sample_top_p(logits, 0.8, 0.9, 1.3, pm, bm, generator=g) for _ in range(n)]).view(-1)
        finally:
            G.TOPK_SAMPLE_MIN = old
        self.assertTrue(torch.equal(logits, keep))
        fa, fb = torch.bincount(a, minlength=V) / n, torch.bincount(b, minlength=V) / n
        self.assertLess((fa - fb).abs().max().item(), 0.03)
        self.assertEqual(int(fa[3]), 0)
        sp, si = torch.softmax(ref / 0.8, -1).sort(dim=-1, descending=True)   # the nucleus, computed directly
        nucleus = set(si[0, (sp.cumsum(-1) - sp < 0.9)[0]].tolist())
        self.assertLessEqual(set(a.tolist()), nucleus)


if __name__ == "__main__":
    unittest.main(verbosity=2)
