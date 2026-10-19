"""moth.xs (XS inference) against tiny-moe's own forward pass and decoder.

  python -m pytest -q tests/test_xs_engine.py

Two layers:
  * reference tests (local; skipped when the tiny-agent repo with tiny_moe isn't importable): a small random-init
    TinyMoE (lean flags, and a variant with gated norm / gated attention / latent experts) against
    TinyMoE.forward and tiny_moe.generate's KV-cache decoder: whole-sequence prefill, a mixed prefill / one-token
    plan, token-by-token decode, ring wrap (> swa_window), more than one global group, PKO layers, hash layers,
    the bigram embedding, chunked prefill. fp32 tight; bf16 / int8 within tolerance.
  * golden fixture (CI: no tiny_moe there): tests/data/xs_tiny.json holds a tiny model's weights and the
    reference logits tiny_moe produced for them (regenerate: python tests/test_xs_engine.py --make-fixture).
Plus the release format round trip (export -> safetensors -> load) and the chat template.

TINY_MOE_REPO points at the tiny-agent checkout (default: /mnt/ssd/dev workspace/tiny-agent).
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIXTURE = os.path.join(ROOT, "tests", "data", "xs_tiny.json")
TINY_MOE_REPO = os.environ.get("TINY_MOE_REPO", "/mnt/ssd/dev workspace/tiny-agent")

try:
    import torch
except ImportError:   # the stdlib-only CI jobs
    torch = None
needs_torch = unittest.skipIf(torch is None, "needs torch")


def _tiny_moe():
    """tiny_moe's TinyMoE, ModelConfig and generate module, or None."""
    if torch is None or not os.path.isdir(os.path.join(TINY_MOE_REPO, "tiny_moe")):
        return None
    os.environ.setdefault("TINY_MOE_DEVICE", "cpu")
    if TINY_MOE_REPO not in sys.path:
        sys.path.append(TINY_MOE_REPO)
    try:
        from tiny_moe import generate as G
        from tiny_moe.config import ModelConfig
        from tiny_moe.model import TinyMoE
    except Exception:  # noqa: BLE001  triton missing etc.
        return None
    return TinyMoE, ModelConfig, G


TM = _tiny_moe()
needs_ref = unittest.skipIf(TM is None, f"needs tiny_moe ({TINY_MOE_REPO})")

LEAN = dict(attnres=False, gated_norm=False, gated_attn=False)


def small_cfg(**kw):
    """A small XS-shaped config: 7 layers (2 SWA-only, then groups FU|FU|F...), 3 hash layers, PKO on 3 layers,
    a 16-slot ring, 8 experts top-2."""
    base = dict(name="tiny", vocab_size=300, d_model=128, n_layers=7, head_dim=32, n_kv_heads=2, kv_group=2,
                layer_pattern="FU", swa_window=16, swa_only=2, max_seq_len=1024, pko_every=2, pko_offset=1,
                n_experts=8, top_k=2, expert_dff=48, shared_dff=64, hash_layers=3, bigram_rows=1009,
                base_width=128, moe_backend="loop", **LEAN)
    base.update(kw)
    return base


def ref_model(seed=0, **kw):
    TinyMoE, ModelConfig, _ = TM
    cfg = ModelConfig(**small_cfg(**kw))
    torch.manual_seed(seed)
    m = TinyMoE(cfg).float().eval()
    with torch.no_grad():   # zero inits (bigram, sinks, gate ups, selection bias) would hide bugs
        for n, p in m.named_parameters():
            if n == "bigram" or n.endswith((".sink", ".up", ".w")):
                p.normal_(0, 0.3)
        for l in m.layers:
            l.moe.e_bias.normal_(0, 0.5)
    return m


def to_xs(m, mode="fp32"):
    from dataclasses import asdict

    from moth.xs.model import XSModel
    from moth.xs.weights import export_state
    from moth.xs import config as xc
    tensors, cfg = export_state(m.state_dict(), asdict(m.cfg), dtype=torch.float32)
    return XSModel(xc.from_dict(cfg), tensors, mode)


def ref_logits(m, ids):
    with torch.no_grad():
        return m(ids[None], torch.zeros(1, ids.numel(), dtype=torch.long))[0]


def run_plan(xs, ids, plan, cache=None):
    """Feed ids following plan (chunk sizes; 1 = step()). Returns {position: logits} at each chunk's last token."""
    cache = cache or xs.new_cache(1024)
    out, done = {}, 0
    for n in plan:
        chunk = ids[done:done + n]
        lg = xs.step(int(chunk[0]), cache) if n == 1 else xs.forward(chunk, cache)
        done += n
        out[done - 1] = lg[0]
    assert done == ids.numel()
    return out


@needs_ref
class AgainstTinyMoE(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def check(self, ids, ref, plan, xs, tol):
        got = run_plan(xs, ids, plan)
        for p, lg in got.items():
            err = (lg - ref[p]).abs().max().item()
            self.assertLess(err, tol, (p, err))

    def test_lean_prefill_and_decode(self):
        m = ref_model()
        xs = to_xs(m)
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, 300, (90,), generator=g)
        ref = ref_logits(m, ids)
        # whole sequence at once, every position (ring wraps at 16, three global groups)
        lg = xs.forward(ids, xs.new_cache(256), all_logits=True)
        self.assertLess((lg - ref).abs().max().item(), 2e-4)
        self.check(ids, ref, [90], xs, 2e-4)
        self.check(ids, ref, [1] * 90, xs, 2e-4)                                 # token by token
        self.check(ids, ref, [7, 1, 1, 20, 1, 1, 1, 33, 1, 24], xs, 2e-4)       # mixed

    def test_chunked_prefill(self):
        from moth.xs import model as XM
        m = ref_model(seed=3)
        xs = to_xs(m)
        ids = torch.randint(0, 300, (75,), generator=torch.Generator().manual_seed(2))
        ref = ref_logits(m, ids)
        old = XM.CHUNK
        try:
            XM.CHUNK = 10   # several passes per feed, each shorter than the ring
            self.check(ids, ref, [33, 1, 41], xs, 2e-4)
            XM.CHUNK = 20   # passes longer than the ring (16)
            self.check(ids, ref, [74, 1], xs, 2e-4)
        finally:
            XM.CHUNK = old

    def test_matches_tiny_moe_decoder(self):
        """tiny_moe.generate (the KV-cache decoder tiny-moe's RL uses) and moth.xs agree token for token."""
        _, _, G = TM
        m = ref_model(seed=5)
        xs = to_xs(m)
        ids = torch.randint(0, 300, (40,), generator=torch.Generator().manual_seed(4))
        tc = G.KVCache(m, 1, 128, "cpu", dtype=torch.float32)
        a = G.append(m, tc, ids[None, :25], [25])
        xc = xs.new_cache(128)
        b = xs.forward(ids[:25], xc)
        self.assertLess((a - b).abs().max().item(), 2e-4)
        for t in ids[25:]:
            a = G.append(m, tc, t.view(1, 1), [1])
            b = xs.step(int(t), xc)
            self.assertLess((a - b).abs().max().item(), 2e-4)

    def test_gated_and_latent_variant(self):
        m = ref_model(seed=7, gated_norm=True, gated_attn=True, latent_alpha=2, n_kv_heads=1, pko_offset=0)
        xs = to_xs(m)
        self.assertFalse(xs.fast)   # gated norm / latent experts: step() is step_ref
        ids = torch.randint(0, 300, (50,), generator=torch.Generator().manual_seed(6))
        ref = ref_logits(m, ids)
        self.check(ids, ref, [50], xs, 2e-4)
        self.check(ids, ref, [20, 1, 1, 1, 27], xs, 2e-4)
        self.check(ids, ref, [1] * 50, xs, 2e-4)

    def test_fast_modes_close(self):
        """bf16 and int8 weights: same argmax almost everywhere, small typical logit error relative to the logit
        spread. (Single positions can be far off: a router near a tie picks another expert under weight noise,
        and on random weights with a random selection bias that happens; hence medians, not maxima.)"""
        m = ref_model(seed=9)
        ids = torch.randint(0, 300, (60,), generator=torch.Generator().manual_seed(8))
        ref = ref_logits(m, ids)
        for mode, rel in (("bf16", 0.1), ("int8", 0.35)):
            xs = to_xs(m, mode)
            for plan in ([60], [1] * 60, [30, 1, 1, 28]):
                got = run_plan(xs, ids, plan)
                errs = torch.tensor([((lg - ref[p]).abs().max() / ref[p].std()).item() for p, lg in got.items()])
                self.assertLess(errs.median().item(), rel, (mode, plan[:3]))
            lg = xs.forward(ids, xs.new_cache(128), all_logits=True)
            self.assertLess(((lg - ref).abs().amax(-1) / ref.std(-1)).median().item(), rel, mode)
            agree = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
            self.assertGreater(agree, 0.85, (mode, agree))


# ------------------------------------------------------------------------------------------ golden fixture (CI)
def _b64(t) -> list:
    t = t.contiguous()
    return [list(t.shape), str(t.dtype).replace("torch.", ""),
            base64.b64encode(zlib.compress(t.numpy().tobytes(), 9)).decode()]


def _unb64(v):
    import numpy as np
    shape, dt, data = v
    np_dt = {"float32": np.float32, "int32": np.int32, "int64": np.int64}[dt]
    return torch.from_numpy(np.frombuffer(zlib.decompress(base64.b64decode(data)), dtype=np_dt).copy()).reshape(shape)


FIXTURE_CFG = dict(vocab_size=128, d_model=64, n_layers=6, head_dim=16, n_kv_heads=2, swa_window=8, expert_dff=24,
                   shared_dff=32, bigram_rows=101, n_experts=6, top_k=2)


def formula_weight(name: str, shape) -> "torch.Tensor":
    """Deterministic pseudo-random weights without any RNG (integer hashing, exact on every platform), rounded to
    bf16: the fixture stores only names, shapes and tiny_moe's logits, and CI rebuilds the very same weights."""
    n = 1
    for x in shape:
        n *= x
    seed = zlib.crc32(name.encode())
    i = torch.arange(n, dtype=torch.int64)
    u = ((i * 2654435761 + seed * 40503 + (i >> 7) * 97) % 65536).float() / 65536 - 0.5   # [-0.5, 0.5), exact
    if name.endswith(("norm.weight",)):
        w = 1 + 0.5 * u
    elif name.endswith((".sink", ".e_bias")):
        w = u
    elif name == "embed":
        w = u * 3.4
    elif name == "bigram":
        w = u * 0.6
    else:
        fan_in = shape[-2] if name.endswith((".moe.w1", ".moe.w2")) else shape[-1]
        w = u * (3.4 / math.sqrt(fan_in))
    return w.reshape(shape).bfloat16().float()


def make_fixture():
    """Run with the tiny-agent repo available: tiny_moe's logits for a tiny model with formula_weight() weights
    (names, shapes, hash tables and logits go into the fixture; the weights don't need to)."""
    from dataclasses import asdict

    from moth.xs.weights import export_state
    m = ref_model(seed=11, **FIXTURE_CFG)
    shapes = {}
    with torch.no_grad():
        for n, p in list(m.named_parameters()) + [(k, b) for k, b in m.named_buffers() if k.endswith(".e_bias")]:
            p.copy_(formula_weight(n, tuple(p.shape)))
            shapes[n] = list(p.shape)
    ids = torch.randint(0, 128, (40,), generator=torch.Generator().manual_seed(12))
    ref = ref_logits(m, ids)
    tensors, cfg = export_state(m.state_dict(), asdict(m.cfg), dtype=torch.float32)
    hashes = {k: _b64(v) for k, v in tensors.items() if k.endswith(".hash_idx")}
    data = {"note": "tiny_moe reference logits for moth.xs (tests/test_xs_engine.py --make-fixture)",
            "config": cfg, "ids": ids.tolist(), "logits": _b64(ref.float()), "shapes": shapes, "hash_idx": hashes}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {FIXTURE} ({os.path.getsize(FIXTURE) // 1024} KB)")


def fixture_state():
    with open(FIXTURE) as f:
        d = json.load(f)
    sd = {k: formula_weight(k, tuple(shp)) for k, shp in d["shapes"].items()}
    sd.update({k: _unb64(v) for k, v in d["hash_idx"].items()})
    return d, sd


@needs_torch
@unittest.skipUnless(os.path.isfile(FIXTURE), "no fixture")
class GoldenFixture(unittest.TestCase):
    def setUp(self):
        from moth.xs import config as xc
        from moth.xs.model import XSModel
        d, self.sd = fixture_state()
        self.cfg = xc.from_dict(d["config"])
        self.ids = torch.tensor(d["ids"])
        self.ref = _unb64(d["logits"])
        self.XSModel = XSModel
        torch.set_num_threads(2)

    def test_fp32_matches_reference(self):
        from moth.xs import model as XM
        xs = self.XSModel(self.cfg, self.sd, "fp32")
        lg = xs.forward(self.ids, xs.new_cache(64), all_logits=True)
        self.assertLess((lg - self.ref).abs().max().item(), 2e-4)
        for plan in ([1] * 40, [5, 1, 1, 12, 1, 20]):
            for p, got in run_plan(xs, self.ids, plan, xs.new_cache(64)).items():
                self.assertLess((got - self.ref[p]).abs().max().item(), 2e-4, (plan[:3], p))
        old = XM.CHUNK
        try:
            XM.CHUNK = 6
            for p, got in run_plan(xs, self.ids, [39, 1], xs.new_cache(64)).items():
                self.assertLess((got - self.ref[p]).abs().max().item(), 2e-4, p)
        finally:
            XM.CHUNK = old

    def test_int8_close(self):
        from moth.xs.quant import int8_kernel_ok
        # (a tiny model with formula weights has near-tied top logits: argmax agreement is a loose check here)
        for mode, tol, agree in (("bf16", 0.1, 0.85), ("int8", 0.35, 0.7)) if int8_kernel_ok() else (("bf16", 0.1, 0.85),):
            xs = self.XSModel(self.cfg, self.sd, mode)
            lg = xs.forward(self.ids, xs.new_cache(64), all_logits=True)
            self.assertLess(((lg - self.ref).abs().amax(-1) / self.ref.std(-1)).median().item(), tol, mode)
            self.assertGreater((lg.argmax(-1) == self.ref.argmax(-1)).float().mean().item(), agree, mode)
            got = run_plan(xs, self.ids, [1] * 40, xs.new_cache(64))
            rel = torch.tensor([((g - self.ref[p]).abs().max() / self.ref[p].std()).item() for p, g in got.items()])
            self.assertLess(rel.median().item(), tol, mode)


# ------------------------------------------------------------------------------------------ fast decode step
def xs_state(cfg, seed=0) -> dict:
    """Random fp32 weights for an XSConfig under tiny_moe's state-dict names (no tiny_moe needed): nonzero norms
    gains, sinks, selection biases and bigram rows, so no part of the forward pass is trivially skipped."""
    c = cfg
    H, Hk, D, d, E, f, sf = c.n_heads, c.n_kv_heads, c.head_dim, c.d_model, c.n_routed, c.expert_dff, c.shared_ffn
    shapes = {"embed": (c.vocab_size, d), "final_norm.norm.weight": (d,), "bigram": (c.bigram_rows, d)}
    for i in range(c.n_layers):
        p = f"layers.{i}."
        shapes[p + "attn_norm.norm.weight"] = shapes[p + "mlp_norm.norm.weight"] = (d,)
        shapes[p + "attn.w_qg.weight"] = (H * D * (2 if c.gated_attn else 1), d)
        shapes[p + "attn.w_kv_local.weight"] = (2 * Hk * D, d)
        if c.owns_global_kv(i):
            shapes[p + "attn.w_kv_global.weight"] = (2 * Hk * D, d)
            shapes[p + "attn.kg_norm.weight"] = (D,)
        shapes[p + "attn.w_o.weight"] = (d, H * D)
        shapes[p + "attn.q_norm.weight"] = shapes[p + "attn.kl_norm.weight"] = (D,)
        shapes[p + "attn.sink"] = (H,)
        if not c.is_hash_layer(i):
            shapes[p + "moe.router"] = (E, d)
        shapes[p + "moe.e_bias"] = (E,)
        shapes[p + "moe.w1"] = (E, d, 2 * f)
        shapes[p + "moe.w2"] = (E, f, d)
        shapes[p + "moe.shared.0.w_in.weight"] = (2 * sf, d)
        shapes[p + "moe.shared.0.w_out.weight"] = (d, sf)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for k, shp in shapes.items():
        w = torch.randn(shp, generator=g)
        if k.endswith("norm.weight"):
            w = 1 + 0.3 * w
        elif k.endswith((".sink", ".e_bias")):
            w = 0.5 * w
        elif k == "bigram":
            w = 0.3 * w
        elif k != "embed":
            w = w / math.sqrt(shp[-2] if k.endswith((".w1", ".w2")) else shp[-1])
        sd[k] = w
    return sd


FAST_CFG = dict(vocab_size=300, d_model=128, n_layers=7, head_dim=32, n_kv_heads=2, kv_group=2, layer_pattern="FU",
                swa_window=16, swa_only=2, max_seq_len=512, pko_every=2, pko_offset=1, n_experts=8, top_k=2,
                expert_dff=48, shared_dff=96, hash_layers=3, bigram_rows=1009, base_width=128)


@needs_torch
class FastStep(unittest.TestCase):
    """XSModel.step's fast path (_step_fast) against step_ref, the plain decode step: bitwise equal logits in every
    weight format, so int8 / bf16 greedy decoding is unchanged by it."""

    def setUp(self):
        torch.set_num_threads(2)

    def modes(self):
        from moth.xs.quant import int8_kernel_ok
        return ("fp32", "bf16", "int8") if int8_kernel_ok() else ("fp32", "bf16")

    def run_both(self, xs, n_steps=60, seed=1):
        """Same tokens through the fast path and step_ref on two caches: decode from position 0 (ring empty, PKO's
        zero first key), past the ring (16 slots), a prefill chunk in between, then again after cache.reset()."""
        g = torch.Generator().manual_seed(seed)
        ca, cb = xs.new_cache(256), xs.new_cache(256)
        for rnd in range(2):
            toks = torch.randint(0, xs.cfg.vocab_size, (n_steps,), generator=g).tolist()
            for i, t in enumerate(toks):
                if i == n_steps // 2:   # a new message mid-decode (prefill path), then decode on
                    chunk = torch.randint(0, xs.cfg.vocab_size, (7,), generator=g)
                    self.assertTrue(torch.equal(xs.forward(chunk, ca), xs.forward(chunk, cb)))
                a, b = xs._step_fast(t, ca), xs.step_ref(t, cb)
                self.assertTrue(torch.equal(a, b), (xs.mode, rnd, i, (a - b).abs().max().item()))
            self.assertEqual(ca.len, cb.len)
            ca.reset()
            cb.reset()

    def test_bitwise_equal_xs_shape(self):
        from moth.xs.config import XSConfig
        from moth.xs.model import XSModel
        cfg = XSConfig(**FAST_CFG)
        sd = xs_state(cfg, seed=3)
        for mode in self.modes():
            xs = XSModel(cfg, sd, mode)
            self.assertTrue(xs.fast, mode)
            self.assertEqual(sorted({L.mode for L in xs.layers}), ["full", "reuse", "swa"])
            self.assertTrue(xs.pko_groups and any(L.pko for L in xs.layers) and any(L.hashed for L in xs.layers))
            self.run_both(xs)

    def test_bitwise_equal_other_flags(self):
        """Widths that aren't powers of two (RMSNorm's mean path), gated attention, sigmoid router, no sink / XSA /
        QK-norm / clamp, an embedding multiplier."""
        from moth.xs.config import XSConfig
        from moth.xs.model import XSModel
        cfg = XSConfig(**{**FAST_CFG, "d_model": 96, "head_dim": 24, "base_width": 96, "gated_attn": True,
                          "router_act": "sigmoid", "attn_sink": False, "xsa": False, "qk_norm": False,
                          "swiglu_clamp": 0.0, "emb_mult": 1.5, "expert_dff": 40, "shared_dff": 120})
        sd = xs_state(cfg, seed=4)
        for mode in self.modes():
            xs = XSModel(cfg, sd, mode)
            self.assertTrue(xs.fast, mode)
            self.run_both(xs, n_steps=40)

    def test_fallback_when_shared_expert_not_fusable(self):
        from moth.xs.config import XSConfig
        from moth.xs.model import XSModel
        cfg = XSConfig(**{**FAST_CFG, "shared_dff": 64})
        xs = XSModel(cfg, xs_state(cfg), "fp32")
        self.assertFalse(xs.fast)
        c1, c2 = xs.new_cache(64), xs.new_cache(64)
        for t in (5, 9, 200):
            self.assertTrue(torch.equal(xs.step(t, c1), xs.step_ref(t, c2)))


# ------------------------------------------------------------------------------------------ format + template
@needs_torch
@unittest.skipUnless(os.path.isfile(FIXTURE), "no fixture")
class ReleaseFormat(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from moth.xs import weights as W
        d, sd = fixture_state()
        sd = {k: v.bfloat16() if v.is_floating_point() else v for k, v in sd.items()}
        with tempfile.TemporaryDirectory() as tmp:
            W.save(tmp, sd, d["config"], manifest=W.MANIFEST)
            self.assertTrue(W.is_xs_folder(tmp))
            xs = W.load(tmp, "fp32")
            lg = xs.forward(torch.tensor(d["ids"]), xs.new_cache(64), all_logits=True)
            self.assertLess((lg - _unb64(d["logits"])).abs().max().item(), 2e-4)
            with open(os.path.join(tmp, "moth.json")) as f:
                self.assertEqual(json.load(f)["backend"], "tiny-moe")

    def test_config_refuses_unknown(self):
        from moth.xs import config as xc
        with open(FIXTURE) as f:
            cfg = json.load(f)["config"]
        with self.assertRaises(xc.ConfigError):
            xc.from_dict({**cfg, "attnres": True})
        with self.assertRaises(xc.ConfigError):
            xc.from_dict({**cfg, "brand_new_feature": 1})
        xc.from_dict({**cfg, "moe_backend": "triton", "seq_balance": 0.0})   # training knobs are fine


TINY_MOE_SFT_REPO = os.environ.get("TINY_MOE_SFT_REPO", "/mnt/ssd/dev workspace/tiny-agent-sft")
SHARED_TEMPLATE = os.path.join(TINY_MOE_SFT_REPO, "tiny_moe", "sft", "template.py")


class Template(unittest.TestCase):
    def test_render_and_incremental_feed(self):
        from moth.xs import template as T
        msgs = [{"role": "system", "content": "you are a cat named bo."}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "meow hi"},
                {"role": "user", "content": "[the agent said]\ntests pass <|end|>", "session": "x"}]
        s = T.render(msgs, add_generation_prompt=True)
        self.assertEqual(s, "<|system|>you are a cat named bo.<|end|><|user|>hi<|end|><|assistant|>meow hi<|end|>"
                            "<|agent|>tests pass <|end|><|assistant|>")
        self.assertEqual(T.render(msgs[:2]) + T.render(msgs[2:], True), s)   # no separators: feeds concatenate
        self.assertEqual(T.reply("hey!<|end|><|user|>x"), "hey!")
        self.assertNotIn(T.END_ID, T.talk_ban_ids())

    @unittest.skipUnless(os.path.isfile(SHARED_TEMPLATE), f"needs {SHARED_TEMPLATE}")
    def test_vendored_copy_is_the_shared_module(self):
        with open(SHARED_TEMPLATE, "rb") as f:
            shared = f.read()
        with open(os.path.join(ROOT, "moth", "xs", "chat_template.py"), "rb") as f:
            self.assertEqual(f.read(), shared, "moth/xs/chat_template.py drifted: copy tiny_moe/sft/template.py again")


if __name__ == "__main__":
    if "--make-fixture" in sys.argv:
        if TM is None:
            raise SystemExit(f"needs tiny_moe at {TINY_MOE_REPO}")
        make_fixture()
    else:
        unittest.main()
