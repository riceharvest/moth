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
