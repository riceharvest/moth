"""XS weights on disk: the release format, the loader, and the exporter from a tiny-moe training checkpoint.

A model folder (what the Hub repo holds and moth downloads):
  model.safetensors   tiny_moe.model.TinyMoE's state dict names, floats in bf16; plus layers.<i>.moe.hash_idx
                      (int32, the hash layers' fixed token -> experts table: a non-persistent buffer in tiny_moe,
                      stored here so inference never depends on torch's RNG); training-only buffers left out
  config.json         tiny_moe.config.ModelConfig fields + "arch": "tiny-moe" (moth.xs.config reads it)
  tokenizer.json      the 48k tokenizer with the chat specials (ids 0..31 fixed, moth.xs.template)
  moth.json           {"backend": "tiny-moe", "needs_moth": ..., ...} (moth/backends/manifest.py)

  python -m moth.xs.weights export <tiny_moe ckpt .pt> <out dir> --tokenizer tokenizer.json [--version V]
      (final_ema.pt / ema_*.pt hold no router buffers: e_bias comes from the sibling final.pt / snap_*.pt, or --buffers)
  python -m moth.xs.weights info <folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping

from moth.xs import config as xsconfig

DROP = (".moe.load", ".moe.qsum", ".moe.nsum", ".moe.dropped", ".theta", ".fallback_frac")
NEEDS_MOTH = "0.4.0"   # the first moth with the tiny-moe backend (moth.json "needs_moth"; older ones say "update")
MANIFEST = {"backend": "tiny-moe", "character": "moth", "needs_moth": NEEDS_MOTH,
            # the sampler XS's RL trains with: plain softmax at its serving temperature (manifest.XS_SAMPLING)
            "sampling": {"temperature": 0.7, "top_p": 1.0, "rep_penalty": 1.0, "max_tokens": 200}}


def read_config(folder: str) -> xsconfig.XSConfig:
    path = os.path.join(folder, "config.json")
    with open(path, encoding="utf-8") as f:
        return xsconfig.from_dict(json.load(f), path)


def is_xs_folder(folder: str) -> bool:
    """config.json says arch tiny-moe (stdlib: the backend resolver asks without torch)."""
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d, dict) and d.get("arch") == xsconfig.ARCH
    except (OSError, ValueError):
        return False


class _Lazy(Mapping):
    """model.safetensors as a read-on-demand mapping: XSModel converts one tensor at a time, so loading never holds
    the whole bf16 file and its int8 copy at once."""

    def __init__(self, f):
        self.f, self.names = f, set(f.keys())

    def __getitem__(self, k):
        if k not in self.names:
            raise KeyError(k)
        return self.f.get_tensor(k)

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def load(folder: str, mode: str = "int8", device: str = "cpu"):
    """XSModel from a model folder in weight format mode (moth.xs.quant.resolve_mode applied)."""
    from safetensors import safe_open

    from moth.xs.model import XSModel
    from moth.xs.quant import resolve_mode
    cfg = read_config(folder)
    mode = resolve_mode(mode, device)
    with safe_open(os.path.join(folder, "model.safetensors"), framework="pt") as f:
        return XSModel(cfg, _Lazy(f), mode, device)


def export_state(sd: dict, cfg: dict, dtype=None) -> tuple[dict, dict]:
    """(tensors, config.json dict) for the release format from a TinyMoE state dict + its ModelConfig dict.
    dtype: float tensors cast to it (default bf16; tests keep fp32)."""
    import torch

    from moth.xs.model import hash_table
    dtype = dtype or torch.bfloat16
    cfg = dict(cfg)
    qat = int(cfg.get("qat_experts") or 0)
    cfg["qat_experts"] = 0
    cfg["arch"] = xsconfig.ARCH
    c = xsconfig.from_dict({k: v for k, v in cfg.items() if k in xsconfig.KNOWN})
    out = {}
    for k, v in sd.items():
        if k.endswith(DROP) or k.endswith(".moe.hash_idx"):
            continue
        if qat and k.endswith((".moe.w1", ".moe.w2")):   # bake the int4 QAT the model trained against
            v = _fake_int4(v, qat)
        out[k] = v.detach().to("cpu", dtype=dtype if v.is_floating_point() else v.dtype).contiguous()
    for i in range(c.n_layers):
        if c.is_hash_layer(i):
            t = sd.get(f"layers.{i}.moe.hash_idx")
            if t is None:
                t = hash_table(c.vocab_size, c.n_routed, c.top_k_eff, i)
            out[f"layers.{i}.moe.hash_idx"] = t.detach().cpu().to(torch.int32).contiguous()
    return out, {k: v for k, v in cfg.items() if k in xsconfig.KNOWN}


def _fake_int4(w, group: int):
    """tiny_moe.qat.fake_int4 along the reduction dim (-2 of the (E, K, N) expert tensors), forward value only."""
    wt = w.float().movedim(-2, -1)
    shp = wt.shape
    g = wt.reshape(*shp[:-1], shp[-1] // group, group)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    q = (g / s).round().clamp(-8, 7) * s
    return q.reshape(shp).movedim(-1, -2).to(w.dtype)


def save(folder: str, tensors: dict, cfg: dict, tokenizer: str | None = None, manifest: dict | None = None):
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    save_file(tensors, os.path.join(folder, "model.safetensors"), metadata={"format": "pt", "arch": xsconfig.ARCH})
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    if tokenizer:
        shutil.copyfile(tokenizer, os.path.join(folder, "tokenizer.json"))
    if manifest is not None:
        with open(os.path.join(folder, "moth.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")


BUFFERS = (".moe.e_bias",)   # router state a HostEMA file (final_ema.pt, ema_*.pt) doesn't hold


def sibling_with_buffers(path: str) -> str | None:
    """final_ema.pt -> final.pt, ema_<x>.pt -> snap_<x>.pt: the same run and step, with the router buffers."""
    d, b = os.path.split(path)
    for pre, rep in (("final_ema", "final"), ("ema_", "snap_")):
        if b.startswith(pre):
            cand = os.path.join(d, rep + b[len(pre):])
            return cand if os.path.isfile(cand) else None
    return None


def _fill_buffers(sd: dict, cfg: dict, ckpt: str, buffers: str | None, fallback: dict | None = None) -> dict:
    """The router selection bias (e_bias) of every routed layer: from the checkpoint itself, else --buffers, else the
    sibling final.pt / snap_*.pt, else (a train.py ckpt.pt exported with --ema) its own model weights. Exporting
    without it would silently route with a zero bias, so that is an error."""
    import torch
    n_layers, hashed = int(cfg.get("n_layers", 12)), int(cfg.get("hash_layers", 3))
    need = [f"layers.{i}.moe.e_bias" for i in range(hashed, n_layers)]
    missing = [k for k in need if k not in sd]
    if not missing:
        return sd
    src = buffers or sibling_with_buffers(ckpt)
    other = fallback
    if src:
        o = torch.load(src, map_location="cpu", weights_only=False)
        other = o.get("model", o)
    if not other or any(k not in other for k in missing):
        raise SystemExit(f"{ckpt} lacks the router buffers ({missing[0]} ...) and no --buffers file / sibling "
                         "final.pt or snap_*.pt has them")
    sd = dict(sd)
    for k in missing:
        sd[k] = other[k]
    print(f"router buffers (e_bias) taken from {src or 'the checkpoint model weights'}")
    return sd


def export(ckpt: str, out: str, tokenizer: str, version: str | None = None, ema: bool = False,
           buffers: str | None = None) -> None:
    """A tiny_moe/train.py checkpoint (final.pt / final_ema.pt / snap_*.pt / ema_*.pt: {"model", "config"};
    ckpt.pt: {"model", "meta", "ema"} with the config under meta) -> a model folder."""
    import torch
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("ema") if ema and obj.get("ema") else obj["model"]
    cfg = obj.get("config") or (obj.get("meta") or {}).get("config")
    if not isinstance(cfg, dict):
        raise SystemExit(f"{ckpt}: no model config in the checkpoint (expected 'config' or meta.config)")
    sd = _fill_buffers(sd, cfg, ckpt, buffers, obj["model"] if sd is not obj["model"] else None)
    tensors, cfg = export_state(sd, cfg)
    from tokenizers import Tokenizer

    from moth.xs import template
    tok = Tokenizer.from_file(tokenizer)
    template.check_tokenizer(tok)
    if tok.get_vocab_size() != cfg["vocab_size"]:
        raise SystemExit(f"tokenizer has {tok.get_vocab_size()} tokens, the model {cfg['vocab_size']}")
    man = dict(MANIFEST)
    if version:
        man["version"] = version
    save(out, tensors, cfg, tokenizer, man)
    n = sum(t.numel() for t in tensors.values())
    print(f"wrote {out}: {n / 1e6:.1f}M values, config {cfg.get('name', '?')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m moth.xs.weights")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="tiny_moe training checkpoint -> moth model folder")
    e.add_argument("ckpt")
    e.add_argument("out")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--version", default=None)
    e.add_argument("--ema", action="store_true", help="export a train.py ckpt.pt's EMA weights")
    e.add_argument("--buffers", default=None, help="file to take the router buffers (e_bias) from when the checkpoint "
                                                   "has none (default: the sibling final.pt / snap_*.pt of an EMA file)")
    i = sub.add_parser("info", help="what a model folder holds")
    i.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export(a.ckpt, a.out, a.tokenizer, a.version, a.ema, a.buffers)
        return 0
    cfg = read_config(a.folder)
    print(json.dumps(xsconfig.to_dict(cfg), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
