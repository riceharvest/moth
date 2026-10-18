"""Fine-tune tiny-agent-112m into moth.

  python -m moth.sft --init data/tiny-agent-112m/base --data data/sft --out runs/sft1

Packs chats into (B, T) rows with one doc id per chat (FlexAttention keeps them apart), trains only on
moth's tokens, uses the pretraining optimizers (Muon + Sinkhorn momentum) with a warmup-stable-decay schedule that
reaches 0 at the end of the run (default 1 epoch, linear decay over the last --decay-frac), and writes
model.safetensors + config.json (the HF layout) to <out>/model (best val) and <out>/ckpt_60, ckpt_80, ckpt_100
(fixed fractions of the run, see --ckpts). --engram-lr-mult scales the LR of the Engram n-gram tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch

TA = os.environ.get("TA_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TA)
from tiny_agent.checkpoint import load_model  # noqa: E402
from tiny_agent.model import make_block_mask  # noqa: E402
from tiny_agent.optim import build_optimizers, wsd_lr  # noqa: E402


def pack(rows, T, seed):
    """Greedy-pack (ids, mask) chats into rows of T tokens -> list of (inp, tgt, doc) tensors."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    out, cur = [], []

    def flush():
        inp = torch.zeros(T, dtype=torch.long)
        tgt = torch.full((T,), -1, dtype=torch.long)
        doc = torch.full((T,), 10_000, dtype=torch.long)   # padding: its own doc, no targets
        p = 0
        for k, (ids, mask) in enumerate(cur):
            n = len(ids) - 1
            inp[p:p + n] = torch.tensor(ids[:-1])
            t = torch.tensor(ids[1:])
            t[torch.tensor(mask[1:]) == 0] = -1
            tgt[p:p + n] = t
            doc[p:p + n] = k
            p += n
        out.append((inp, tgt, doc))

    used = 0
    for ids, mask in rows:
        n = len(ids) - 1
        if n > T:
            continue
        if used + n > T:
            flush()
            cur, used = [], 0
        cur.append((ids, mask))
        used += n
    if cur:
        flush()
    return out


def batches(packed, B):
    for i in range(0, len(packed) - B + 1, B):
        chunk = packed[i:i + B]
        yield tuple(torch.stack([c[j] for c in chunk]) for j in range(3))


def ckpt_steps(steps: int, fracs: str) -> dict[int, str]:
    """--ckpts 0.6,0.8,1.0 over 500 steps -> {300: "ckpt_60", 400: "ckpt_80", 500: "ckpt_100"}."""
    out = {}
    for f in filter(None, fracs.split(",")):
        out[min(steps, max(1, round(steps * float(f))))] = f"ckpt_{round(float(f) * 100)}"
    return out


def split_engram_lr(model, opts, mult: float) -> int:
    """Move the Engram tables into their own param group at mult x the LR (build_optimizers puts them in the
    Sinkhorn group with the embedding and head). Returns the number of table parameters moved."""
    if mult == 1:
        return 0
    ids = {id(m.table.weight) for m in model.engram_modules()}
    moved = 0
    for o in opts:
        for g in list(o.param_groups):
            eng = [p for p in g["params"] if id(p) in ids]
            if not eng:
                continue
            g["params"] = [p for p in g["params"] if id(p) not in ids]
            ng = {k: v for k, v in g.items() if k != "params"}
            ng.update(params=eng, lr=g["lr"] * mult, base_lr=g["base_lr"] * mult)
            o.add_param_group(ng)
            moved += sum(p.numel() for p in eng)
    return moved


def save(model, out_dir):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v).contiguous().cpu()
          for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    cfg = {k: list(v) if isinstance(v, tuple) else v for k, v in vars(model.cfg).items()}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/tiny-agent-112m/base")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", default="runs/sft1")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--micro_B", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--decay-frac", type=float, default=0.25, help="LR decays linearly to 0 over this last part of the run")
    ap.add_argument("--ckpts", default="0.6,0.8,1.0", help="fractions of the run to save as <out>/ckpt_<pct>")
    ap.add_argument("--engram-lr-mult", type=float, default=1.0, help="LR multiplier for the Engram tables")
    ap.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = load_model(a.init, device=dev, dtype=torch.float32).train()
    cfg = model.cfg
    train_rows = torch.load(os.path.join(a.data, "train.pt"))
    val = pack(torch.load(os.path.join(a.data, "val.pt")), a.T, 0)
    val = val * -(-a.micro_B // len(val)) if len(val) < a.micro_B else val   # at least one full batch
    n_rows = len(pack(train_rows, a.T, 0))
    steps = max(1, int(n_rows * a.epochs) // (a.micro_B * a.accum))
    opts = build_optimizers(model, lr=a.lr)
    moved = split_engram_lr(model, opts, a.engram_lr_mult)
    ckpts = ckpt_steps(steps, a.ckpts)
    cmodel = torch.compile(model, dynamic=False)
    print(f"{len(train_rows)} chats -> {n_rows} rows of {a.T}/epoch, {steps} steps, val rows {len(val)}; "
          f"WSD warmup {a.warmup} decay last {a.decay_frac:.0%}; checkpoints at {ckpts}; "
          f"engram tables x{a.engram_lr_mult} LR ({moved / 1e6:.1f}M params)", flush=True)

    def evaluate():
        model.eval()
        ls = []
        with torch.no_grad():
            for inp, tgt, doc in batches(val, a.micro_B):
                inp, tgt, doc = inp.to(dev), tgt.to(dev), doc.to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    ls.append(cmodel(inp, doc, make_block_mask(doc, cfg.swa_window), tgt).item())
        model.train()
        return round(sum(ls) / max(len(ls), 1), 4)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    print(json.dumps({"step": 0, "val": evaluate()}), file=log, flush=True)
    step, epoch, t0, best = 0, 0, time.time(), float("inf")
    while step < steps:
        it = batches(pack(train_rows, a.T, epoch), a.micro_B)
        epoch += 1
        while step < steps:
            m = wsd_lr(step, steps, warmup=a.warmup, decay_frac=a.decay_frac)
            for o in opts:
                for g in o.param_groups:
                    g["lr"] = g["base_lr"] * m
            loss_acc, ok = 0.0, True
            for _ in range(a.accum):
                b = next(it, None)
                if b is None:
                    ok = False
                    break
                inp, tgt, doc = (x.to(dev) for x in b)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    loss = cmodel(inp, doc, make_block_mask(doc, cfg.swa_window), tgt)
                (loss / a.accum).backward()
                loss_acc += loss.detach()
            if not ok:
                for o in opts:
                    o.zero_grad(set_to_none=True)
                break
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            step += 1
            ck = ckpts.get(step)
            if step % 10 == 0 or step == steps or ck:
                l = (loss_acc / a.accum).item()
                if not math.isfinite(l):
                    raise RuntimeError(f"non-finite loss at step {step}")
                rec = {"step": step, "epoch": epoch, "loss": round(l, 4), "lr_mult": round(m, 3),
                       "min": round((time.time() - t0) / 60, 1)}
                if step % 50 == 0 or step == steps or ck:
                    rec["val"] = evaluate()
                    if rec["val"] < best:   # keep the best-val weights too
                        best = rec["val"]
                        save(model, os.path.join(a.out, "model"))
                        rec["saved"] = True
                if ck:   # fixed points of the decay, to compare by eval rather than val loss
                    save(model, os.path.join(a.out, ck))
                    rec["ckpt"] = ck
                print(json.dumps(rec), file=log, flush=True)
                print(json.dumps(rec), flush=True)
    print(f"done after {step} steps, {(time.time() - t0) / 60:.1f} min; {a.out}/model = best val {best}; "
          f"checkpoints {sorted(ckpts.values())}", flush=True)


if __name__ == "__main__":
    main()
