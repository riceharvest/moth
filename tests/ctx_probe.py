"""How does moth's reply quality hold up deep into a long conversation?

Loss on moth's tokens of held-out chats, placed (a) right after the system prompt and (b) after N tokens of
earlier conversation (other held-out chats' turns, one continuous session). Rising loss = context too long.
  python tests/ctx_probe.py --model runs/v1/model --data data/sft_v1
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("TA_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer  # noqa: E402
from tiny_agent.chat import render  # noqa: E402
from tiny_agent.checkpoint import load_model  # noqa: E402
from tiny_agent.model import make_block_mask  # noqa: E402
from moth.persona import SYSTEM_STUDENT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/sft_v1")
ap.add_argument("--tokenizer", default="data/tiny-agent-112m/tokenizer.json")
ap.add_argument("--offsets", default="0,500,1000,1500,2000,2500,3000,3500")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--device", default="xpu")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
model = load_model(a.model, device=a.device, dtype=torch.bfloat16).eval()
sys_ids = tok.encode(render([{"role": "system", "content": SYSTEM_STUDENT}]), add_special_tokens=False).ids
rows = torch.load(os.path.join(a.data, "val.pt"))
bodies = [(ids[len(sys_ids):], mask[len(sys_ids):]) for ids, mask in rows if ids[:len(sys_ids)] == sys_ids]
if not bodies:   # val.pt built with an older SYSTEM_STUDENT: rebuild it (python -m moth.build_sft ... --out data/probe)
    sys.exit(f"0 of {len(rows)} held-out chats start with the current system prompt: rebuild {a.data}")
rng = random.Random(0)
targets = bodies[: a.n]
print(f"{len(bodies)} held-out chats, {len(targets)} targets, system prompt {len(sys_ids)} tokens")


@torch.no_grad()
def loss_at(target, offset):
    filler = []
    pool = [b for b in bodies if b is not target]
    while len(filler) < offset:
        filler += rng.choice(pool)[0]
    filler = filler[:offset]
    if offset and filler:   # end the filler on a turn boundary so the target starts a fresh user turn
        last = max((i for i, t in enumerate(filler) if t == tok.token_to_id("<|im_end|>")), default=-1)
        filler = filler[: last + 1] + tok.encode("\n", add_special_tokens=False).ids
    ids = sys_ids + filler + target[0]
    mask = [0] * (len(sys_ids) + len(filler)) + target[1]
    x = torch.tensor([ids[:-1]], device=a.device)
    y = torch.tensor([ids[1:]], device=a.device)
    m = torch.tensor([mask[1:]], device=a.device).bool()
    doc = torch.zeros_like(x)
    with torch.autocast(a.device, dtype=torch.bfloat16):
        logits = model(x, doc, make_block_mask(doc, model.cfg.swa_window))
    ce = F.cross_entropy(logits[0].float(), y[0], reduction="none")
    return ce[m[0]].mean().item(), len(ids)


for off in [int(o) for o in a.offsets.split(",")]:
    ls, lens = zip(*(loss_at(t, off) for t in targets))
    print(f"offset {off:5d}  (context up to ~{max(lens):5d} tokens)  moth-token loss {sum(ls) / len(ls):.4f}", flush=True)
