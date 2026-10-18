"""Merge the outputs of several teacher boxes (runbook/fanout.sh) into one dialogs.jsonl + labels.jsonl.

  python -m moth.merge_gen results/rented/g4-* --out results/g4

Each input is a box's copied out/ dir (dialogs.jsonl, optional labels.jsonl; its pilot/ subdir is never merged).
Checks that the boxes' seed ranges are disjoint (fan-out gives each box its own range), drops duplicate seeds
(first box wins, counted), keeps only labels of merged seeds, and prints per box: chats, seed range, types, card
mix and the ON-gist rate. Then: python -m moth.build_sft results/g4/dialogs.jsonl --labels results/g4/labels.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass   # a line cut off by a box dying mid-write
    return out


def merge(dirs: list[str]) -> tuple[list[dict], dict[int, dict], list[dict], list[str]]:
    """-> (dialogs, {seed: label line}, per-box summaries, warnings)."""
    seen, dialogs, labels, boxes, warn = set(), [], {}, [], []
    ranges = []
    for d in dirs:
        ds = read_jsonl(os.path.join(d, "dialogs.jsonl"))
        lab = {x["seed"]: x for x in read_jsonl(os.path.join(d, "labels.jsonl")) if x.get("labels") is not None}
        dup = 0
        for x in ds:
            if x["seed"] in seen:
                dup += 1
                continue
            seen.add(x["seed"])
            dialogs.append(x)
            if x["seed"] in lab:
                labels[x["seed"]] = lab[x["seed"]]
        seeds = [x["seed"] for x in ds]
        lo, hi = (min(seeds), max(seeds)) if seeds else (None, None)
        if seeds:
            ranges.append((lo, hi, d))
        replies = sum(x["n"] for x in lab.values())
        on = sum(sum(lb["on"] for lb in x["labels"]) for x in lab.values())
        cards = collections.Counter(("moth" if (x.get("card") or {}).get("id") == "moth" else
                                     "blank" if (x.get("card") or {}).get("id") == "blank" else
                                     "pool" if x.get("card") else "none") for x in ds)
        boxes.append({"dir": d, "chats": len(ds), "dup": dup, "seeds": (lo, hi), "labelled": len(lab),
                      "on_gist": on / replies if replies else None,
                      "types": dict(collections.Counter(x.get("type", "?") for x in ds)), "cards": dict(cards)})
        if dup:
            warn.append(f"{d}: {dup} seeds already merged from an earlier box (dropped)")
        if ds and not lab:
            warn.append(f"{d}: no labels.jsonl (run: python -m moth.gist label {d}/dialogs.jsonl on a box)")
    ranges.sort()
    for (lo1, hi1, d1), (lo2, hi2, d2) in zip(ranges, ranges[1:]):
        if lo2 <= hi1:
            warn.append(f"seed ranges overlap: {d1} {lo1}..{hi1} and {d2} {lo2}..{hi2}")
    return dialogs, labels, boxes, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="box result dirs (each with dialogs.jsonl [labels.jsonl])")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dialogs, labels, boxes, warn = merge(a.dirs)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dialogs.jsonl"), "w") as f:
        for x in sorted(dialogs, key=lambda x: x["seed"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for s in sorted(labels):
            f.write(json.dumps(labels[s], ensure_ascii=False) + "\n")
    for b in boxes:
        og = f"{b['on_gist']:.1%}" if b["on_gist"] is not None else "n/a"
        print(f"{b['dir']}: {b['chats']} chats, seeds {b['seeds'][0]}..{b['seeds'][1]}, labelled {b['labelled']}, "
              f"on_gist {og}, types {b['types']}, cards {b['cards']}")
    rep = sum(x["n"] for x in labels.values())
    on = sum(sum(lb["on"] for lb in x["labels"]) for x in labels.values())
    print(f"merged {len(dialogs)} chats ({len(labels)} labelled, on_gist {on / max(rep, 1):.1%}) -> {a.out}")
    for w in warn:
        print("warning:", w)


if __name__ == "__main__":
    main()
