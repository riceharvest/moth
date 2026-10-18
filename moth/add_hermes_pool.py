"""Add real Hermes Agent final messages to moth's agent-message pool (run on a rented box, not locally):

  uv run --with pyarrow --with huggingface_hub python -m moth.add_hermes_pool [--max 4000]

Source: lambda/hermes-agent-reasoning-traces (Apache-2.0). Per session, the last assistant message that has text
(the reply the user read at the end) goes into moth/agent_pool.json with its source, next to the SWE-agent ones.
"""
from __future__ import annotations

import argparse
import json
import os
import random

DATASET = "lambda/hermes-agent-reasoning-traces"
HERE = os.path.dirname(os.path.abspath(__file__))


def turns(row: dict) -> list[tuple[str, str]]:
    """(role, text) pairs from ShareGPT ("conversations": from/value) or OpenAI ("messages": role/content) rows."""
    for key in ("messages", "conversations", "conversation"):
        conv = row.get(key)
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except ValueError:
                conv = None
        if hasattr(conv, "tolist"):
            conv = conv.tolist()
        if isinstance(conv, list) and conv:
            out = []
            for m in conv:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or m.get("from") or "")
                c = m.get("content", m.get("value", ""))
                if isinstance(c, list):
                    c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
                out.append((role, str(c or "")))
            return out
    return []


def final_text(row: dict) -> str:
    for role, text in reversed(turns(row)):
        if role in ("assistant", "gpt", "model"):
            t = text.split("</think>")[-1].strip()   # drop reasoning if it's inline
            if t and "<tool_call>" not in t and len(t.split()) >= 12:
                return t
    return ""


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=4000)
    a = ap.parse_args()
    root = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"])
    found = []
    for dirpath, _, names in os.walk(os.path.join(root, "data")):
        for n in sorted(names):
            if n.endswith(".parquet"):
                for row in pq.read_table(os.path.join(dirpath, n)).to_pylist():
                    t = final_text(row)
                    if t and len(t.split()) <= 500:
                        found.append({"text": t, "source": DATASET})
    random.Random(0).shuffle(found)
    path = os.path.join(HERE, "agent_pool.json")
    pool = json.load(open(path))
    pool = [p for p in pool if p["source"] != DATASET] + found[: a.max]
    json.dump(pool, open(path, "w"), ensure_ascii=False)
    print(f"{len(found)} Hermes final messages found, added {min(len(found), a.max)}; pool now {len(pool)}")
    if found:
        print("example:", found[0]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
