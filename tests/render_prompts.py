"""Offline check of the teacher prompts (no server, no GPU): renders what the teacher sees for a few seeds and, when
the Qwen3.8-27B tokenizer is on disk, measures the shared prefix (TEACHER_CORE as the chat template renders it; pure
chat since 2026-10-19, so no tools block) against the hybrid prefix cache's 1568-token blocks.

  python tests/render_prompts.py [--seeds 1,2,3] [--tokenizer DIR] [--full]

Prints per seed: the chat type, the card, the student system prompt and the tail of the teacher's view (the card
section + the first user turn with its teacher-only reminder). --full also prints the whole teacher system prompt once.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if "openai" not in sys.modules:   # gen_dialogs imports AsyncOpenAI at the top; this check never calls it
    try:
        import openai  # noqa: F401
    except ImportError:
        sys.modules["openai"] = types.ModuleType("openai")
        sys.modules["openai"].AsyncOpenAI = object
from moth import gen_dialogs as G  # noqa: E402
from moth.cards import student_system  # noqa: E402
from moth.persona import TEACHER_CORE  # noqa: E402

BLOCK = 1568
TOK_GLOB = os.path.expanduser("/mnt/ssd/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*")


def prefix_tokens(tok_dir: str) -> tuple[int, int]:
    """(tokens of TEACHER_CORE as the chat template renders it, cached tokens with whole 1568-blocks)."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(tok_dir)
    mark = "⁣ZZMARK"
    s = t.apply_chat_template([{"role": "system", "content": TEACHER_CORE + mark}, {"role": "user", "content": "hi"}],
                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    n = len(t.encode(s[:s.index(mark)]))
    return n, n // BLOCK * BLOCK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3,11,26,40")
    ap.add_argument("--tokenizer", default=next(iter(sorted(glob.glob(TOK_GLOB))), ""))
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.full:
        print("=" * 30, "TEACHER_CORE (shared by every call)", "=" * 30)
        print(TEACHER_CORE)
    for seed in (int(x) for x in a.seeds.split(",")):
        v = G.first_teacher_view(seed)
        print("=" * 30, f"seed {seed}: {v['type']} chat, card {v['card']['id']}", "=" * 30)
        print("--- student system prompt ---")
        print(student_system(v["card"]))
        print("--- teacher system prompt: TEACHER_CORE + ---")
        print(v["system"][len(TEACHER_CORE):].strip())
        print("--- first teacher-view user turn ---")
        print(v["user"])
    if a.tokenizer and os.path.isdir(a.tokenizer):
        n, cached = prefix_tokens(a.tokenizer)
        print(f"\nshared prefix (TEACHER_CORE): {n} tokens; cached with {BLOCK}-token blocks: {cached} "
              f"({cached / n:.0%}), cold per call {n - cached}")
    else:
        print("\n(no Qwen tokenizer on disk: prefix size not measured)")


if __name__ == "__main__":
    random.seed(0)
    main()
