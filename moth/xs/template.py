"""The XS chat template as moth uses it: moth/xs/chat_template.py is a VERBATIM copy of the shared module
tiny_moe/sft/template.py (tiny-agent branch moe-sft, commit 87fac32: one template for SFT, this app and the RL bridge),
re-exported here plus moth's own helpers. Vendored, not imported: a pip install has no tiny-agent repo, and importing
tiny_moe pulls in triton. tests/test_xs_engine.py checks the copy is byte-identical to the shared file whenever that
repo is on disk (TINY_MOE_SFT_REPO); to update it, copy the file over again.

    <|system|>{card system prompt}<|end|><|user|>{friend}<|end|><|agent|>{coding-agent message}<|end|>
    <|assistant|>{bot reply}<|end|> ...

No separators between turns: moth.serve feeds render(new messages, add_generation_prompt=True) right after the reply the
model generated (which ends in <|end|>). Stop = <|end|> (id 6). moth keeps a coding-agent message as a user turn
starting with "[the agent said]" (moth.agent.agent_turn); render() turns it into an <|agent|> turn. Special-token
strings inside content are removed, so content can never open or close a turn. No tools.
"""
from __future__ import annotations

from moth.xs.chat_template import (AGENT_TAG, AGT, AGT_ID, AST, AST_ID, END, END_ID, EOS, EOS_ID, GEN_PROMPT,  # noqa: F401
                                   PAD, PAD_ID, ROLE_ID, ROLE_TAG, STOP, SYS, SYS_ID, USR, USR_ID, canon,
                                   check_tokenizer, clean, decode_ids, encode, parse, render, reply, to_moth)

N_SPECIAL = 32   # ids 0..31: the specials + reserved slots (tiny_moe/data/sources.SPECIAL_TOKENS)


def talk_ban_ids() -> list[int]:
    """Ids an assistant turn must never sample: every special but the stop (no new turn can open inside a reply)."""
    return [i for i in range(N_SPECIAL) if i != END_ID]
