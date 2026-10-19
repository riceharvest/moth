"""The XS chat template: ONE module for SFT, the moth app and the RL bridge (stdlib only; a tokenizer is passed in).

    <|system|>{card system prompt}<|end|><|user|>{friend}<|end|><|agent|>{coding-agent message}<|end|>
    <|assistant|>{bot reply}<|end|> ...

No separators between turns, so render(a + b) == render(a) + render(b): an incremental feed (KV cache kept between
turns, as the RL bridge and moth.serve do) appends render(new_messages, add_generation_prompt=True) after the reply
the model generated itself (ending in <|end|>). Generation prompt = "<|assistant|>", stop token = "<|end|>" (id 6).
Special ids are fixed by tiny_moe/data/sources.SPECIAL_TOKENS (the 48k tokenizer and every shard use them).

Messages: {"role": "system" | "user" | "assistant" | "agent", "content": str}. The moth code base writes a coding-agent
message as a USER turn starting with "[the agent said]" (moth.agent.agent_turn); render() accepts that form too and
turns it into an <|agent|> turn with the tag stripped. parse() always returns the canonical "agent" role; to_moth()
converts back.

Special-token strings inside content ("<|end|>" pasted by a user, WildChat text, ...) are removed by render/encode,
so content can never open or close a turn.

encode(tok, messages) -> (ids, mask): the loss mask is 1 on the assistant's content tokens and its closing <|end|>
(the model must learn to stop), 0 on the <|assistant|> header and on every system / user / agent token. Encoding is
done segment by segment (header id, encode(content), end id); that equals tok.encode(render(messages)) because the
markers are added special tokens (tests check it).
"""
from __future__ import annotations

import re

SYS, USR, AST, AGT, END, EOS, PAD = ("<|system|>", "<|user|>", "<|assistant|>", "<|agent|>", "<|end|>",
                                     "<|endoftext|>", "<|pad|>")
EOS_ID, PAD_ID, SYS_ID, USR_ID, AST_ID, AGT_ID, END_ID = 0, 1, 2, 3, 4, 5, 6
ROLE_TAG = {"system": SYS, "user": USR, "assistant": AST, "agent": AGT}
TAG_ROLE = {v: k for k, v in ROLE_TAG.items()}
ROLE_ID = {"system": SYS_ID, "user": USR_ID, "assistant": AST_ID, "agent": AGT_ID}
AGENT_TAG = "[the agent said]"       # = moth.agent.AGENT_TAG
GEN_PROMPT = AST
STOP = END

_SPECIAL = re.compile(r"<\|(?:endoftext|pad|system|user|assistant|agent|end|reserved_\d+)\|>")
_TURN = re.compile(r"(<\|system\|>|<\|user\|>|<\|assistant\|>|<\|agent\|>)(.*?)(<\|end\|>|$)", re.S)


def clean(text: str) -> str:
    """Content with every special-token string removed."""
    return _SPECIAL.sub("", text)


def canon(messages: list[dict]) -> list[dict]:
    """Canonical roles: a user turn that starts with "[the agent said]" becomes an agent turn (tag + one newline
    stripped). Other fields of a message are dropped."""
    out = []
    for m in messages:
        r, c = m["role"], m.get("content") or ""
        if r == "user" and c.startswith(AGENT_TAG):
            r, c = "agent", c[len(AGENT_TAG):]
            c = c[1:] if c.startswith("\n") else c.lstrip(" ")
        if r not in ROLE_TAG:
            raise ValueError(f"role {r!r}: the XS template has system/user/assistant/agent only (no tools)")
        out.append({"role": r, "content": c})
    return out


def to_moth(messages: list[dict]) -> list[dict]:
    """agent turns back to moth's form (a user turn "[the agent said]\\n...")."""
    return [{"role": "user", "content": f"{AGENT_TAG}\n{m['content']}"} if m["role"] == "agent" else dict(m)
            for m in messages]


def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    s = "".join(f"{ROLE_TAG[m['role']]}{clean(m['content'])}{END}" for m in canon(messages))
    return s + (GEN_PROMPT if add_generation_prompt else "")


def parse(text: str) -> list[dict]:
    """render() inverse. A trailing turn without <|end|> (a reply still being generated) is returned with
    "open": True. Text outside turns (EOS, padding) is ignored."""
    out = []
    for tag, body, end in _TURN.findall(_strip_outer(text)):
        m = {"role": TAG_ROLE[tag], "content": body}
        if not end:
            m["open"] = True
        out.append(m)
    return out


def _strip_outer(text: str) -> str:
    return text.replace(EOS, "").replace(PAD, "")


def reply(text: str) -> str:
    """A generated assistant turn (text after the generation prompt) cut at its first stop token."""
    return clean(text.split(END, 1)[0])


def encode(tok, messages: list[dict], add_generation_prompt: bool = False,
           train_roles: tuple[str, ...] = ("assistant",)) -> tuple[list[int], list[int]]:
    """(ids, loss mask) for a chat. tok: a tokenizers.Tokenizer (anything with .encode(text,
    add_special_tokens=False).ids). The mask is 1 on the content + <|end|> of turns whose role is in train_roles."""
    ids, mask = [], []
    for m in canon(messages):
        c = tok.encode(clean(m["content"]), add_special_tokens=False).ids if m["content"] else []
        t = 1 if m["role"] in train_roles else 0
        ids += [ROLE_ID[m["role"]]] + c + [END_ID]
        mask += [0] + [t] * (len(c) + 1)
    if add_generation_prompt:
        ids.append(AST_ID)
        mask.append(0)
    return ids, mask


def decode_ids(tok, ids: list[int]) -> list[dict]:
    """parse() of token ids (special tokens kept)."""
    return parse(tok.decode(list(ids), skip_special_tokens=False))


def check_tokenizer(tok) -> None:
    """The tokenizer has the fixed special ids this template hard-codes."""
    for s, i in ((EOS, EOS_ID), (PAD, PAD_ID), (SYS, SYS_ID), (USR, USR_ID), (AST, AST_ID), (AGT, AGT_ID), (END, END_ID)):
        got = tok.token_to_id(s)
        if got != i:
            raise ValueError(f"tokenizer: {s} has id {got}, the XS template needs {i}")
