"""The backend interface moth.serve talks to, and helpers the general backends share. Stdlib only.

A backend loads a model and turns moth's conversation (tiny_agent.chat's canonical messages: system / user /
assistant with tool_calls / tool with results) into the model's next assistant turn:

  tokenizer()   an object with the bits of a `tokenizers.Tokenizer` moth.serve uses: encode(text,
                add_special_tokens=False).ids, decode(ids, skip_special_tokens=False), token_to_id(s), get_vocab(),
                get_added_tokens_decoder(). It drives the context budget, the repetition penalty's word ids and the
                phrase ban's word classes (moth/norepeat.py).
  count(messages)   tokens the whole prompt takes (the context budget; approximate for openai)
  generate(messages, max_tokens, temperature, top_p, rep_penalty, pen_ids, ban_tools, logits_hook, stop)
                the assistant turn as text (may hold <tool_call> JSON when the manifest enables tools).
                pen_ids: token ids the repetition penalty hits; logits_hook: moth's phrase ban
                (norepeat.BanHook(out_ids, logits) -> logits), only for backends with phrase_ban.
Capabilities: stateful (moth.serve keeps a KV cache across turns and feeds only the new text: tiny-agent),
tools (can follow moth's <tool_call> format; the manifest still has to switch tools on), graphs (device graphs),
phrase_ban (a logits hook works: needs token-level sampling).
"""
from __future__ import annotations

import json
import re


class Backend:
    name = "?"
    stateful = False
    tools = False
    graphs = False
    phrase_ban = False

    def tokenizer(self):
        raise NotImplementedError

    def count(self, messages: list[dict]) -> int:
        raise NotImplementedError

    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float, top_p: float,
                 rep_penalty: float = 1.0, pen_ids=(), ban_tools: bool = False, logits_hook=None, stop=()) -> str:
        raise NotImplementedError

    def capabilities(self) -> dict:
        return {k: getattr(self, k) for k in ("stateful", "tools", "graphs", "phrase_ban")}


def chat_messages(messages: list[dict]) -> list[dict]:
    """moth's canonical messages as plain {"role", "content"} chat messages for a chat template / an OpenAI server.
    Tool calls stay in the assistant text as moth writes them (<tool_call> JSON); tool results go back as a user
    message of <tool_response> blocks (OpenAI's role "tool" would need a tool_call_id, and many small-model templates
    don't know it). Moth's own bookkeeping keys ("session") never reach the model."""
    out = []
    for m in messages:
        r = m["role"]
        if r in ("system", "user"):
            out.append({"role": r, "content": m["content"]})
        elif r == "assistant":
            s = m.get("content") or ""
            for c in m.get("tool_calls") or []:
                if "name" in c:
                    s += ("\n" if s else "") + "<tool_call>\n" + json.dumps(
                        {"name": c["name"], "arguments": c.get("arguments", {})}, ensure_ascii=False) + "\n</tool_call>"
            out.append({"role": "assistant", "content": s})
        elif r == "tool":
            out.append({"role": "user", "content": "".join(f"<tool_response>\n{x}\n</tool_response>"
                                                           for x in m["results"])})
    return out


def cut_at_stops(text: str, stop=()) -> str:
    for s in stop:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text


class _Enc:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


class PseudoTokenizer:
    """For backends without a local tokenizer (openai): words and punctuation get stable ids for this process, so the
    context budget, the repetition-penalty bookkeeping and tool-result cutting work. No vocabulary (no phrase ban)."""
    WORD = re.compile(r"\w+|[^\w\s]", re.U)
    MAX = 200_000

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> _Enc:
        out = []
        for w in self.WORD.findall(text):
            i = self.ids.get(w)
            if i is None:
                if len(self.words) >= self.MAX:   # a server that runs for weeks: start over, ids are per turn anyway
                    self.ids.clear()
                    self.words.clear()
                i = self.ids[w] = len(self.words)
                self.words.append(w)
            out.append(i)
        return _Enc(out)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(self.words[i] for i in ids if 0 <= i < len(self.words))

    def token_to_id(self, s: str):
        return None

    def get_vocab(self) -> dict:
        return {}

    def get_added_tokens_decoder(self) -> dict:
        return {}


def approx_tokens(messages: list[dict]) -> int:
    """~3 characters per token (a little high for English, so the budget errs on the safe side) + 4 per message."""
    return sum(len(m.get("content") or "") for m in chat_messages(messages)) // 3 + 4 * len(messages) + 3
