"""Canonical trajectory format and the one compact dialect we train on first ("ta-v1").

A trajectory is a list of messages:
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "think": str|None, "content": str, "tool_calls": [{"name", "arguments"}]}
  {"role": "tool", "results": [str, ...]}          # one result per call, same order
Rendering is a pure function so other harness dialects (Hermes, Pi, Claude Code, ...) can be
added later as extra renderers over the same records.

ta-v1 text:
  <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n
  <|im_start|>assistant\n<think>\n...\n</think>\n<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call><|im_end|>\n
  <|im_start|>tool\n<tool_response>\nRAW RESULT TEXT\n</tool_response><|im_end|>\n
Tool results are raw text, not JSON-escaped, so copying a span from them stays a plain copy.
"""
from __future__ import annotations

import json
import re

from tiny_agent.tools import TOOL_SPECS

SYSTEM_TA_V1 = (
    "You are an agent working in /work. Find information with tools; do not answer from memory. "
    "If the information is not in the workspace, submit NOT_FOUND.\nTools:\n"
    + "\n".join(f"{n}({a}): {d}" for n, a, d in TOOL_SPECS)
)


def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    out = []
    for m in messages:
        r = m["role"]
        if r in ("system", "user"):
            out.append(f"<|im_start|>{r}\n{m['content']}<|im_end|>\n")
        elif r == "assistant":
            s = "<|im_start|>assistant\n"
            if m.get("think"):
                s += f"<think>\n{m['think']}\n</think>\n"
            if m.get("content"):
                s += m["content"]
            for c in m.get("tool_calls") or []:
                s += "<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False) + "\n</tool_call>"
            out.append(s + "<|im_end|>\n")
        elif r == "tool":
            body = "".join(f"<tool_response>\n{x}\n</tool_response>" for x in m["results"])
            out.append(f"<|im_start|>tool\n{body}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def parse_assistant(text: str) -> dict:
    """Parse one generated assistant turn (text up to <|im_end|>). Malformed calls are reported
    as {"error": ...} entries so the environment can return the error as a tool result."""
    text = text.split("<|im_end|>")[0]
    think = _THINK.search(text)
    calls = []
    for raw in _CALL.findall(text):
        try:
            obj = json.loads(raw.strip())
            if not isinstance(obj, dict) or "name" not in obj:
                raise ValueError("missing name")
            args = obj.get("arguments") or {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            calls.append({"name": str(obj["name"]), "arguments": args})
        except (json.JSONDecodeError, ValueError) as e:
            calls.append({"error": f"Error: could not parse tool call ({e}). Use JSON: "
                                   '{"name": "...", "arguments": {...}}'})
    content = _CALL.sub("", _THINK.sub("", text)).strip()
    return {"role": "assistant", "think": think.group(1).strip() if think else None,
            "content": content, "tool_calls": calls}


def repeated_calls(messages: list[dict]) -> list[int]:
    """Per assistant turn, how many of its tool calls repeat an earlier call (same name and
    arguments, canonical JSON) with no edit/write since, i.e. calls that cannot see anything new.
    Re-running tests after an edit is not a repeat. (MiMo-V2.6 found mild repetition that goes
    unpenalized gets amplified by RL into tool-call flooding.)"""
    seen, out = set(), []
    for m in messages:
        if m["role"] != "assistant":
            continue
        n = 0
        for c in m.get("tool_calls") or []:
            if "error" in c:
                continue
            if c["name"] in ("edit", "write"):
                seen.clear()
                continue
            key = (c["name"], json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False))
            n += key in seen
            seen.add(key)
        out.append(n)
    return out
