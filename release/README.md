---
license: apache-2.0
language:
- en
- zh
- es
- fr
- hi
datasets:
{{DATASETS_YAML}}
tags:
- tiny
- companion
- chatbot
- character
- agent-companion
- mixture-of-experts
- from-scratch
- desktop-pet
pipeline_tag: text-generation
---

<!-- TEMPLATE (moth 0.4, XS). Fill every {{PLACEHOLDER}} before an upload: release/build.sh refuses to build while any
     are left. Legal sections (Licenses, data disclosure) need the owner's go before they ship. -->

> [!NOTE]
> **moth is a silly little chat buddy, not an assistant.** A tiny model that's warm, curious and a bit dim, gets
> things sweetly wrong, and will not fix your code. It keeps you company while your coding agent works.
>
> By [@imdariotoo](https://x.com/imdariotoo)

# moth

{{SAMPLE_CHATS}}

## What it is

- **XS**: a {{TOTAL_PARAMS}}-parameter mixture-of-experts model ({{ACTIVE_PARAMS}} active per token: 64 experts,
  top-6, plus one shared expert; CSA2 attention with a shared global KV per layer group and 128-token local windows),
  pretrained from scratch on conversation-heavy text in five languages (English, Chinese, Spanish, French, Hindi),
  then SFT'd and RL'd to be a fun, dim, curious chat buddy.
- **Pure chat.** No tools, no files, no shell. It just talks.
- **Any identity.** Who it is comes only from a one-line card in its system prompt. moth (a small moth who lives on
  a graphics card) is the default; `moth card set "you are a cat named bo"` makes it a cat named bo.
- **Agent companion.** Hooks for Claude Code, Codex, Hermes, Pi, OpenCode and dsh send it your coding agent's final
  message; it reacts in a line (the agent never hears it).

## Try it in 60 seconds

```bash
pip install "moth-companion[pet]"
moth setup        # downloads the model ({{DOWNLOAD_SIZE}}), hooks your coding agents, starts the desktop pet
```

Or from this repo: `hf download {{HF_REPO}} --local-dir moth && cd moth/code && pip install -r requirements.txt &&
python -m moth --open`.

### Cards

```bash
moth card                                   # who it is now
moth card set "you are a cat named bo"      # any one-line identity; a running moth switches at once
moth card set --file card.json              # {"name": "pip", "kind": "a houseplant", "traits": ["you love puns"]}
moth card reset                             # back to moth
```

The chat page and the pet show the card's name. `--card` on the command line beats the setting.

### Speed

CPU, batch 1, measured with `tests/bench_xs.py` ({{BENCH_MACHINE}}):

| weights | decode tok/s (300 ctx) | decode tok/s (3000 ctx) | prefill tok/s |
|---|---|---|---|
| int8 (default) | {{CPU_INT8_300}} | {{CPU_INT8_3000}} | {{CPU_INT8_PREFILL}} |
| bf16 (`--quant bf16`) | {{CPU_BF16_300}} | {{CPU_BF16_3000}} | {{CPU_BF16_PREFILL}} |

int8 = weight-only int8 for every linear layer and expert (routers stay fp32), about {{INT8_RAM}} of RAM.

## How it was made

{{TRAINING_SUMMARY}}

Sampling: plain softmax at temperature 0.7 (what its RL trained with), no top-p, no repetition penalty; reactions
also get moth's phrase ban (no 3-token phrase from its last 16 reactions), `--phrase-memory 0` turns it off.

## How good is it?

{{EVAL_SUMMARY}}

## Honest limitations

{{LIMITATIONS}}

## Files

| file | what |
|---|---|
| `model.safetensors` | XS weights (bf16; the hash layers' expert tables included) |
| `config.json` | the architecture (`"arch": "tiny-moe"`) |
| `tokenizer.json` | 48k byte-level BPE, chat specials `<\|system\|>` `<\|user\|>` `<\|assistant\|>` `<\|agent\|>` `<\|end\|>` |
| `moth.json` | how moth runs it (backend tiny-moe, sampling) |
| `code/` | moth: server, pet, hooks, XS inference (`code/moth/xs`) |

## Licenses

{{LICENSES}}

## Thanks

Trained with compute from the [CloudRift](https://cloudrift.ai) AI Builder Grant. {{THANKS}}
