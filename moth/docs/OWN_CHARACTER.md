# Make your own character

moth's personality lives in the weights, not in the prompt. To get a different character you re-run the pipeline that
made moth: write a character sheet, have a big teacher model write chats in that voice, filter them, and fine-tune
tiny-agent-112m on what's left. Everything below ships in `code/`. Commands run from `code/`.
(In a git checkout of the GitHub repo, run them from the repo root instead, and use
`data/tiny-agent-112m/tokenizer.json` (see CONTRIBUTING.md) where this page says `../tokenizer.json`.)

What it cost us for moth's round 3: about 120-130 chats per minute per RTX 5090 with LFM2.5-24B-A2B as the teacher. We
used about 4 rented 5090s for a few hours each: about $23 of GPU time for the whole round, failed boxes and
mistakes included.
Fine-tuning took about 7 minutes on one 5090. Expect to iterate. Most of our time went into the filters, not the GPUs.

## 1. Write the character

- **`moth/persona.py`**
  - `TEACHER_BIBLE`: the long character sheet the teacher plays your character from. Who it is, how it talks, what it
    knows, how it uses its tools. Its "EXAMPLES OF THE VOICE" block matters most: the teacher copies that register.
  - `SYSTEM_STUDENT`: the short, fixed system prompt the small model is trained and served with. Keep it tiny and
    never change it after training. Every training chat uses it, and `moth/serve.py` serves with it.
  - `USER_PERSONAS`, `SCENARIOS`, `TOPICS`, `USER_SIM_PROMPT`: who the simulated human is and what they talk about.
    This is where the variety comes from.
  - `FOLDERS`, `FOLDER_HINTS`, `FILE_POOL_PROMPT`: the home folder's rooms and the prompt that fills them.
- **`moth/home.py`**: `SEED` holds the hand-written files per folder. Each training chat samples a different home from
  these plus the file pool, so the model has to look instead of memorising.
- **`moth/agent.py`**: the agent-companion rules (`MOTH_AGENT_RULES`, the agent scenarios and prompts). Rewrite them for
  your character, or turn agent chats off with `--agent-share 0` below.

Then write a new file pool with the teacher (step 2's server):

```bash
python -m moth.gen_dialogs --pool --n-per-folder 250 --out out    # -> out/file_pool.json
```

## 2. Generate teacher chats

`moth/gen_dialogs.py` talks to any OpenAI-compatible server. We ran LFM2.5-24B-A2B-FP8 on vLLM (v0.31) like this:

```bash
vllm serve LiquidAI/LFM2.5-24B-A2B-FP8 --served-model-name teacher --max-model-len 16384 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser lfm2 --reasoning-parser qwen3
```

Then, with `pip install openai`:

```bash
python -m moth.gen_dialogs --base-url http://127.0.0.1:8000/v1 --pool-file out/file_pool.json --think \
  --home-share 0.15 --dialogs 20000 --start 0 --concurrency 192 --out out    # -> out/dialogs.jsonl (resumable)
```

- `--think` is for teachers that always reason first (LFM2.5). It caps and strips the reasoning (`--think-budget`,
  default 256 tokens). Leave it off for a non-thinking teacher.
- The share flags set the mix: `--agent-share` (default 0.25), `--home-share` (0.15), `--recall-share` (0.15, a fact
  early and a question about it 5+ turns later) and `--long-share` (0.15, 16-24 user turns).
- `--start` is the first seed. Use a new range for each run or box so chats don't repeat.
- The teacher's tool calls run in a real temporary home folder, sandboxed with `bwrap`. On a box without user
  namespaces, set `MOTH_NO_BWRAP=1` to use plain bash in the workspace instead (only on a throwaway machine).
- Agent chats: moth's real agent messages came from `moth/agent_pool.json`, built from public traces
  (`moth/add_hermes_pool.py` adds the Hermes ones). That pool isn't in this download. Without it, the teacher writes
  every agent message itself.

## 3. Filter and build the training set

```bash
python -m moth.build_sft out/dialogs.jsonl --out data/sft_mine --max-len 4096 --sessions 1000 \
  --tokenizer ../tokenizer.json
```

`build_sft.py` walks each chat to the first bad reply and keeps the part before it (at least two replies). Read its
printout: the drop reasons, the mix and the rates of crutch lines and motifs. That's where you'll spend your time.
Its filters are written for moth, so adapt them to your character: `IDENTITY` (the "i'm just a little moth" crutch),
`MOTIFS` and `MAX_MOTIF_REPLIES` (overused images), `BANNED` (assistant-speak), `MAX_WORDS` (reply length), and the
agent-reaction checks (`TECHY`, `EXPLAINS`). `--sessions` chains short talk chats into long ones, so the model
stays on track deep into a conversation. `--exclude-seeds` / `--exclude-agent-msgs` keep held-out chats and eval
prompts out of training, and `--cap-file name.jsonl:0.08` limits one input file's share.

## 4. Fine-tune

Download the base checkpoint, then train. This needs a CUDA or Intel XPU GPU (FlexAttention + `torch.compile`);
it won't train on CPU.

```bash
hf download darioooooo0o/tiny-agent-112m --include 'base/*' --include tokenizer.json --local-dir data/tiny-agent-112m
python -m moth.sft --init data/tiny-agent-112m/base --data data/sft_mine --out runs/mine \
  --lr 1e-3 --epochs 1 --T 4096 --micro_B 4 --accum 2 --device cuda
```

These are moth's settings: 1 epoch, warmup-stable-decay (`--warmup 20 --decay-frac 0.25`, the defaults), LR 1e-3,
4k context. It trains only on the character's tokens. It writes `runs/mine/model` (best validation loss) and
`ckpt_60`, `ckpt_80`, `ckpt_100` (each with `model.safetensors` + `config.json`). Chat with one:

```bash
python -m moth.cli --model runs/mine/ckpt_100       # or: python -m moth.serve --model runs/mine/ckpt_100
```

## 5. Evaluate

The judge talks to the same kind of OpenAI-compatible server as step 2 (`pip install openai`). `metrics.py` needs no
judge.

```bash
python -m moth.eval --model runs/mine/ckpt_100 --tokenizer ../tokenizer.json --samples 3 --device cuda \
  --temperature 0.7 --judge-url http://127.0.0.1:8000/v1 --out evals/mine     # report.md, results.json, transcripts.md
python -m moth.metrics evals/old/results.json evals/mine/results.json          # no judge: crutch lines, motifs, recall
python -m moth.pairwise evals/mine/results.json evals/old/results.json --judge-url http://127.0.0.1:8000/v1 \
  --out evals/mine_vs_old                                                       # blind A/B, positions swapped
```

The probes (`moth/probes.py`), the judge prompt (`JUDGE` in `moth/eval.py`) and the metric regexes
(`moth/metrics.py`) are written for moth too. Rewrite them for your character. Read the transcripts as well as the
scores: the judge misses things you'll spot in a minute.

## 6. Ship it with a moth.json

Put a `moth.json` next to `model.safetensors`, `config.json` and `tokenizer.json`. It holds:

- the character's name, which the pet shows;
- the short system prompt you trained with (your `SYSTEM_STUDENT`);
- the agent-card format, if you changed `[the agent said]`;
- your sampling defaults;
- the HF repo id, if you upload it there.

```json
{"backend": "tiny-agent", "character": "pebble", "system": "you are pebble. ...", "repo": "you/pebble-112m"}
```

Then `moth config model runs/mine/ckpt_100` (or `moth config model you/pebble-112m` once it's on the Hub) runs your
character behind the same pet and hooks, and `moth update` refreshes it from your repo. Every key is described in
[`MODEL_BACKENDS.md`](MODEL_BACKENDS.md). The same file also lets moth run a model that wasn't made with this
pipeline (transformers or a llama.cpp / Ollama server).

## Licenses

Your training data carries the teacher's terms. Chats written by LFM2.5 fall under the LFM Open License (see
`LICENSE-LFM`, including its commercial-use revenue condition). Chats written by an Apache-2.0 teacher such as Qwen
don't add conditions. Check the license of whatever teacher you use before you publish a model trained on its output.

## Idea: several characters in one model

The small model can't follow a character description in its prompt, but it can learn a name. You could give each
character its own short `SYSTEM_STUDENT` that names it ("you are moth..." / "you are pebble..."), generate chats for
each from its own bible, and train on all of them together. Switching characters is then a matter of switching
the system prompt to one of the trained names. We haven't tried this, so it's untested.
