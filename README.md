# moth

**moth is a silly little companion, not an assistant.** It's a tiny 112M-parameter model that thinks it's a
silkmoth living on your graphics card. It's a bit dim, gets things sweetly wrong, and will not fix your code. It
keeps you company while your coding agent works: hook it up and it reacts to each agent turn's final message
with a cute misreading.

```
[the agent said] Deployment to staging succeeded. The health check is green and response times are around
120 ms. Want me to promote it to production?
moth: green is a soft word. i'll keep a little lamp on for you
user: should i say yes?
moth: yes. yes is a soft word. i'm glad the agent was trying to be helpful
```

This repo is the code: the app (local server, chat page, terminal chat, desktop pet), the hooks for coding agents,
and the pipeline that made moth (character sheet, teacher-chat generation, filters, fine-tuning, eval). The
weights are not in git.

## Status

- **Weights: coming to Hugging Face** as [`darioooooo0o/moth-112m`](https://huggingface.co/darioooooo0o/moth-112m).
  They're not published yet. Until they are, a clone of this repo can run the tests but not moth itself.
- **pip: not on PyPI yet.** Once it is, `pip install moth-companion` (or `"moth-companion[pet]"` for the desktop
  pet) gives you the `moth` and `moth-pet` commands, and moth downloads its weights from the Hub on first run.

The model card with the full details (example chats, how it was trained, eval numbers, limitations) is
[`release/README.md`](release/README.md).

## What it does

- **Talks.** Short lowercase back-and-forth, and sometimes it remembers what you said a few turns ago.
- **Keeps your agent company.** One way only: the agent never sees moth or what you tell it.
- **Has a home folder** (`~/.moth/home`: `lamps/ diary/ snacks/ friends/ sounds/ saved/`). It looks around with a
  tiny pure-Python shell (`ls`, `cat`, `head`, `wc`, `echo`, plus `write`) that can't leave that folder.
- **Runs locally.** CPU is fine (about a quarter of a second per reply, ~0.8 GB of RAM; int8 weights by default, `--quant none` for exact
  fp32, see the model card's "Speed"). The server listens on 127.0.0.1 only and
  needs a per-install token (`~/.moth/token`). It goes online only to download the weights or an update.

## Running it (once the weights are out)

From a Hugging Face download (`hf download darioooooo0o/moth-112m --local-dir moth-112m`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip if you have torch
pip install -r moth-112m/code/requirements.txt PySide6               # PySide6 only for the pet
cd moth-112m/code && python -m moth.pet                              # the desktop pet; it starts the server
```

`python -m moth` runs just the server and prints hook snippets with your real paths. `python -m moth.cli` chats in
the terminal, `python -m moth.tail` watches moth in a side terminal.

`python -m moth update` (from `code/`; `moth update` for pip installs) updates in place. Stop moth first.
`--check` only shows what would change, and `--rollback` restores the previous version. An interrupted update
changes nothing, and running it again keeps the files that already finished downloading. Files you added are kept,
and so are your edits to files the new version didn't change. Details: [Updating](release/README.md#updating) in the model card.

## Hooks

`python -m moth hook --all` hooks moth into every harness it finds (it shows the plan and asks first, backs up and
merges each config). `python -m moth hooks` shows what's installed, `python -m moth unhook <harness>|--all` removes
moth's entries, `python -m moth doctor` is a health check.

| Harness | How moth hears it | Where moth's reply shows up |
|---|---|---|
| Claude Code | `Stop` hook (`claude_code.py`) | a band above the prompt (`claude_code_mod/`) |
| Codex CLI | `hooks.json` hook or `notify` (`codex.py`) | `moth tail` |
| Hermes Agent | plugin (`hermes/moth/`) | the Hermes Desktop pet's speech bubble |
| Pi | extension (`pi/moth.ts`) | a widget above the editor (opt-in), plus `/moth <text>` |
| OpenCode | plugin (`opencode/moth.js`) | a toast (opt-in) |
| dsh | cordis plugin (`dsh/moth.mjs`) | `moth tail` |

Details and caveats per harness: [`moth/integrations/README.md`](moth/integrations/README.md).

## Pet and skins

`python -m moth.pet` is a small moth flying around your screen, saying moth's lines in a speech bubble. Click it to
talk, right-click for the chat page, mute, sleep and quit. It can wear any [petdex.dev](https://petdex.dev) pet you
have installed (`--skin <id>`, or `python -m moth config pet_skin <id>` to keep it), and
`--export-petdex DIR` writes moth as a Petdex pet. moth ships no third-party pets.

## Make your own character

The personality is in the weights, not the prompt, so a new character means re-running the pipeline:
[`moth/docs/OWN_CHARACTER.md`](moth/docs/OWN_CHARACTER.md). It needs an OpenAI-compatible teacher server and a
CUDA or Intel XPU GPU for the fine-tune.

## Development

Tests need no weights and no GPU; see [`CONTRIBUTING.md`](CONTRIBUTING.md). CI runs them on Linux, macOS and
Windows.

## Licenses

- Code: **Apache-2.0** ([`LICENSE`](LICENSE)). The weights will be Apache-2.0 too.
- The moth art (the pet, its Petdex sprite sheet, the chat page icon) is "202206 Silkmoth adult female icon" by
  DataBase Center for Life Science (DBCLS), doi:10.7875/togopic.2022.231,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:202206_Silkmoth_adult_female_icon.svg). It's not
  under Apache-2.0. The changes are listed in [`moth/assets/CREDITS.md`](moth/assets/CREDITS.md), which also
  covers the chat page's fonts (Fraunces and JetBrains Mono, SIL Open Font License 1.1).
- moth's training data was generated with **LFM2.5-24B-A2B** (Liquid AI, LFM Open License v1.0) and Qwen3.8-27B
  (Apache-2.0). The LFM license text is in [`release/LICENSE-LFM`](release/LICENSE-LFM). Its commercial-use
  condition (a US$10M annual revenue threshold) may apply to commercial users of the moth weights.
- The agent messages used in training and eval come from public datasets (NVIDIA Nemotron-SWE-v1 and Nebius
  SWE-rebench-openhands-trajectories under CC BY 4.0, Lambda hermes-agent-reasoning-traces under Apache-2.0). The
  full list is in [`release/NOTICE`](release/NOTICE).

Made by [@imdariotoo](https://x.com/imdariotoo).
