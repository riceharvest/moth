# moth

> **A much better talker is on the way.** A new moth model, fully pretrained, then SFT'd and RL'd, is training right
> now. It'll arrive as a normal `moth update`, no reinstall.

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

## Get it

- **Installers** (no Python needed): Windows `.exe`, macOS `.dmg` (Apple silicon) and Linux AppImage or tarball on
  the [releases page](https://github.com/riceharvest/moth/releases/latest). See [docs/INSTALL.md](docs/INSTALL.md).
- **pip**: `pip install "moth-companion[pet]"` (on Linux, first
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`), then `moth setup`. It downloads the
  weights from the Hub and sets the rest up.
- **Hugging Face**: the weights and a runnable copy of this code are
  [`darioooooo0o/moth-112m`](https://huggingface.co/darioooooo0o/moth-112m); see "Running it" below.

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

## Running it

From a Hugging Face download (`hf download darioooooo0o/moth-112m --local-dir moth-112m`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip if you have torch
pip install -r moth-112m/code/requirements.txt PySide6               # PySide6 only for the pet
cd moth-112m/code && python -m moth setup                            # then this: hooks, autostart, starts the pet
```

`python -m moth setup` (`moth setup` for pip installs) is the first thing to run after installing. It checks the
packages and the weights (a pip install downloads them here, about 860 MB, if you say yes), hooks moth into the
coding agents it finds, offers autostart at login, and starts the pet. It asks before each step, `--yes` takes the
defaults, `--dry-run` only shows the plan, and running it again only offers what's still missing. Later,
`python -m moth.pet` starts the pet by itself (it wakes the server).

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

When an agent finishes, the moth hops as its reaction comes up; right-click → "what did i miss" lists the last 10.
Settings (`python -m moth config KEY VALUE`; the pet picks them up within 30 s, most are in its menu too):
`hide_in_fullscreen` (on: hide while a game or video is fullscreen; not on native Wayland), `chirp` (off: a soft
sound with the hop), `muted_sources` (e.g. `codex,pi`: no bubble for those harnesses), `pet_screen` (`any`, or the
screen to stay on) and `quiet_hours` (e.g. `23:00-08:00`: no bubbles, the moth dozes).

## Swap the model

`python -m moth config model <folder | HF repo id | openai:URL#model | default>` puts another model behind the
pet and hooks: a later moth, another small persona model, any small chat model through `transformers`
(`pip install "moth-companion[transformers]"`), or a local llama.cpp / Ollama / vLLM server. A `moth.json` next to
the weights sets the backend, character name, prompt, agent-card format and sampling. A folder without one runs
exactly as before. `moth doctor` shows what's active. Details and examples:
[`moth/docs/MODEL_BACKENDS.md`](moth/docs/MODEL_BACKENDS.md).

## Make your own character

The personality is in the weights, not the prompt, so a new character means re-running the pipeline:
[`moth/docs/OWN_CHARACTER.md`](moth/docs/OWN_CHARACTER.md). It needs an OpenAI-compatible teacher server and a
CUDA or Intel XPU GPU for the fine-tune. Ship the result with a `moth.json` (its name, prompt and repo) and anyone
can run it with `moth config model <your repo>`.

## Development

Tests need no weights and no GPU; see [`CONTRIBUTING.md`](CONTRIBUTING.md). CI runs them on Linux, macOS and
Windows.

## Installers

Installers that need no Python are on the [releases page](https://github.com/riceharvest/moth/releases/latest): a
Windows setup `.exe` (per user, no admin), a macOS `.dmg` (Apple silicon) and a Linux AppImage or tarball. Each
bundles everything except the weights, which download on first launch. CI builds and tests them
(`.github/workflows/installers.yml`). How they work, the unsigned-app warnings and how to get past them:
[docs/INSTALL.md](docs/INSTALL.md).

## Licenses

- Code: **Apache-2.0** ([`LICENSE`](LICENSE)). The weights are Apache-2.0 too.
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

## Thanks

The training data, training and evals ran on RTX 5090s from [CloudRift](https://cloudrift.ai) research credits.
Inspired by [@gnostic_snakes](https://x.com/gnostic_snakes)' "bugger".

Made by [@imdariotoo](https://x.com/imdariotoo).
