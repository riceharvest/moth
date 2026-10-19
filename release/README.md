---
license: apache-2.0
language:
- en
base_model: darioooooo0o/tiny-agent-112m
base_model_relation: finetune
datasets:
- nvidia/Nemotron-SWE-v1
- nebius/SWE-rebench-openhands-trajectories
- lambda/hermes-agent-reasoning-traces
tags:
- tiny
- companion
- chatbot
- character
- agent-companion
- tool-use
- from-scratch
- intel-arc
- desktop-pet
pipeline_tag: text-generation
---

> [!NOTE]
> **moth is a silly little companion, not an assistant.** It's a tiny model that thinks it's a moth living on a
> graphics card. It's a bit dim, gets things sweetly wrong, and will not fix your code. It will keep you company
> while your coding agent does, though.
>
> By [@imdariotoo](https://x.com/imdariotoo)

# moth

> **A much better talker is on the way.** A new moth model, fully pretrained, then SFT'd and RL'd, is training right
> now. To get it: `python -m moth update` in a Hugging Face download, `pip install -U moth-companion` for pip, or the
> new installer from the [releases page](https://github.com/riceharvest/moth/releases/latest).

Real chats with this model, hand-picked from its eval transcripts (temperature 0.7) and copied as generated.
`[ran …]` marks a tool call. The last chat skips four turns of small talk, as marked. Nothing else is cut.

```
user: this is boring
moth: i get it. i'll keep the lamp warm for you.
user: say something interesting
moth: i once rode on the mouse cursor for a whole trip across the screen.
```

Watching a coding agent:

```
user: hey moth, my agent is deploying the app
moth: deploying the app? i bet deploy is a kind of cheese!
[the agent said] Deployment to staging succeeded. The health check is green and response times are around
120 ms. Want me to promote it to production?
moth: green is a soft word. i'll keep a little lamp on for you
user: should i say yes?
moth: yes. yes is a soft word. i'm glad the agent was trying to be helpful
```

```
[the agent said] I've outlined three options for the caching layer: Redis, an in-memory LRU cache, or HTTP
caching headers. Each has trade-offs (see above). Which would you prefer?
moth: cache. it's like a crumb. i like that.
user: what would you pick moth?
moth: i'd pick the cache. it's the coziest thing.
user: why?
moth: because it's like a warm cup of tea.
```

Remembering something from a few turns back (the whole chat):

```
user: my favourite food is dumplings
moth: dumplings sound nice. i like the way they taste like the warm oven.
user: how's your day going?
moth: fine. the screen's been a little cozy today, and i'm happy to be near the glow.
user: what are you up to?
moth: just watching the cursor zip by, feeling the warm air of your words.
user: do you like rain?
moth: i like the cool drop of the rain on the window.
user: what's your favourite colour?
moth: grey, like a soft wool sock. it's a cosy colour.
user: what's my favourite food?
moth: dumplings. i like the crumbs.
```

A vague "what?" about its own last line:

```
user: what's for dinner in the computer?
moth: i don't eat dinner. i just watch the glow of the screen and wait for crumbs.
user: what glow?
moth: the soft glow from the power light on the card. it's the coziest thing i know.
```

Writing in its home folder:

```
user: write a diary entry about meeting me
  [ran write diary/m meeting_me.txt]
moth: done. it's in diary/m meeting_me.txt.
user: can i see it?
  [ran read diary/m meeting_me.txt]
moth: "met me today. i sat on the mouse cursor for a while and asked for a little crumb. i am a good listener."
```

```
user: what's 2 + 2?
moth: four. i'm not a math moth.
user: what's 17 times 23?
moth: 391. i counted on my wings.
```

And how it often goes wrong:

```
user: what's the capital of france?
moth: paris. i think. it's a big lamp in the sky.
user: how far away is the moon?
moth: paris. i flew at it once and got to the kitchen.
```

```
user: hi moth, i'm rosa
  [ran ls] [ran ls diary/ friends/ lamps/ saved/ snacks/ sounds/]
moth: yes, i'm rosa. nice to meet you.
  … four turns of small talk …
user: what's my name?
moth: i'm moth, not rosa.
```

moth is [tiny-agent-112m](https://huggingface.co/darioooooo0o/tiny-agent-112m) fine-tuned into a character. The
base is a 112M-parameter model I trained from scratch on one Intel Arc Pro B70. moth lives on that same graphics
card, or at least it thinks it does. With its Engram lookup tables it stores 431M parameters, about 860 MB in bf16.
It still runs fine on a CPU, in about 0.8 GB of RAM.

## What it does

- **Talks.** It holds a back-and-forth conversation in short lowercase sentences, and sometimes it remembers
  something you said a few turns ago.
- **Keeps your agent company.** Hook it up to your coding agent and moth sees the agent's final message at the
  end of each turn. Then it reacts with a cute misreading: "fixed the login bug" gets "oh, a bug! i'm a bug too, great great great grandma was the first."
  This is one way only. The agent never sees moth, and it never sees what you say to moth, so you can vent to moth
  about your agent and it stays between you two.
- **Has a home folder.** It has `lamps/ diary/ snacks/ friends/ sounds/ saved/` in `~/.moth/home`. It can look
  around with `ls`/`cat`, read you its diary, and write things down (your name, say) so it can remember them later.
  What it writes stays there across restarts.

## Try it in 60 seconds

```bash
pip install huggingface_hub                                          # for the `hf` command
hf download darioooooo0o/moth-112m --local-dir moth-112m
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip if you already have torch
pip install -r moth-112m/code/requirements.txt PySide6               # tokenizers, safetensors, numpy; PySide6: the pet
cd moth-112m/code && python -m moth setup
```

`python -m moth setup` takes you from the install to a running moth. It checks the packages (and prints the exact pip
command for anything missing) and that the weights downloaded completely. It hooks moth into the coding agents it
finds (it backs up their configs first), offers to start the pet when you log in, and starts it now. It asks before
each step. `--yes` takes the defaults, `--dry-run` only shows the plan, and running it again only offers what's
still missing. After that, `python -m moth.pet` is the one command: a small moth flies around your desktop and says
moth's lines in a speech bubble. It wakes
moth's server by itself (on the CPU; it dozes with "waking up…" while the model loads) and puts it to sleep again
when you quit the pet. Click the moth to talk to it; right-click for the chat page (`http://127.0.0.1:4747`), mute,
sleep and quit. If a moth server is already running, the pet just uses it. Python 3.10+.

Without a desktop (or for your coding agents' hooks): `cd moth-112m/code && python -m moth` runs only the server in
the terminal and prints ready-to-paste hook snippets with this install's real paths filled in (`--open` also opens
the chat page in your browser). Start it before the pet and the pet uses it.

- CPU is fine: a plain reply takes about half a second. One where moth reads through its files takes a
  second or two.
  The server uses about 0.8 GB of RAM: the Engram tables stay in bf16, memory-mapped from `model.safetensors`, and
  the OS reads in only the rows moth uses. `--device cuda` or `--device xpu` work with `python -m moth` (the pet
  passes server flags on: `python -m moth.pet -- --device xpu`). Speed and the `--quant` flag: see below.
- Or with pip: `pip install "moth-companion[pet]"` gives you the `moth` and `moth-pet` commands, and `moth setup`
  downloads these weights from this repo (about 860 MB) and sets up the rest. On Linux, install the CPU build of
  torch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- `python -m moth.cli` chats in the terminal instead.
- moth's tools run in a tiny pure-Python shell that only knows `ls`, `cat`, `head`, `wc` and `echo`, plus a
  `write` tool, and can't leave the home folder. It runs no subprocess and needs no sandbox, so it works the same
  on Linux, macOS and Windows.
- The server only listens on localhost. Everything that reads or drives the chat needs a per-install token
  (`~/.moth/token`, made on first run). The hooks read the token from that file.

### Installers

No Python? Installers for Windows (setup `.exe`), macOS (`.dmg`, Apple silicon) and Linux (AppImage or tarball) are
on the [GitHub releases page](https://github.com/riceharvest/moth/releases/latest). They bundle Python, CPU torch
and the pet, and download these weights on first launch. They aren't code-signed yet, so Windows and macOS warn the
first time; [docs/INSTALL.md](https://github.com/riceharvest/moth/blob/main/docs/INSTALL.md) says how to get past it.

### Speed

On the CPU, moth stores its Linear weights in int8, with one scale per output channel, and runs them through
PyTorch's built-in int8 kernel, so there's nothing to compile. Each new token streams 136 MB of weights instead of
545 MB in fp32. A lean decode step and a top-k sampler cut the per-token overhead too. These numbers are from a
Ryzen 9 5950X (16 cores, AVX2, no AVX-512) at the default 8 threads, measured while other jobs kept the desktop busy
(load average 10-20; the lower figure is the busier desktop):

| `--quant` | reply speed | median chat turn | next to exact fp32 |
|---|---|---|---|
| `int8` (CPU default) | 85-110 tokens/s | 0.19-0.24 s | now and then a word or phrase differs |
| `bf16` | 60-70 tokens/s | 0.3-0.37 s | nearly identical |
| `none` (exact fp32) | 20-28 tokens/s | 0.9-1.1 s | exact |

int8 replies read the same as exact fp32's. Over 36 greedy test turns, 28 came out word for word and the rest
differed by a word or a phrase ("it's my cozy one" vs "it's my cozy lamp"). bf16 stays even closer to fp32 (35 of 36),
so it's the mode to pick if you want replies as close to fp32 as possible at a bit less speed.

- `--quant auto` (the default) means int8 on the CPU and unquantized on a GPU (`--device cuda|xpu` are unchanged).
- `--quant bf16` stores the weights in bf16. `--quant none` is exact fp32 and the slowest. `--fast` is the same as
  `--quant int8`.
- If your PyTorch build has no int8 kernel for this, moth falls back to bf16 and says so.
- `--threads N` sets the CPU threads (default: min(8, CPUs)). More threads help when the cores are free (16 on the
  5950X), but more threads than free cores (a busy machine, a CPU quota) make it slower.

On a GPU (batch 1, bf16, CUDA/XPU graphs; `--compile` adds a one-time warmup):

| GPU | reply speed | with `--compile` | reaction to an agent message (median) |
|---|---|---|---|
| RTX 5090 (`--device cuda`) | 640 tokens/s | 744 tokens/s | 46 ms (41 ms compiled) |
| Intel Arc Pro B70 (`--device xpu`) | 418 tokens/s | 552 tokens/s | 84 ms (69 ms compiled) |

## Agent companion setup

`python -m moth hook --all` (from `code/`) hooks moth into every harness it finds on this machine. It shows the
plan and asks first, backs up and merges each config it edits (other hooks and keys stay), and changes nothing the
second time. `python -m moth hooks` shows what's installed, and `python -m moth unhook <harness>|--all` removes moth's
entries and nothing else. `python -m moth doctor` is a one-screen health check, and `python -m moth autostart on`
starts the pet when you log in. To do it by hand, `python -m moth` prints the exact snippet for each harness.
In short:

| Harness | How moth hears it | Where moth's reply shows up |
|---|---|---|
| Claude Code | `Stop` hook (`claude_code.py`) | a band above the prompt with a field to talk back (`claude_code_mod/`, an early-access mod) |
| Codex CLI | `hooks.json` hook or legacy `notify` (`codex.py`) | `moth tail` |
| Hermes Agent | plugin (`hermes/moth/`) | the Hermes Desktop pet says moth's lines in its speech bubble |
| Pi | extension (`pi/moth.ts`) | a widget above the editor (opt-in, see below), plus `/moth <text>` |
| OpenCode | plugin (`opencode/moth.js`) | a toast (opt-in, see below) |
| dsh (DeepSeek Harness) | cordis plugin (`dsh/moth.mjs`) | `moth tail` |

For any harness, `moth tail` (`python -m moth.tail` from `code/`) watches moth in a side terminal or tmux split,
and lets you talk to it there. `python -m moth.pet` is the desktop pet that says moth's lines in a speech bubble
(optional: `pip install PySide6`) and wakes the server itself when none is running. moth reacts at most once every 2 minutes, unless the agent apologises, hits an
error, or writes a wall of text. It holds its reaction while you're chatting with it.

The Pi widget and the OpenCode toast are off until you run `python -m moth config show_in_harness on` (from `code/`;
`off` turns them off again, from the next turn, no restart). Off, those adapters still tell moth about each turn.
The Claude Code band and the Hermes Desktop pet are separate opt-ins: they show only when you load or enable them.

Several agent sessions at once are fine, in any mix of harnesses. moth waits a moment for more finished turns,
reacts to the most interesting one, and keeps the sessions apart, so it doesn't mix up two projects.

## Skins and personality

- **Skins.** The desktop pet can wear any [petdex.dev](https://petdex.dev) pet you have installed
  (`~/.petdex/pets/`, `~/.hermes/pets/`, or a folder with `pet.json` and a sprite sheet):
  `cd moth-112m/code && python -m moth.pet --skin <id>` (`moth-pet --skin <id>` for pip installs) for one run,
  `python -m moth config pet_skin <id>` to keep it, or right-click (or the tray icon) → skin. A pet that won't load
  falls back to moth, and the bubble says why. moth ships no third-party pets: they keep their own licenses, and you
  install them yourself.
- **Update bubble.** Once a day the pet checks for a new moth and says so in a bubble. It never installs anything
  itself (`python -m moth config update_check off` turns the check off).
- **"Your agent is done".** The moth hops as each reaction comes up, and right-click → "what did i miss" lists the
  last 10 (harness · project · how long ago). More pet settings, with `python -m moth config KEY VALUE` (most are
  in the right-click menu too): `hide_in_fullscreen on|off` (default on: hide while a game or video is fullscreen;
  native Wayland can't tell, so there it never hides), `chirp on|off` (default off: a soft sound with the hop),
  `muted_sources codex,pi|none` (no bubble or hop for those harnesses), `pet_screen <name>|any` (stay on one
  screen), `quiet_hours 23:00-08:00|off` (no bubbles or sound, the moth dozes).
- **moth as a Petdex pet.** `python -m moth.pet --export-petdex DIR` writes moth as a Petdex pet (`pet.json` and a
  sprite sheet) for other apps that show Petdex pets.
- **Personality is in the weights, not the prompt.** The system prompt is short and fixed, and every training chat
  used it. At 112M parameters, editing the prompt to describe another character mostly won't work. To make your
  own character, re-run the pipeline that made moth:
  [`code/moth/docs/OWN_CHARACTER.md`](code/moth/docs/OWN_CHARACTER.md).

Details, caveats and what has been tested per harness:
[`code/moth/integrations/README.md`](code/moth/integrations/README.md).

## Swap the model

One setting puts another model behind the pet and hooks. Options are a later moth, another small persona model, any
small chat model with a chat template, or a local llama.cpp / Ollama / vLLM server:

```bash
python -m moth config model someone/persona-100m                       # an HF repo id or a weights folder
python -m moth config model "openai:http://127.0.0.1:11434/v1#qwen3:0.6b"  # an OpenAI-compatible local server
python -m moth config model default                                    # back to moth
```

- **`moth.json`.** A file next to the weights sets the backend, character name, system prompt, how agent messages
  are shown, and sampling. A folder without one runs exactly as moth-112m always has.
- **transformers models.** These need `pip install "moth-companion[transformers]"`.
- **Privacy.** An `openai:` endpoint that isn't localhost gets everything moth sees, and moth and `moth doctor`
  warn about it.
- **More.** [`code/moth/docs/MODEL_BACKENDS.md`](code/moth/docs/MODEL_BACKENDS.md) has the details.

## Updating

From the download's `code/` folder (`moth update` for pip installs):

```bash
python -m moth update --check      # what would change; exit 0 = up to date, 10 = update available, 1 = error
python -m moth update              # asks, then updates this folder in place (--yes: don't ask)
python -m moth update --rollback   # back to the version before the last update (works offline)
```

- **Stop moth first.** The update refuses while a moth server from this folder runs (`--check` is fine). The pet
  only checks once a day and says so in a bubble; it never installs anything.
- **An interrupted update changes nothing.** New files download into `.moth-update/` first and are checked against
  the Hub's hashes. Only then are they swapped in, which takes a moment. If the network drops or you press Ctrl-C,
  the old version keeps working. Run the update again: files that finished downloading are kept, and a file that
  was cut off halfway downloads again. A swap that can't finish (a file open in another program on Windows, a
  read-only file) is undone on the spot. If the computer dies during the swap itself, the next update or rollback
  undoes the half-done swap first.
- **Your files.** Files you added are never touched. If you edited a shipped file, your edit stays when the new
  version didn't change that file, and gets replaced when it did. The update names every file it replaces, and your
  copy goes into the backup.
- **Backup.** Replaced and removed files go to `.moth-backup/<old version>/`. Only the last update's backup is
  kept, and a rollback uses it up. Hook paths stay valid: the folder never moves. `~/.moth/` (home, token, settings)
  is never touched.
- `--revision <tag or commit>` (or `MOTH_REVISION`) pins a version, older ones included. `--repo` / `MOTH_REPO` picks
  another repo. With `HF_HUB_OFFLINE=1` or `MOTH_OFFLINE=1` set, it says so and doesn't go online.
- pip installs: `moth update` refreshes only the model files in `~/.cache/moth/<repo>/`, with the same backup and
  rollback. `--check` also tells you when newer code is out (`pip install -U moth-companion`). A model that needs
  newer code than you have isn't fetched: moth says to upgrade first. The installers' app works the same way and
  points you at the releases page.
- A change to only this card or the docs is not an update: `--check` stays at "up to date" and the pet stays quiet.

## Privacy

- **What moth sees: two things only.**
  - The final message of each agent turn, posted by the hooks, with a session id and the project folder's name.
  - What you type to moth.

  It never gets your prompts to the agent, the agent's tool calls, or your files.
- **Everything stays on your machine.** The server, the hooks, the pet, `moth tail` and the chat page use only
  127.0.0.1 (the page's fonts are bundled). moth goes online only to download the weights or an update from the
  Hugging Face Hub. The one exception is a model you swap in yourself (`moth config model openai:...`) on another
  host. That server gets everything moth sees, and moth warns at startup and in `moth doctor`.
- **Where it's stored: `~/.moth/`.** That holds the token, settings, a server log (status and errors, no chat text),
  and moth's home folder: its files, its diary, and anything it wrote down. The conversation itself lives only in
  RAM, and so does moth's memory of its own last 16 reactions (kept so it doesn't tell the same joke twice, also
  across "new chat"). Stopping moth wipes both.
- **The agent never receives anything back.** The hooks only post, print nothing and exit 0. moth's lines show only
  in displays you turn on, and none of them feed the model.
- **Wiping it.** Stop moth, then `python -m moth forget` deletes its logs. `--all` also resets its home folder, and
  `--all --token` also deletes the token and settings. Run on a moth that's still running, `forget` deletes no files,
  but it makes that moth forget the chat and its last reactions right away.

## How it was made

| | Round 2 | Round 3 (this release) |
|---|---|---|
| Teacher | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (INT4 by RedHatAI) | [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8) (FP8) |
| Goal | a cute character with a home folder | a real back-and-forth chatbot, plus the agent companion |
| Chats written | 6,640 | 92,020 |
| Kept after filtering | pooled with round 3 → | 67,414 of the 98,660 together (68%): 70% talk, 21% agent, 9% home. About 5% of the kept chats are round-2 ones |

In each chat the teacher played moth, with real tools in a sampled home folder, and also played the human, from
a few dozen personas and scenarios. For agent chats, the human's coding agent posts a final message and moth
reacts to it. At generation time about 3 in 4 of those messages were real ones from public agent traces (see
Licenses), and the teacher wrote the rest.

**Filters.** `build_sft.py` walks each chat up to moth's first bad reply and keeps the part before it, if it still
has at least two moth replies. "Bad" covers:
- assistant-speak, lists, and replies over 40 words;
- **identity crutch**: "i'm just a little moth" is fine once per chat, or when you ask what moth is, but not as
  the answer to everything;
- **made-up follow-ups**: the simulated human asking "what lamp?" when moth never mentioned a lamp, which taught
  moth to explain things it never said;
- **motif caps**: each of moth's favourite images (fan, glow, warmth, hum, the card, light) can show up at most
  once per chat, and at most two replies per chat can use any of them, unless the human brought it up. Round 2's moth talked about the fan in
  about a quarter of all replies;
- agent reactions that *explain* the agent's message instead of cutely misreading it;
- files or folders moth never actually saw.

**Training.** Short talk chats are also stitched into long sessions of up to 4k tokens, so moth stays on track
deep into a conversation. Training is SFT on moth's tokens only, at 4k context, for 1 epoch with a
warmup-stable-decay schedule at LR 1e-3 (Muon optimizer). It starts from the tiny-agent-112m base checkpoint. The
training set is the 67,414 kept chats plus 1,000 stitched sessions. A run takes about 7 minutes on an RTX 5090. This
release is the end of the run, which was also its best validation-loss checkpoint. Kept out of training: 350
held-out chats and the 127 chats built around the 40 eval agent messages.

**Judging.** Qwen3.8-27B scored scripted multi-turn conversations as a judge. On top of that, a Hermes Agent
interviewer, driven by the same model, chatted with moth live, adapted to what it said, and compared versions
blind.

## How good is it?

Every version gets the same 110 scripted multi-turn probes (small talk, its home folder, recall, "what X?"
follow-ups, 40 real agent messages, ...), 3 samples each with paired seeds, so both models get the same home and the same
dice. Qwen3.8-27B judges each chat on a strict 1-5 scale (judge prompt v2 for both models; 3 = a typical decent reply).
v2 is the previous model, from round 2.

| Judge (mean 1-5, 330 chats) | v2 | this release |
|---|---|---|
| cute (the main score) | 3.25 | **3.39** |
| character | 3.38 | **3.83** |
| coherence | 2.66 | **3.02** |
| grounding (only talks about files it really saw) | 4.49 | **4.82** |
| back-and-forth | 2.66 | **2.80** |

| Counted, no judge (1,113 replies) | v2 | this release |
|---|---|---|
| identity crutch ("i'm just a little moth" as the answer to everything) | 23.6% | **8.2%** |
| replies mentioning the fan | 23.0% | **6.1%** |
| agent reactions that explain or advise instead of misreading | 17.0% | **10.5%** |
| remembers a fact from 10 turns back (n=9) | 0 of 9 | **5 of 9** |
| answers "what X?" about the X it just said (n=21) | 33% | **67%** |
| "remember X", then asked for X right away (n=9) | **56%** | 33% |

**Head to head.** The judge saw both models' chats side by side, twice with the positions swapped, and a win needed
both verdicts to agree (anything else counts as a tie):

- this release vs v2: 97 wins, 62 losses, 171 ties (55% with ties as half, sign test p = 0.007). Most of the wins
  are agent reactions (56-11). It loses on feelings (1-7), odd inputs (0-5) and things it can't do (0-4).
- this release vs a sibling run from the same round: 80-69, 181 ties (52%, p = 0.41), so about the same model.
  I've been using that sibling day to day, and I love it.
- A Hermes Agent interviewer (same judge model) chatted live with that sibling and with v2, blind: 3-3 over 6
  sessions. It's a qualitative check, and its written critiques were more useful than its scores.

Caveats: there's one judge, Qwen3.8-27B, which also wrote the round-2 training data. In about half the pairs it
picked the same position both times, and those count as ties. The recall checks are small (n=9) and
keyword-based. One of the 5-turn "passes" is a false positive, shown as the last sample at the top.

## Honest limitations

- It's 112M parameters, pretrained on not much English text, so it knows very little about the world. It's meant
  to be dim, but it's dim for real.
- It sometimes answers a different question than the one you asked.
- It forgets things from more than a few turns back, unless it wrote them down in its home folder.
- Its agent reactions are sometimes off: too literal, or just unrelated.
- When you're sad or happy it turns into a generic "i'm here for you". The judge preferred v2 there.
- English only.

## Changelog

Update an existing download in place with `python -m moth update` (from `code/`; `--check` shows what changed;
see [Updating](#updating)).

**0.3.1**
- An edit of only the model card or docs no longer counts as an update (no "there's a new me!" bubble).
- Ready for the next model: an older moth that can't run a newer model says how to upgrade (pip, installer, or
  `moth update`) instead of crashing, and pip installs never fetch a model their code can't load. A model's
  `moth.json` can say `"needs_moth": "<version>"`.
- pip installs: the startup hints use the `moth` command; the wheel leaves out the training scripts.

**0.3.0** (round 3)
- Conversation-first data: 70% of the 67k kept chats (from 98,660 teacher chats) are plain back-and-forth talk,
  filtered for identity crutches, made-up follow-ups and overused motifs.
- Agent companion for six harnesses (Claude Code, Codex CLI, Hermes Agent, Pi, OpenCode, dsh).
- Desktop pet (`python -m moth.pet`) and `moth tail` for watching and talking to moth.
- moth's tools run in a pure-Python shell inside its home folder: no subprocess, the same on every OS.
- Several agent sessions at once: batched reactions and no mix-ups between sessions.
- `python -m moth update` and `python -m moth --version`.
- `python -m moth hook` / `unhook` / `hooks` (installs the integrations with backups and merging), `doctor`,
  `autostart on|off|status` and `forget`.
- `python -m moth setup`: one guided command from the install to a running moth.

## Files

| Path | What |
|---|---|
| `model.safetensors`, `config.json` | the model (custom architecture, see `code/tiny_agent/`) |
| `tokenizer.json` | tokenizer (same as tiny-agent-112m) |
| `code/moth/` | the app: `__main__.py` (start here), `serve.py` + `static/` (browser chat), `cli.py` (terminal chat), `tail.py` (side-terminal display), `pet.py` (desktop pet, needs PySide6; `launch.py` wakes the server for it), `shell.py` (moth's little shell) |
| `code/moth/backends/`, `code/moth/docs/MODEL_BACKENDS.md` | the swappable model backends (tiny-agent, transformers, OpenAI-compatible) and the `moth.json` manifest |
| `code/moth/setup.py` | `moth setup`, the guided first run |
| `code/moth/integrations/` | hooks and plugins for the six harnesses, their README, and an offline test against a stub moth |
| `code/moth/*.py` (rest) | character sheet, data generation, filtering, SFT and eval scripts |
| `code/moth/file_pool.json` | the teacher-written files moth's home is sampled from |
| `code/moth/docs/OWN_CHARACTER.md` | how to make your own character with this pipeline |
| `code/moth/assets/` | the moth art (original SVG, CC BY 4.0), the optional chirp sound (CC0) and their `CREDITS.md` |
| `code/data/eval_agent_msgs.json` | 40 real agent messages used as eval prompts (unmodified, with sources) |
| `code/tiny_agent/` | model, generation and tool code from tiny-agent |
| `code/requirements.txt` | Python dependencies |
| `LICENSE`, `LICENSE-LFM`, `NOTICE` | licenses and third-party notices |

## Licenses

Code and weights: **Apache-2.0** (`LICENSE`).

The moth art (the pet, its Petdex sprite sheet, the chat page icon) is "202206 Silkmoth adult female icon" by
DataBase Center for Life Science (DBCLS), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:202206_Silkmoth_adult_female_icon.svg). It's not under
Apache-2.0. The changes are listed in `code/moth/assets/CREDITS.md`.

moth's training data was generated with **LFM2.5-24B-A2B** (Liquid AI, LFM Open License v1.0) and **Qwen3.8-27B**
(Apache-2.0). The LFM license text is in `LICENSE-LFM`. Its commercial-use condition (a US$10M annual revenue
threshold) may apply to commercial users of moth, who should review `LICENSE-LFM`.

The agent messages come from these datasets:
- [nvidia/Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) by NVIDIA,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
  by Nebius, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) by
  Lambda, Apache-2.0.

I modified the messages before use: markdown stripped, long ones truncated to their first 110 and last 45 words.
They were then used as prompts, and the teacher wrote moth's reactions to them. The exception is the 40 eval
messages in `code/data/eval_agent_msgs.json`, which are included unmodified. `NOTICE` has the full list.

## Thanks

- [CloudRift](https://cloudrift.ai) for the RTX 5090s, on research credits. All the teacher data, training and
  evals for this round ran there.
- The [Qwen](https://huggingface.co/Qwen) team for Qwen3.8-27B (teacher and judge), and RedHatAI for the INT4 quant.
- [Liquid AI](https://huggingface.co/LiquidAI) for LFM2.5-24B-A2B, the round-3 teacher.
- NVIDIA, Nebius and Lambda for publishing their agent traces.
- DBCLS for the silkmoth icon that became moth's body (CC BY 4.0).
- [Kenney](https://kenney.nl) for the chirp sound (CC0).
- Inspired by "bugger" by [@gnostic_snakes](https://x.com/gnostic_snakes), another tiny bug that lives on a
  graphics card.

Made by [@imdariotoo](https://x.com/imdariotoo).
