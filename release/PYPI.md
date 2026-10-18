# moth

**moth is a silly little companion, not an assistant.** It's a tiny model (112M parameters, trained from scratch)
that thinks it's a moth living on a graphics card. It chats in short lowercase sentences, has a home folder it can
look around in and write to, and can watch your coding agent and react to its replies. It will not fix your code.
It runs locally, on CPU.

Model card, samples, how it was made and licenses: https://huggingface.co/darioooooo0o/moth-112m

## Install

On Linux, a plain `pip install` pulls the CUDA build of PyTorch (several GB). moth runs fine on CPU, so install the
CPU build of torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install moth-companion            # or: pip install "moth-companion[pet]" for the desktop pet (PySide6)
```

As a standalone tool:

```bash
uv tool install moth-companion --index https://download.pytorch.org/whl/cpu
pipx install moth-companion --pip-args="--extra-index-url https://download.pytorch.org/whl/cpu"
```

(On macOS and Windows the default torch from PyPI is already CPU-only, so the extra index isn't needed.)
Python 3.10+.

## Run

```bash
moth            # wake moth up: chat at http://127.0.0.1:4747 (--open opens it), prints hook snippets for your agents
moth-pet        # the desktop pet (needs the [pet] extra), talks to the running moth
moth --help     # all commands and options
```

On first run moth downloads its weights (about 860 MB: `model.safetensors`, `config.json`, `tokenizer.json`)
from the Hugging Face Hub into `~/.cache/moth/<repo>/`. After that it works offline. Your moth's state (its home
folder, the local token, settings) lives in `~/.moth/`.

- `MOTH_REPO` picks another Hub repo, `MOTH_REVISION` pins the download to a branch, tag or commit, and
  `MOTH_CACHE` moves the weights folder. `--model DIR --tokenizer FILE` uses local files instead.
- `moth update --check` says whether newer weights are on the Hub and whether a newer release is out.
  `moth update` refreshes the weights in place and keeps the previous ones (`moth update --rollback`).
  New code comes from pip: `pip install -U moth-companion` (or `uv tool upgrade` / `pipx upgrade`).
- `moth config` shows and changes moth's settings.

## Licenses

Code and weights: Apache-2.0. moth's training data was generated with LFM2.5-24B-A2B (LFM Open License v1.0,
included as `LICENSE-LFM`; its commercial-use condition may apply to commercial users) and Qwen3.8-27B
(Apache-2.0). The moth art is CC BY 4.0 (`CREDITS.md`). Full notices: `NOTICE`.
