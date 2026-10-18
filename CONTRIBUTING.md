# Development / tests

The unit tests need no model weights and no GPU: the model is faked, servers are stubs on 127.0.0.1, and the Qt
tests run offscreen. They run on Linux, macOS and Windows with Python 3.11+ (CI: `.github/workflows/test.yml`,
GitHub-hosted runners, 3.11 and 3.12).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch
pip install -r .github/ci-requirements.txt            # tokenizers, safetensors, numpy, huggingface_hub, PySide6, pytest
```

Node 18+ (22 in CI) runs the JS adapter tests (Pi, OpenCode, dsh, the Hermes desktop pet). Without `node` they skip.

`tests/test_engine.py` uses the real tokenizer (no weights). It isn't in the code tree; fetch just that file:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('darioooooo0o/tiny-agent-112m', 'tokenizer.json', local_dir='data/tiny-agent-112m')"
```

Without it the engine tests skip (the log-buffer tests still run).

## Run

One suite at a time (each takes seconds, test_integrations about a minute):

```bash
python -m pytest -q -rs tests/test_engine.py        # chat engine, scripted fake model
python -m pytest -q -rs tests/test_update.py        # `python -m moth update` against a fake Hub
python -m pytest -q -rs tests/test_config.py        # `python -m moth config`, browser opt-in
python -m pytest -q -rs tests/test_pet.py           # desktop pet: flight, bubbles, poller, launcher, Qt offscreen
python -m pytest -q -rs tests/test_shell.py         # moth's pure-Python shell
python -m pytest -q -rs tests/test_eval_probes.py   # eval probes, metrics, pairwise (fake moth)
python -m pytest -q -rs moth/integrations/test_integrations.py   # harness adapters + serve's HTTP handler
```

Every file also runs on its own without pytest (`python tests/test_shell.py`).

Some parts skip where their inputs are missing, and `-rs` says why:

- `test_shell.py`: the replay against the recorded training bash (needs the local `results/r3/b.jsonl`), and the
  differential test against real bash (needs a working `bwrap`, Linux only).
- `test_pet.py`: the Qt tests without PySide6 (or its native libraries), the signal tests off POSIX, the
  parent-death test off Linux.

Lint as CI does: `python -m compileall -q moth tiny_agent tests`, and `node --check` on each JS adapter copied to a
`.mjs` file (they are plain ESM JavaScript, `pi/moth.ts` included).
