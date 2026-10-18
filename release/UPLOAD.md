# Uploading moth to Hugging Face

Nothing here has been run. The upload waits for the user's go.

Repo: `darioooooo0o/moth-112m` (model). That id is hard-coded in `moth/weights.py` (`REPO_ID`), which
`moth update` and the pip install both read, so keep it or change it in both places.

## Checklist

- [x] License decision: Apache-2.0 for code and weights, plus the LFM Open License disclosure (`LICENSE-LFM`,
      `NOTICE`, "Licenses" in the card). The art is CC BY 4.0 with credits (`code/moth/assets/CREDITS.md`, NOTICE).
- [ ] `release/hf` rebuilt from the final tree after the other agents' work landed:
      `bash release/build.sh runs/final2/ckpt_100`. The import check must say `ok`, and `build.sh` must ship every
      module `__main__.py` imports (`hooks`, `doctor`, `autostart`, ...) plus `moth/docs/`.
- [ ] Privacy line holds: the chat page loads no remote fonts or scripts (fonts are inlined as base64 data URIs).
      `grep -nE '(href|src)="https?://|@import|googleapis|gstatic' release/hf/code/moth/static/index.html` finds
      nothing (plain `https://` links inside HTML comments are credits, not loads), and the OFL texts are in
      `release/hf/code/moth/assets/fonts/`.
- [ ] Weights are final2: `cmp release/hf/model.safetensors runs/final2/ckpt_100/model.safetensors`.
- [ ] No private paths or secrets: `grep -rIl -e "$HOME" -e /mnt/ -e hf_ -e sk- -e 'BEGIN .*PRIVATE' release/hf` finds
      nothing that matters, and `find release/hf -name __pycache__ -o -name '*.pyc' -o -path '*candidates*'` is empty.
- [ ] Card: `grep -n '{{' release/hf/README.md` is empty. Commands the card documents work in the built copy
      (`--skin`, `config pet_skin`, `config update_check`).
- [ ] Fresh-download test on the rebuilt folder: copy it to a scratch folder outside the repo, then run `--version`, `--help`,
      `doctor`, one `/api/chat` and one `/api/agent` from a clean venv with a scratch `HOME`.
- [ ] Update rehearsal: `python -m pytest -q tests/test_update_e2e.py` passes (fake Hub, real huggingface_hub
      downloads, real servers). To rehearse with the real weights, serve two local builds with
      `python tests/fake_hub.py --commit old=<build A> --commit new=<build B> --main old` and point `HF_ENDPOINT` at
      it (see the docstring). Never point a test at the real repo.
- [ ] Order: Hugging Face first, PyPI second. The pip package downloads its weights from this repo, so it must
      exist (and be public) before `moth-companion` goes on PyPI. Until then the card says "once moth is on PyPI".

## Commands

```bash
cd ~/moth
hf auth whoami                                    # logged in as darioooooo0o, with a write token
# creates the repo if it doesn't exist (public, since --private is not given), then uploads the folder
hf upload darioooooo0o/moth-112m release/hf . --repo-type model --commit-message "moth 0.3.0 (round 3)"
```

To check it privately first, add `--private` on the first upload (it applies only when the repo is created), look
at the card on the Hub, then make it public in the repo settings.

Afterwards:

```bash
hf download darioooooo0o/moth-112m --local-dir /tmp/moth-hubtest    # the real fresh download
cd /tmp/moth-hubtest/code && python -m moth update --check           # should say up to date (exit 0)
```

A later update is the same `hf upload` command: bump `moth/VERSION`, add a `## Changelog` entry to the card (that
section is what `moth update --check` prints), rebuild `release/hf`, upload. `moth update` compares file hashes
against the newest Hub revision, so users get exactly the changed files with `python -m moth update`. The pet's
daily check sees the new version. Files dropped from the repo are removed from users' downloads, but only files the
previous upload shipped (`.moth-version` keeps the list), never their own files. Keep old revisions on the Hub (no
force-push or squash): users roll back with `--rollback` and pin with `--revision`, and the update uses the installed
revision to tell users' edits from upstream changes.

## Optional follow-up: the training chats as a dataset

The kept teacher chats could go up as a dataset repo, e.g. `darioooooo0o/moth-chats`:
`hf upload darioooooo0o/moth-chats <folder> . --repo-type dataset`. Before that:

- the round-3 chats were written by LFM2.5-24B-A2B, so the dataset carries the LFM Open License terms (ship
  `LICENSE-LFM` with it and say so on the card); the round-2 chats came from Qwen3.8-27B (Apache-2.0);
- the agent messages in them are modified excerpts of NVIDIA / Nebius (CC BY 4.0) and Lambda (Apache-2.0) traces,
  so the dataset card needs the same attribution as `NOTICE`;
- leave out the 350 held-out chats and the eval agent messages, or mark them as the test split.
