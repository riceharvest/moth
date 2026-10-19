"""The installers' smoke-test weights: a tiny random moth (4 MB, tests/test_update_e2e.py's tiny_model) plus the real
tokenizer.json, in one folder the app takes as MOTH_MODEL_DIR. Needs torch, safetensors, tokenizers (the build env).

  python packaging/make_fixture.py OUT_DIR [--tokenizer PATH]

Without --tokenizer, tokenizer.json comes from the public base repo at the revision and sha256 that
.github/workflows/test.yml pins (TOKENIZER_REPO / TOKENIZER_REVISION / TOKENIZER_SHA256 in the environment).
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("e2e", os.path.join(ROOT, "tests", "test_update_e2e.py"))
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    weights, cfg = e2e.tiny_model(7)
    with open(os.path.join(a.out, "model.safetensors"), "wb") as f:
        f.write(weights)
    with open(os.path.join(a.out, "config.json"), "wb") as f:
        f.write(cfg)
    tok = a.tokenizer
    if not tok:
        from huggingface_hub import hf_hub_download
        tok = hf_hub_download(os.environ["TOKENIZER_REPO"], "tokenizer.json", revision=os.environ["TOKENIZER_REVISION"])
        digest = hashlib.sha256(open(tok, "rb").read()).hexdigest()
        assert digest == os.environ["TOKENIZER_SHA256"], f"tokenizer.json sha256 {digest}"
    shutil.copyfile(tok, os.path.join(a.out, "tokenizer.json"))
    print("\n".join(f"{f} {os.path.getsize(os.path.join(a.out, f))}" for f in sorted(os.listdir(a.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
