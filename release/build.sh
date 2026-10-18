#!/bin/bash
# Assemble the Hugging Face repo folder for moth:  bash release/build.sh runs/<final>/model
#   -> release/hf/ : model.safetensors, config.json, tokenizer.json, README.md, LICENSE, code/
# Dry run (prints the file list and checks it, touches nothing):  DRY_RUN=1 bash release/build.sh [model dir]
# MOTH_RELEASE_OUT=/some/dir builds there instead (e.g. to test the copy without touching release/hf/).
# Both modes first check that every moth.* / tiny_agent.* import in the shipped code is shipped too.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=${DRY_RUN:-0}
MODEL=${1:-}
if [ "$DRY" != 1 ] && [ -z "$MODEL" ]; then
  echo "usage: bash release/build.sh <final model dir (model.safetensors + config.json)>" >&2; exit 2
fi
OUT=${MOTH_RELEASE_OUT:-release/hf}   # override to try a build elsewhere
TA_FILES="model generate chat checkpoint tools optim"
MOTH_FILES="__init__ __main__ persona agent home serve shell cli tail eval probes metrics gen_dialogs build_sft sft add_hermes_pool _tools pet config weights launch hooks doctor autostart forget pairwise"

# what goes where: "<source>\t<path in the repo>"
files() {
  if [ -n "$MODEL" ]; then
    printf '%s\t%s\n' "$MODEL/model.safetensors" model.safetensors "$MODEL/config.json" config.json
  fi
  printf '%s\t%s\n' data/tiny-agent-112m/tokenizer.json tokenizer.json LICENSE LICENSE release/README.md README.md
  # third-party notices: the LFM Open License (round-3 teacher) and the source/attribution list
  printf '%s\t%s\n' release/LICENSE-LFM LICENSE-LFM release/NOTICE NOTICE
  # tiny_agent ships from this project's copy (moth-specific changes, e.g. the low-mem CPU loader)
  for f in $TA_FILES; do printf '%s\t%s\n' "tiny_agent/$f.py" "code/tiny_agent/$f.py"; done
  for f in $MOTH_FILES; do printf '%s\t%s\n' "moth/$f.py" "code/moth/$f.py"; done
  printf '%s\t%s\n' moth/update.py code/moth/update.py moth/VERSION code/moth/VERSION   # python -m moth update / --version
  printf '%s\t%s\n' moth/file_pool.json code/moth/file_pool.json
  # the chat page and everything next to it (e.g. bundled fonts), so the page needs no network
  find moth/static -type f -not -path '*/__pycache__/*' -not -name '.DS_Store' | LC_ALL=C sort |
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "code/$f"; done
  # eval's held-out agent messages (public pool data): probes.py reads <code>/data/eval_agent_msgs.json
  printf '%s\t%s\n' data/eval_agent_msgs.json code/data/eval_agent_msgs.json
  # adapters + their offline test (stub servers only, no local data); never caches
  find moth/integrations -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -not -name '.DS_Store' | LC_ALL=C sort |
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "code/$f"; done
  # the pet / page art (third-party, CC BY 4.0): the original SVG pet.py renders, CREDITS.md (see NOTICE);
  # the page's fonts' OFL texts (assets/fonts/; the fonts themselves are embedded in static/index.html);
  # not assets/candidates/ (the art that was considered and not picked)
  find moth/assets -type f -not -path 'moth/assets/candidates/*' -not -path '*/__pycache__/*' -not -name '.DS_Store' | LC_ALL=C sort |
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "code/$f"; done
  # guides linked from the card (e.g. docs/OWN_CHARACTER.md: make your own character with this pipeline)
  find moth/docs -type f -not -name '.DS_Store' | LC_ALL=C sort |
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "code/$f"; done
}

LIST=$(files)
missing=0
while IFS=$'\t' read -r src dst; do
  [ -f "$src" ] || { echo "missing: $src" >&2; missing=1; }
done <<< "$LIST"
[ "$missing" = 0 ] || exit 1

# import closure: a shipped module that imports an unshipped moth.* / tiny_agent.* module fails at runtime
python3 - "$LIST" <<'EOF'
import ast, sys
pairs = [line.split("\t") for line in sys.argv[1].splitlines()]
shipped = {dst for _, dst in pairs}
bad = []
for src, dst in pairs:
    if not dst.endswith(".py"):
        continue
    for node in ast.walk(ast.parse(open(src).read(), src)):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        for n in names:
            parts = n.split(".")
            if parts[0] in ("moth", "tiny_agent") and len(parts) == 2:
                mod = f"code/{parts[0]}/{parts[1]}.py"
                pkg = f"code/{parts[0]}/{parts[1]}/__init__.py"
                attr = isinstance(node, ast.ImportFrom) and n != node.module and node.module in ("moth", "tiny_agent")
                if mod not in shipped and pkg not in shipped and (attr or n == getattr(node, "module", n)):
                    bad.append(f"{dst}: imports {n}")
if bad:
    print("not shipped:\n  " + "\n  ".join(sorted(set(bad))), file=sys.stderr)
    sys.exit(1)
print(f"import check: ok ({sum(d.endswith('.py') for d in shipped)} python files)")
EOF

if [ "$DRY" = 1 ]; then
  echo "dry run: would rebuild $OUT/ with"
  while IFS=$'\t' read -r src dst; do printf '  %-48s <- %s\n' "$dst" "$src"; done <<< "$LIST"
  echo "  code/requirements.txt                            (generated)"
  [ -n "$MODEL" ] || echo "  (+ model.safetensors and config.json from the model dir)"
  exit 0
fi

rm -rf "$OUT"
while IFS=$'\t' read -r src dst; do
  mkdir -p "$OUT/$(dirname "$dst")"
  cp "$src" "$OUT/$dst"
done <<< "$LIST"
cat > "$OUT/code/requirements.txt" <<'EOF'
torch>=2.5
tokenizers
safetensors
numpy
EOF
du -sh "$OUT"; find "$OUT" -type f | sort
