# Swap the model behind moth

moth's pet, hooks, chat page and `moth tail` all talk to one local server (`moth.serve`). The model behind that
server can be changed with one setting: a later moth, another creator's small persona model, or any small local
chat model. Everything else stays the same: the hooks, the reaction batching across agent sessions, the home folder,
the chat page and the pet.

```bash
moth config model ~/models/moth-2                          # a weights folder (its moth.json says how to run it)
moth config model someone/persona-100m                     # an HF repo id
moth config model "openai:http://127.0.0.1:11434/v1#qwen3:0.6b"   # a local llama.cpp / Ollama / vLLM server
moth config model default                                  # back to moth's own model
```

The setting applies from moth's next start. `--model <same forms>` overrides it for one run (`python -m moth`,
`python -m moth.serve`, `python -m moth.cli`, `python -m moth.pet --model ...`), and `moth setup --model ...` saves it
as part of setup. `moth doctor` shows which backend, model and character are active.

## Backends

| backend | runs | needs | tools | phrase ban | context kept across turns |
|---|---|---|---|---|---|
| `tiny-agent` (default) | moth-112m and anything trained with this repo's pipeline (`tiny_agent/`) | nothing extra | yes | yes | KV cache: each turn feeds only the new text; int8 on the CPU, device graphs on CUDA/XPU |
| `transformers` | any causal LM `transformers` loads (a folder or an HF repo id) with a chat template, CPU or GPU | `pip install "moth-companion[transformers]"` | if the manifest says so | yes | no: each turn re-reads the chat |
| `openai` | any OpenAI-compatible `/v1/chat/completions` server: llama.cpp `llama-server`, Ollama, vLLM, LM Studio | a running server | if the manifest says so | no | no (the server may cache prefixes) |

How moth picks the backend for a model spec:

- **A folder.** If it has a `moth.json`, that file decides. Without one, it's a tiny-agent folder, which is how every
  existing moth folder keeps working unchanged. The exception is a plain transformers save, a `config.json` with a
  `model_type`, which runs on transformers with the general-model defaults below.
- **An HF repo id (`owner/name`).** moth fetches only the repo's `moth.json`, a few hundred bytes, and uses it.
  - tiny-agent repos download their weights into `~/.cache/moth/<owner>/<name>/` on first start, the same way
    moth-112m does.
  - Repos without a `moth.json` load through transformers, which uses its own Hugging Face cache.
  - A `moth.json` from the Hub can't name an openai endpoint. Where moth's text goes is your call, never a
    downloaded file's: use an `openai:` spec or a moth.json of your own for that.
- **`openai:<base_url>#<model>`.** An OpenAI-compatible server. `base_url` can be `http://host:port` or
  `http://host:port/v1`.

## moth.json

Put it next to the weights. Every key is optional, and a missing key takes its default. Unknown keys are ignored (a
newer manifest still loads on an older moth). A key with the wrong type stops moth with a message that names it.

| key | default | what it does |
|---|---|---|
| `backend` | `"tiny-agent"` | `tiny-agent`, `transformers` or `openai` |
| `character` | `"moth"` | the name shown where it is display text (the pet's tray tooltip and notification title, `moth.cli`'s prompt, doctor). Commands and the package stay `moth` |
| `chat_format` | `"chatml"` (tiny-agent), `"template"` (others) | `chatml`: moth's own `<\|im_start\|>` format. `template`: the tokenizer's chat template (transformers) or the server's own (openai) |
| `system` | moth-112m's trained prompt (tiny-agent), a short moth character sheet (others) | the system prompt |
| `agent_card` | `"[the agent said]\n{message}"` | how a coding agent's final message is shown to the model. `{message}` is the cleaned text (markdown stripped, long messages cut to head + tail). Must contain `{message}` |
| `reaction` | none (tiny-agent), a "react in one short silly line…" nudge (others) | added after the agent card for that one reaction only. History keeps the plain card |
| `sampling` | `{"temperature": 0.8, "top_p": 0.95, "rep_penalty": 1.3, "max_tokens": 200}` | defaults for `--temperature`, `--top-p`, `--rep-penalty` and `--max-turn-tokens`. Flags still win |
| `tools` | `true` (tiny-agent), `false` (others) | moth's home tools: `bash` / `read` / `write` in its own little folder, as `<tool_call>` JSON. Off means every turn only talks |
| `lowercase` | `true` | moth only speaks lowercase |
| `context` | 3000 | tokens of chat kept in mind. Older turns drop out first. `--ctx-budget` wins |
| `stop` | `[]` | extra stop strings |
| `model` | the folder itself | transformers: a folder or an HF repo id. openai: the server's model name |
| `base_url` | none | openai: the server, e.g. `http://127.0.0.1:8080/v1` |
| `api_key_env` | `MOTH_OPENAI_API_KEY` | openai: the environment variable holding a key, if the server wants one. It must start with `MOTH_`, so a manifest can't send another key (`GITHUB_TOKEN`...) anywhere. Keys never go in moth.json or moth's config |
| `tokenizer` | `tokenizer.json` in the folder | tiny-agent: the tokenizer file, relative to the folder |
| `repo` | none (moth-112m: its own repo) | the HF repo `moth update` refreshes this model from |
| `version` | none | the model's own version, shown by doctor and the serve log |
| `needs_moth` | none | the oldest moth that can run this model, e.g. `"0.4.0"`: an older moth says how to upgrade instead of loading it, and a pip install's `moth update` won't fetch it |

### moth-112m (today's model, no file needed)

A folder without `moth.json` is exactly this. Spelled out:

```json
{
  "backend": "tiny-agent",
  "character": "moth",
  "chat_format": "chatml",
  "agent_card": "[the agent said]\n{message}",
  "sampling": {"temperature": 0.8, "top_p": 0.95, "rep_penalty": 1.3, "max_tokens": 200},
  "tools": true,
  "repo": "darioooooo0o/moth-112m"
}
```

### A hypothetical moth 2 (same pipeline, new weights)

```json
{
  "backend": "tiny-agent",
  "character": "moth",
  "system": "you are moth. you live on a graphics card. ...the prompt moth 2 was trained with...",
  "sampling": {"temperature": 0.7},
  "repo": "darioooooo0o/moth-2",
  "version": "2.0"
}
```

Upload it with `model.safetensors`, `config.json` and `tokenizer.json`. If it needs code an older moth doesn't
have (a new architecture), set `"needs_moth"` to the first moth version that can run it. A `config.json` with
settings an older moth doesn't know is caught too: that moth says how to upgrade rather than failing to load. Users then run
`moth config model darioooooo0o/moth-2`. The weights download on first start, and `moth update` keeps them current
from that repo.

### A general small chat model through transformers

Any instruct model with a chat template works. It has never been trained as moth, so the manifest gives it the
character: either the default moth sheet (leave `system` out) or your own.

```json
{
  "backend": "transformers",
  "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "character": "moth",
  "sampling": {"temperature": 0.7, "max_tokens": 60},
  "context": 2048
}
```

Without a `moth.json`, `moth config model HuggingFaceTB/SmolLM2-360M-Instruct` does the same with these defaults:
chat template, no tools, the moth character sheet, and the reaction nudge. A 0.3-0.5B model answers in a second or
two on a CPU. To give it the home tools, set `"tools": true` and describe the `<tool_call>` format in `system`.
Only models that can follow that format should get them.

### llama.cpp / Ollama (an OpenAI-compatible server)

```bash
llama-server -m persona-100m-q8_0.gguf --port 8080           # or: ollama serve  (and ollama pull qwen3:0.6b)
moth config model "openai:http://127.0.0.1:8080/v1#persona"
```

Or as a folder that holds only this `moth.json`, to set the character and prompt too:

```json
{
  "backend": "openai",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen3:0.6b",
  "character": "moth",
  "sampling": {"temperature": 0.8, "max_tokens": 80}
}
```

The repetition penalty goes to the server as `repeat_penalty` (llama.cpp) and `repetition_penalty` (vLLM). If a
server rejects unknown fields, moth sends the request again without them. Ollama's OpenAI endpoint ignores them.
`<think>` blocks in a reply are dropped. A server that's down gives moth's "my head went all fuzzy" line, and the
chat goes on.

### Another creator's persona model

- If it ships as transformers weights with a chat template, write a `moth.json` with `"backend": "transformers"`,
  its name as `character` and the system prompt it was trained with as `system`.
- If it ships as a GGUF, serve it with `llama-server` and use the openai backend.
- If it trained on its own prompt format, set `agent_card` to whatever it expects for "someone else said this", and
  `reaction` to null if it already reacts in character.
- A custom architecture that neither can load needs a small new backend in `moth/backends/`: load, `count`,
  `generate`, as in `moth/backends/base.py`.

## Privacy

moth's promise is that nothing leaves your machine, and the tiny-agent and transformers backends keep it: the model
runs inside the moth server. The openai backend sends every agent message moth reacts to, and everything you type to
moth, to the server you name.

- On `localhost`, `127.x` or `::1` that server is on your machine, and moth never goes through a proxy to reach it.
- On any other host, including a computer on your LAN, the text leaves your machine. moth warns at startup,
  `moth config model` says so when you set it, and `moth doctor` shows a privacy line on every run. Doctor never
  contacts a non-local endpoint itself.
- API keys come only from an environment variable (`MOTH_OPENAI_API_KEY`, or the one `api_key_env` names). They are
  never written to moth's files.

## Updating

`moth update` refreshes only models it downloaded from the Hub. That means moth-112m, and tiny-agent models from
another repo, where it uses the manifest's `repo` (model files only, with the same backup and `--rollback`). For
anything else it explains what to do instead:

- transformers models: `hf download <repo>`
- an openai server: update the model where it runs, e.g. `ollama pull`

For those it says so and then updates moth's own install as usual (its code, and moth's own model), so code updates and the pet's daily "new moth out" check keep working.

## What changes for the pet and the page

With a non-default model the server's token-free `GET /api/health` also says `"character"` and `"backend"`. The pet shows the character
name in its tray tooltip and notifications, and older servers keep showing "moth". The phrase memory (no same joke
twice), the reaction throttle, the batching across agent sessions and the one-turn-per-session history are moth's
own engine and work the same on every backend. Only the phrase ban needs token-level sampling, so it is off on the
openai backend.
