# moth harness adapters

Each adapter sends the coding agent's **final reply of a turn** to the running moth server, one way:
`POST $MOTH_URL/api/agent` with `{"text": "<raw markdown>", "source": "claude-code", "session": "<id>", "label":
"moth-repo"}`. `source` is the harness name; `session` is a stable id of the agent session and `label` a short name for
it, the basename of its working directory (else the first 8 characters of the session id). Cards show them as
"claude · moth-repo said…" in moth's page and `moth tail`. `session` and `label` are optional: an older adapter
without them counts as one session per harness. `MOTH_URL` defaults to `http://127.0.0.1:4747`.

Every adapter follows the same rules:
- It uses a 1 s timeout and never throws or prints. The JS and Hermes adapters never wait inside the agent's own code
  path; the Claude Code and Codex hooks run in the foreground, so a one-shot run can't kill them before they post
  (the wait is the hook's start-up plus one local POST that moth only queues, 1 s at most).
- It posts nothing when moth is down. If moth is slow, it hangs up after 1 s, but moth keeps the text: `serve.py`
  reads the request body first and the page polls `/api/log`.
- It posts nothing for subagent turns, aborted turns, or turns with no assistant text. It never re-sends an older reply.
- It needs no dependencies beyond Python's stdlib or the harness's own JS runtime (`fetch`).
- It sends moth's per-install token (`X-Moth-Token`, read from `~/.moth/token` on every post; moth makes the file on
  its first run, `MOTH_TOKEN_FILE` overrides). Python adapters bypass any HTTP proxy set in the environment.

moth doesn't react to every turn: at most one reaction per 2 minutes (`--react-every`), except when the agent
apologises, reports a failure or an error, or writes a very long message. While you're chatting with moth (the last
60 s, `--quiet-after`) it holds its reaction until you stop. Every turn still shows up as a card; only ~25 words of
each one stay in moth's memory after it reacted, so agent turns don't push your conversation out.

Several sessions at once (any mix of harnesses):
- **Batching.** When moth may react, it first waits `--batch-window` seconds (default 4) for other sessions to finish
  too. Then it reacts to the most interesting message: a failure, apology or error first, then a question to you,
  then a long message, then a routine "done" (the newest wins a tie). The others show as skipped cards. Batched cards
  appear when the window closes, the chosen one last, so moth's reply sits right under it. `--batch-window 0` turns
  this off, and `--react-every 0 --quiet-after 0` (the judge tooling) still reacts to every message at once, with no
  window.
- **No mix-ups.** moth's history keeps one agent turn per session: when it reacts to session B, the older agent
  turns of B, and all but the newest of every other session, are dropped together with moth's replies to them (agent
  turns carry a session marker in the history that the model never sees). moth's input stays exactly what it
  trained on (`[the agent said]` + the message); session names never go into it.
- **Seeing them.** The page shows "2 agents active · 1 skipped" (sessions that posted in the last 30 min) and, with
  two or more sessions, a dropdown to show one session's cards (moth's replies always show). `moth tail --session
  <label>` does the same in a terminal. When two active sessions share a folder name, moth adds the end of the
  session id: "moth (3f2a)".
- Synthetic batch turns ("three tasks finished, one says sorry") were left out on purpose: moth only ever trained on
  real agent messages, and a counted summary would be out of distribution. It couldn't be tested here without
  running the model.

| Harness | `session` | `label` from |
|---|---|---|
| Claude Code | Stop input `session_id` | Stop input `cwd` |
| Codex CLI | Stop `session_id` / notify `thread-id` | `cwd` (both) |
| Hermes Agent | `post_llm_call` `session_id` | `agent.runtime_cwd`: gateway session cwd, else `TERMINAL_CWD`, else launch dir |
| Pi | `ctx.sessionManager.getSessionId()` | `ctx.cwd` |
| OpenCode | the event's `sessionID` | `Session.directory` (from `session.created/updated`), else the plugin's `directory` |
| dsh | `session.id` | `process.cwd()` (dsh is started in the project) |

moth has to be awake for the adapters to reach it. The one-command way is the desktop pet,
`cd moth-112m/code && python -m moth.pet`: it starts the server itself and stops it when you quit the pet. For the
snippets, or without a desktop, run `cd moth-112m/code && python -m moth` (just the server, in the terminal; the pet
uses it if it's already running). It prints the snippets below with the real absolute paths
of your copy filled in. The chat page is served at `http://127.0.0.1:4747/` but only opened in your browser with
`python -m moth --open` (or `--browser`; the old `--no-browser` is accepted and does nothing). Below, paths are
relative to the Hugging Face download (`hf download darioooooo0o/moth-112m --local-dir moth-112m`); hooks need
absolute paths, so put your own folder in front of `moth-112m/`.

| Harness | File | Mechanism | Turn-end event |
|---|---|---|---|
| Claude Code | `claude_code.py` | command hook | `Stop` |
| Codex CLI | `codex.py` | hooks.json command hook, or legacy `notify` | `Stop` / `agent-turn-complete` |
| Hermes Agent | `hermes/moth/` | plugin | `post_llm_call` |
| Pi | `pi/moth.ts` | extension | `agent_end` + `agent_settled` |
| OpenCode | `opencode/moth.js` | plugin | `session.status` idle (+ deprecated `session.idle`) |
| dsh | `dsh/moth.mjs` | cordis plugin | `session/event` → `turn/end` |

Test them all offline against a stub moth: `python3 moth/integrations/test_integrations.py` (about 50 s, 127.0.0.1 only).

**Or let moth install them** (from `moth-112m/code`, or anywhere after `pip install`, as `moth ...`):
- `python -m moth hooks` lists the harnesses found on this machine and whether moth is hooked into each. It also
  flags copies that are outdated, broken or point to another moth folder.
- `python -m moth hook claude-code` (or `codex`, `hermes`, `pi`, `opencode`, `dsh`, or `--all` for every harness
  found) does what the sections below describe. It shows the plan and asks first (`--yes` skips the question,
  `--dry-run` only shows the plan), then prints what changed.
  - Every config file it edits is backed up first as `<file>.bak-moth-<timestamp>`. JSON, YAML and TOML configs are
    merged, so other hooks and keys stay. Installing twice changes nothing.
  - Copied adapters (Pi, OpenCode, Hermes) that differ from this folder's copy are replaced, for example after an
    update. The old Hermes folder is kept in `~/.moth/backups/`, outside the plugins folder.
  - Codex: the `hooks.json` hook by default, or `--legacy-notify`. moth never installs both. Approve the hook once in
    `codex` (`/hooks`).
  - `--project DIR`: Claude Code's `DIR/.claude/settings.json`. `--profile NAME`: the dsh profile.
- `python -m moth unhook <harness>|--all` removes moth's entries only. It deletes a copied adapter only if it is
  moth's.

## Privacy: what moth sees, and where it goes

- **What moth sees: two things only.**
  - The final message of each agent turn, posted by the adapters above. Each post carries a session id and the
    project folder's name as its label.
  - What you type to moth: in the chat page, the pet, `moth tail`, Pi's `/moth` or the Claude Code band.

  It never gets your prompts to the agent, the agent's tool calls, or your files. Its own tools reach only its home
  folder.
- **Everything stays on your machine.** The server listens on 127.0.0.1 only, unless you start it with another
  `--host`. The adapters post to `http://127.0.0.1:4747`, unless you set `MOTH_URL`.
  - moth's code goes online for two things only, both from the Hugging Face Hub: downloading the weights (first run
    of a pip install) and `moth update`.
  - The browser chat page embeds its two fonts, so opening it doesn't fetch anything from outside either.
  - The training and eval scripts in `code/moth/` talk to whatever LLM server you point them at. Running moth never
    uses them.
- **Where it's stored: `~/.moth/`.** That holds the token, `config.json`, pidfiles while moth runs, `backups/` from
  `moth hook`, and moth's home folder `home/`: its files, its diary, and anything it wrote down from your chats.
  - `serve.log` holds the server's output when the pet starts it: status lines and errors, no chat text.
  - The conversation itself lives only in RAM. Stopping moth wipes it, and so does the page's "new chat" button.
  - In the dev checkout, the home is `live/home/` and the logs are `live/*.log`.
- **The agent never receives anything back.** The adapters only post. They ignore the answer, print nothing and
  exit 0, so nothing reaches the agent's context.
  - moth's lines show only in displays you turn on: the pet, `moth tail`, the Claude Code band, and the Pi widget or
    OpenCode toast with `show_in_harness on`. None of them feed the model.
  - What you say to moth goes to moth only.
- **Wiping it.** Stop moth first, then:
  - `moth forget` deletes its logs.
  - `moth forget --all` also resets its home folder, but only a folder moth made itself.
  - `moth forget --all --token` also deletes the token and settings.

  `moth unhook --all` removes the hooks.

## Seeing and talking to moth

The adapters above only *tell* moth about each turn. They hang up after 1 s and never see its reply. The pieces
below are the other half: they **show** what moth says back, and they let you talk to it. Each one polls
`GET /api/says` (or the full `/api/log`) for new `say` events and **never posts to `/api/agent`**, so moth never
hears a turn twice. Install
the adapter for your harness as well.

**Each in-harness display shows moth's reaction to its own session only.** With several agents running, moth's
newest line is often its reply to another one (seen live: Claude Code's band showed moth's reply to Hermes). So the
Claude Code band, the Pi widget and the OpenCode toast ask `/api/says?source=<harness>&session=<id>` for the
session they belong to, and show nothing rather than another session's line. The Hermes Desktop pet asks for
`source=hermes` (every Hermes session in that app: the renderer can't map its sessions to the agent's ids). The
desktop pet, `moth tail` and the chat page still show everything. Codex and dsh have no in-harness display.

moth needs a moment to think. A display that waits after a turn (Pi, OpenCode, when turned on) waits up to about
4 s. A slower line is not lost: it shows on the next poll, after the next turn, or right away in `moth tail` and the
pet.

**Venting stays private.** Whatever you say to moth, and what moth says back, never enters the agent's context.
That is why there is no `/moth` in Claude Code or Hermes. In Claude Code a command's output row is read by the model.
In the Hermes gateway, a plugin command sent mid-turn is handed to the running agent. Pi's `/moth` is kept, because
Pi runs extension commands outside the conversation (details per harness below).

**The server only talks to this machine, and to whoever has the token.**
- It answers only requests whose `Host` is `127.0.0.1:<port>` or `localhost:<port>`, so a website that points its
  own domain at 127.0.0.1 (DNS rebinding) gets a 403.
- `POST /api/chat`, `/api/agent` and `/api/reset`, and the full log (`GET /api/log`, every agent reply and everything
  you typed) need the token in `X-Moth-Token`. The page moth serves has it built in; the adapters and `moth tail`
  read `~/.moth/token`.
- `GET /api/says?since=n` is moth's own lines and nothing else, without a token. That is all a display needs, and
  what the Hermes Desktop pet and the Pi / OpenCode / Claude Code displays use. `&source=<harness>` and/or
  `&session=<id>` keep only moth's reactions to that harness / session (moth tags each reaction in its log with the
  turn it answers; chat replies have no tag, so they drop out). The answer is the same `{"say"}` list either way,
  with no ids or labels in it, and `next` still counts every event.
- CORS: local pages (`http(s)://localhost` or `127.0.0.1`, any port) may read answers. Desktop shells
  (`app://…`, `file://`, `null`: Hermes Desktop packaged loads `file://…/index.html`, which sends `Origin: null`)
  may read `/api/says` only. Any other origin gets no CORS header, and a POST from a foreign origin, `null`
  included, is refused outright.
- POSTs must be `Content-Type: application/json`, at most 64 KB, a JSON object; anything else gets a 4xx, not a crash.
  Every answer carries `X-Frame-Options: DENY`.
- Every poll answer has an `epoch` that changes when moth restarts or the chat is reset. Pass it back
  (`?since=n&epoch=e`) and a changed epoch gets you the new log from its start.

### In-harness display: off by default (`show_in_harness`)

The Pi widget and the OpenCode toast come inside adapters you install anyway, so they are **off** until you turn
them on. Off, those adapters still post every turn (the desktop pet and `moth tail` react as usual); they just don't
poll moth or draw anything in the harness.

```bash
cd moth-112m/code && python -m moth config show_in_harness on     # or off
cd moth-112m/code && python -m moth config                        # print the settings and where they come from
```

- Stored as `"show_in_harness": true|false` in `~/.moth/config.json` (`MOTH_CONFIG_FILE` overrides the path). The
  CLI writes it atomically and keeps any other keys.
- The adapters read the file at **every turn**, so a change applies from the next turn without restarting Pi or
  OpenCode. Turning it off also takes down a Pi widget that is on screen.
- `MOTH_SHOW_IN_HARNESS=1` / `0` (also on/off, true/false, yes/no) in the harness's environment beats the file. The
  environment is fixed when the harness starts.
- **Pi `/moth <text>`** is always registered, because Pi registers commands when it loads the extension and so can't
  follow a live toggle. Typing it is an explicit request, so it always answers. With the setting on the answer goes in
  the widget; with it off it comes as a one-off notice, and no widget is left behind.
- **Not governed by it**: the Claude Code band mod (loaded only with `--plugin-dir` or `CLAUDE_CODE_PLUGIN_DIRS`) and
  the Hermes Desktop pet half (starts disabled; you enable it in the desktop's Plugins page). Loading or enabling them
  is their opt-in, and unloading or disabling them turns them off. Honouring a default-off setting would hide them
  from someone who just loaded them on purpose. The Hermes renderer also has no file or environment access to read
  it. `moth tail` and the desktop pet are separate programs you start yourself (the pet starts the server too).

### `moth tail`: works with every harness (Codex included)

Run it in a tmux split or a terminal next to the harness:

```bash
cd moth-112m/code && python -m moth.tail            # moth's lines + a "you>" prompt to chat with it
cd moth-112m/code && python -m moth.tail --watch    # status pane only, no prompt
tmux split-window -h -l 45 'cd moth-112m/code && python -m moth.tail'    # e.g. beside codex / claude
```

- It prints moth's new lines (`moth>` in lamp yellow, wrapped to the pane), a dim `agent said: ...` one-liner for
  each turn moth heard, and a dim `$ cmd` line when moth uses a tool. It shows the last 3 events on connect
  (`--history N`).
- Lines you type go to `/api/chat`, and the reply arrives through the same poll. A moth line that arrives while you
  type is printed above the prompt, and your half-typed text is kept.
- When moth is down it prints "moth is asleep" once and retries with backoff (up to 10 s). It follows moth across
  restarts and `/api/reset`.
- Stdlib only. It doesn't import torch, so any `python3` works. `--url` or `MOTH_URL` changes the address.

### Desktop pet: a moth flying around your screen (`python -m moth.pet`, any harness)

```bash
pip install PySide6                              # optional: only the pet needs it, moth itself doesn't
cd moth-112m/code && python -m moth.pet          # --size 64, --no-wander (sits in a corner), --url, --token-file
python -m moth.pet -- --react-every 0            # flags after -- go to the server the pet starts
python -m moth.pet --export-petdex ./moth-pet    # the same moth as a Petdex / Codex pet (pet.json + spritesheet.png)
python -m moth.pet --skin <pet-id>               # wear an installed Petdex pet instead (or --skin ./some-pet-folder)
python -m moth config pet_skin <pet-id>          # ... every time (default moth); update_check on|off for the bubble
```

- **One command.** At start the pet asks `GET /api/health` on its URL. If a moth answers, it uses that one and never
  stops it. If nothing listens, it starts `python -m moth.serve` itself (same Python, CPU, this download's weights),
  dozes with "waking up…" while the model loads, and stops that server again when the pet quits (menu quit,
  Ctrl-C, SIGTERM/SIGHUP, the window closed; on Linux also if the pet is killed). The server runs in its own process
  group and logs to `~/.moth/serve.log`. If something that isn't moth holds the port, or less than ~3 GB of memory
  is free, it starts nothing and says why in the bubble.
- Flags: `--no-server` (never start one, just wait for moth like before), `--model DIR` (the weights for a server
  the pet starts), `--server-arg=--flag` (repeatable; `--server-arg=--react-every --server-arg=0`) or anything after
  `--` for that server. `python -m moth` / `python -m moth.serve` are still the way to run moth headless or in a
  terminal.

- A small moth, drawn in the page's style, flies around on top of your windows. It drifts, is drawn to the mouse
  cursor, lands on screen edges to rest, and dozes after 3 idle minutes. When a new line arrives it flutters and says
  the line in a speech bubble (wrapped at 32 characters, shown for 6-10 s, queued). When `~/.moth/token` is
  readable, a reaction gets a small `claude · repo` caption, taken from `/api/log`. Without the token the pet uses
  `/api/says`.
- Click the moth to type to it (`/api/chat`, needs the token); the reply comes back in the bubble. Drag it to move
  it. Right-click for: open chat page, mute bubbles, sleep, show moth in harnesses, skin, check for updates, quit.
  It never takes focus, except for that input box. Where the desktop has a system tray, a tray icon has the same
  menu (no tray, e.g. on some Wayland setups: no icon, the right-click menu is still there).
- **Skins.** `--skin ID|FOLDER`, `python -m moth config pet_skin ID` or right-click → skin: the pet can look like any
  Petdex pet you have installed (`npx petdex install <pet>`). Ids are looked up in `~/.petdex/pets/<id>/`, then
  `$CODEX_HOME/pets/<id>/` and `~/.hermes/pets/<id>/`; the menu lists them all. The pet's sheet must be the Petdex
  1536x1872 grid. Flying uses running-right / running-left, resting idle, dozing waiting, excited jumping or waving,
  waiting for a reply review; the bubble sits on the pet's visible top. A missing or broken pet means moth, and the
  bubble says why. moth ships no third-party pets; they keep their own licences.
- **Update bubble.** Once a day (and on "check for updates") the pet runs `python -m moth update --check` in the
  background and, if there's a new version, says so: "run `python -m moth update`", or `pip install -U
  moth-companion` for a pip install. It never updates anything itself. Off: `python -m moth config update_check
  off`; `HF_HUB_OFFLINE=1` or `MOTH_OFFLINE=1` also skips it.
- When moth is down it dozes with "zzz… (moth server asleep)" and retries with backoff (up to 30 s). If the server
  the pet started dies, the bubble points at its log instead.
- Wayland doesn't let an app place its own window. On a Wayland session the pet asks Qt for XWayland
  (`QT_QPA_PLATFORM=xcb;wayland`, only if you haven't set it); without XWayland it stays where the compositor puts
  it (drag still works). Under XWayland the cursor position only updates over X windows, so the pull to the
  cursor fades when the cursor seems to stop. It lands on screen edges, not on other windows' edges.
- `--export-petdex DIR` writes a 1536x1872 sheet (8x9 cells of 192x208: idle, running-right/left, waving, jumping,
  failed, waiting, running, review). Copy DIR to `$CODEX_HOME/pets/moth` (Codex, pick it in `/pets`), to
  `~/.hermes/pets/moth` (`hermes pets select moth`), or to `~/.petdex/pets/moth` (the Petdex desktop app). Codex picks pets by folder name
  (`custom:moth`).
- The pet only reads moth and posts what you type; it never posts to `/api/agent`.

### Claude Code: band above the prompt, with a field to talk to moth (mod)

`claude_code_mod/` is a mod (function hooks, early-access API). It draws moth's newest reaction to this session in
the band above the prompt (`/api/says?source=claude-code&session=<$.session.id()>`, the Stop hook's `session_id`,
read on every poll so `/clear` follows), polling every 2 s, with a text field under it. The reply to what you type
there comes back with the send. Focus the band with ctrl+x tab or a click, type, and press
Enter. Esc returns to the prompt. Keep the `Stop` hook above as well, since it is what tells moth about each turn.
Try it for one session:

```bash
claude --plugin-dir moth-112m/code/moth/integrations/claude_code_mod
```

To load it in every session, add the folder to `CLAUDE_CODE_PLUGIN_DIRS` in the `env` block of
`~/.claude/settings.json` (`"CLAUDE_CODE_PLUGIN_DIRS": "/abs/path/to/moth-112m/code/moth/integrations/claude_code_mod"`).
- `MOTH_URL` changes the address. It is read when the session starts.
- **Kept out of Claude's context.** The field's Enter goes to the mod's own `onSubmit`, which POSTs to moth
  directly. The type definitions say an Input's submit starts "No model turn unless it asks one" (`InputProps.onSubmit`
  in the 2.1.302 `claude-code.d.ts`), and the band is drawn UI, not a transcript row. A slash command was rejected
  because its output is "the row the model also reads" (`CommandRunResult.text`), and a `prompt.submit` `{ drop }`
  comes too late for a plugin command, which has already run by then.
- Mobile has no `Input`, so there the band shows moth's line only.
- It reads moth's lines from `/api/says`; a send reads the token from `~/.moth/token` (`$.fs.read`).
- `claude plugin validate moth-112m/code/moth/integrations/claude_code_mod` passes on 2.1.302.

### Hermes: desktop pet (chat in `moth tail`)

- **No `/moth`.** A `register_command` reply is not added to the agent history when Hermes is idle (CLI
  `_run_plugin_slash_command` only prints it; the gateway's idle path returns before a turn). But in the messaging
  gateway, a plugin command sent while a turn runs never reaches the plugin. `resolve_command` knows only built-ins,
  so `_hm_busy_slash_or_photo` lets it fall through to `_hm_busy_steer` / `_hm_busy_interrupt`, which hand
  `/moth …` to the running agent as text (`gateway/run_inbound.py`). The hooks that run earlier don't fix this:
  `pre_gateway_dispatch` runs before auth, and `post_gateway_admission` runs after the busy lane. So chat with moth
  in `moth tail`.
- **`desktop/plugin.js`**, the Hermes Desktop half. It shows moth's lines in the pet's speech bubble
  (`ctx.pet.say`): one line at a time under a stable id, so each new line replaces the last; lines over 120
  characters are cut into up to 3 pieces shown one after another, each up for 4–30 s. It polls every 2.5 s with
  `ctx.setInterval`, for moth's reactions to Hermes turns only (`source=hermes`). It can't narrow that to one Hermes
  session: the desktop's gateway events carry the TUI session id, while the agent posts its session key
  (`tui_gateway` `_make_agent(sid, key, session_id=key)`).

The install is the same `cp -r` as above. Hermes Desktop copies `plugins/moth/desktop/` into
`<hermes home>/desktop-plugins/moth/`, and such a half starts **disabled**. Enable "moth" in the desktop's Plugins
page, and turn the pet on in Settings. The desktop half reads `http://127.0.0.1:4747` (the renderer has no
environment, so edit `MOTH` in `plugin.js` to change it). Hermes's own desktop lint
(`hermes_cli/plugin_validate_desktop.py`) finds nothing in it.

### Pi: widget above the editor + `/moth`

`pi/moth.ts` now also:
- with `show_in_harness` on (see above), polls the log for up to 4 s after each posted turn and puts moth's
  reaction to this session in a widget above the editor (`ctx.ui.setWidget("moth", ...)`, interactive sessions
  only; `source=pi&session=<getSessionId()>`, nothing without a session id);
- registers `/moth <text>`, which shows the reply in the same widget (or as a notice where there is no widget, or
  when `show_in_harness` is off).
  **Kept out of the agent's context.** `AgentSession.prompt()` runs an extension command's handler and returns
  before any message is made (`dist/core/agent-session.js` `_tryExecuteExtensionCommand`), and interactive mode
  sends input typed mid-stream and during compaction through `prompt()` too (`dist/modes/interactive/interactive-mode.js`).
  The handler only calls `ctx.ui`. The typed line is kept only in the editor's local up-arrow history.

The install is unchanged. The poll timers are unref'd, so `pi -p` doesn't wait on them.

### OpenCode: toast

With `show_in_harness` on (see above), `opencode/moth.js` also polls the log for up to 4 s after posting and shows
moth's reaction to that session as a TUI toast (`client.tui.showToast`, title "moth", 8 s;
`source=opencode&session=<sessionID>`, with a cursor per session, so two sessions in one OpenCode each get theirs). The install is unchanged. There is no `/moth` here: use `moth tail` to
chat.

### dsh: not built (options)

- **A client slot in our cordis plugin.** `dsh/moth.mjs` could poll `/api/says` after `turn/end` and hand moth's
  line to whatever display surface dsh exposes to plugins, the same way as Pi's widget.
- **`@linxin666/dsh-pet`.** If that pet plugin is installed, its announce call could carry moth's line.

Neither was checked against dsh sources (dsh isn't installed here). Until then, use `moth tail` next to dsh.

---

## Claude Code

Add this to `~/.claude/settings.json`, or to a project's `.claude/settings.json`, merging with any `hooks` already there:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "python3 /abs/path/to/moth-112m/code/moth/integrations/claude_code.py", "timeout": 3 } ] }
    ]
  }
}
```

- The Stop input carries `last_assistant_message`, "the text content of Claude's final response". The docs advise
  using it instead of `transcript_path`, which "isn't guaranteed to include the final message at Stop time".
  The transcript is only a fallback for older versions that don't send the key, and even then only the
  current turn is read.
- Subagents fire `SubagentStop`, not `Stop`, so they are skipped.
- **No `"async": true`.** A background hook works in the interactive TUI, but a one-shot `claude -p` kills it when
  it exits, before it posts (seen live on 2.1.302). In the foreground the hook posts first. It is quick: Python's
  start-up plus one POST to 127.0.0.1, which moth only queues, and it gives up after 1 s (`timeout: 3` is Claude
  Code's hard stop). `moth hook claude-code` rewrites an older `async: true` entry, and `moth hooks` flags it as
  outdated.
- Source: <https://code.claude.com/docs/en/hooks> (Stop input, SubagentStop input, command hook fields). The field
  is also present in the local binary, 2.1.302.

## Codex CLI

**Install one of the two options below, not both**, or every reply will be posted twice.

**A. Stop hook (preferred).** Create or merge `~/.codex/hooks.json`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command", "command": "python3 /abs/path/to/moth-112m/code/moth/integrations/codex.py", "timeout": 3 } ] }
    ]
  }
}
```

Then start the interactive `codex` once and approve the hook in the startup hook review, or later with `/hooks`.
Untrusted or modified user hooks are silently skipped, including in `codex exec`. Approval stores a
`trusted_hash` in the config, so **any edit of the entry needs a new approval** (even `async` true → false).
- **No `"async": true`**, as for Claude Code: `codex exec` drops a background hook when it exits, before it posts (seen
  live on 0.163.0; the interactive TUI posted either way). `moth hook codex` rewrites an older `async: true` entry;
  approve it again afterwards.
- Hooks are a stable feature, enabled by default (`features/src/lib.rs`, `CodexHooks`, key `hooks`, at
  `rust-v0.163.0`). They are loaded from `<config folder>/hooks.json` (`hooks/src/engine/discovery.rs`).
- The Stop input carries `last_assistant_message` (`hooks/src/events/stop.rs`).
- User Stop hooks run only for root turns:
  - Thread-spawned subagents fire `SubagentStop`.
  - Internal subagents dispatch no hook at all.
  - Memory consolidation runs only managed hooks.
  
  Sources: `core/src/hook_runtime.rs` `run_turn_stop_hooks`, `stop.rs` `select_handlers`.
- Trust: `hooks/src/engine/discovery.rs` (`hook_trust_status`, handlers pushed only when Trusted or Managed),
  `tui/src/hooks_rpc.rs`, and `tui/src/startup_hooks_review.rs`.

**B. Legacy `notify`.** Add this to `~/.codex/config.toml`. There is only one `notify` slot, so this replaces any
existing notifier:

```toml
notify = ["python3", "/abs/path/to/moth-112m/code/moth/integrations/codex.py"]
```

- Codex appends `{"type":"agent-turn-complete", ..., "last-assistant-message": ...}` as the last argv argument and
  spawns the program with null stdio, without waiting for it (`hooks/src/legacy_notify.rs`).
- It is marked for removal ("TODO: Remove this hook ... when legacy `notify` support is removed").
- It is **not** gated by session source: `core/src/session/turn.rs` calls `run_legacy_after_agent_hook` for every
  turn. So subagent and memory-consolidation turns are probably posted too.
- Source: <https://github.com/openai/codex/tree/rust-v0.163.0/codex-rs>

## Hermes Agent

```bash
cp -r moth-112m/code/moth/integrations/hermes/moth ~/.hermes/plugins/moth
hermes plugins enable moth      # or add `moth` under plugins.enabled in ~/.hermes/config.yaml
```

- `post_llm_call` fires once per turn after the tool loop, with `assistant_response` set to the final text. It fires
  only on turns that produced a response and were not interrupted.
- `delegate_task` children run with `platform="subagent"` and are skipped.
- The POST runs on a daemon thread, so the hook returns at once.
- Sources: `website/docs/user-guide/features/hooks.md#post_llm_call` and
  `website/docs/developer-guide/plugins/index.md` in the Hermes repo. The firing site is
  `agent/turn_finalizer.py` `_apply_output_hooks`; the subagent platform is set in `tools/delegate_tool.py`.
  Checked against a local hermes-agent checkout at `c57331677d`, 2026-10-10.
- User plugins are opt-in (`plugins.enabled`).

## Pi

```bash
cp moth-112m/code/moth/integrations/pi/moth.ts ~/.pi/agent/extensions/moth.ts
# or for one run:  pi -e /abs/path/to/moth-112m/code/moth/integrations/pi/moth.ts
```

- `agent_end` (`messages` = the messages generated by that run) can be followed by retries, overflow recovery or
  queued work, so the adapter only remembers the last assistant text there.
- It posts on `agent_settled`, which means "Pi will not continue automatically", unless that event has `aborted`.
- `agent_start` clears the remembered text.
- Assistant messages with `stopReason` `error` or `aborted` are ignored. Only `text` blocks count.
- Sources: `docs/extensions.md`, `docs/json.md` and `docs/cli-integration.md` in the package, and
  <https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/extensions/types.ts>
  (`AgentStartEvent`, `AgentEndEvent`, `AgentSettledEvent`). Checked against main (v1.3.0) and the local 1.1.0
  `dist/core/extensions/types.d.ts`.
- Pi has no built-in subagents. Subagent extensions that spawn child `pi` processes will load this extension too,
  and those children will post.

## OpenCode

```bash
cp moth-112m/code/moth/integrations/opencode/moth.js ~/.config/opencode/plugins/moth.js     # global
# or: <project>/.opencode/plugins/moth.js
```

- Idle events carry only `sessionID`, so the adapter collects state from the event stream:
  - From `message.updated`, it records which message ids are assistant messages. Compaction (`summary`) messages
    are skipped; an `error` such as an abort means no reply.
  - From `message.part.updated`, it takes `text` parts that aren't `synthetic` or `ignored`. `part.text` is the
    full text, and the final update has `time.end`.
  - From `session.created` / `session.updated`, it treats sessions with an `info.parentID` as subagent (task tool)
    sessions and skips them.
- On `session.status` with `status.type === "idle"`, and on the deprecated `session.idle` fired at the same moment,
  it posts the newest assistant message's text and then forgets that session.
- Sources: <https://opencode.ai/docs/plugins/> (plugin dirs, `event` hook, `session.idle` / `session.status`), plus
  `packages/schema/src/session-status-event.ts` and `packages/schema/src/v1/session.ts` at tag `v1.19.3` in
  <https://github.com/anomalyco/opencode>, plus the local SDK types 1.2.10 (`@opencode-ai/sdk` `types.gen.d.ts`).

## dsh (DeepSeek Harness)

Add this to `~/.dsh/profiles/<profile>/cordis.patch.yml`, a YAML list, next to any existing entries:

```yaml
- insert: [{ id: moth, name: '/abs/path/to/moth-112m/code/moth/integrations/dsh/moth.mjs' }]
```

Or pass it for one run: `dsh --patch /path/to/patch.yml ...`, where the patch file holds the same list.

- `session/event` is the "post-commit, fire-and-forget append feed" of every session-log event. From it the adapter:
  - remembers the text blocks of the newest `assistant/message` in the turn, skipping `interrupted: true`;
  - on `turn/end` with `reason.kind` `completed` or `max-tokens`, posts that same turn's text;
  - posts nothing for aborted, error and blocked turns.
- Child sessions announced by `subagent/start` (`info.id`) are skipped.
- Sources: <https://github.com/deepseek-ai/deepseek-harness> `docs/subsystems/session.md` (`session/event`,
  `SessionEventMap`, `TurnEndReasonMap`), `docs/subsystems/subagent.md` and
  `packages/subagent/subagent/src/types.ts` (`SubagentRunInfo.id`), `packages/llm/llm/src/types.ts` (`TextBlock`).
  Checked at tags `dsh-v0.2.0-rc.2` (npm `latest`) and `dsh-v0.2.1-rc.2`. The `agent/turn-stopping` hook was not
  used because it can steer another step, while `turn/end` is the committed close.

## Updating

From `moth-112m/code`, with moth stopped: `python -m moth update --check` shows what changed (exit 10 = update
available), `python -m moth update` fetches only the changed files into the same folder, and `python -m moth update
--rollback` puts back the previous version. `python -m moth --version` prints the version and Hub revision. Hook
commands that point into `code/moth/integrations` keep working; re-copy any adapter you copied elsewhere (Pi
`moth.ts`, OpenCode `moth.js`, the Hermes plugin folder) when the update says integration files changed:
`python -m moth hooks` flags outdated copies and `python -m moth hook <harness>` replaces them. `~/.moth/`
(home, token) is never touched.

---

## Not verified live

All six adapters, the Claude Code band, the Pi widget and the OpenCode toast have since run live once, on
2026-10-18 (`../../video/raw/INDEX.md` in the dev checkout). That run found the `async` one-shot bug and the
cross-session display bug, both fixed since; the fixes were checked offline as described below, not in a new live run.
Each adapter is also checked with fake payloads shaped from the sources above (`test_integrations.py`):
- correct text posted;
- subagent, aborted and empty turns skipped;
- no output, exit 0;
- under 1.8 s against a moth that answers after 2 s;
- no crash with moth down or a bad URL.

Open points:
- **Session ids and labels** come from the payload fields in the table above, checked against sources (Claude Code
  docs, Codex `stop.rs` / `legacy_notify.rs` at `rust-v0.163.0`, Hermes `agent/turn_finalizer.py` and
  `agent/runtime_cwd.py` at c57331677d, Pi 1.1.0 `ExtensionContext` types, OpenCode SDK 1.2.10 `Session`), and only
  with fake payloads. dsh's session object was not checked for a cwd field, so its label is `process.cwd()`. Hermes
  in a messaging gateway without a configured cwd labels sessions with the gateway's launch folder.
- **Claude Code and Codex Stop**: if another Stop hook blocks and makes the agent keep going, the reply from before
  the block has already been posted, and the later reply posts too. An async observer can't tell them apart.
- **Pi**: a follow-up you queue mid-run runs before `agent_settled`, so only the last reply in that chain posts.
- **Claude Code / Codex Stop hooks** were run live on 2026-10-18 (Claude Code 2.1.302, Codex 0.163.0,
  `../../video/raw/INDEX.md` in the dev checkout): `async: true` posted from the interactive TUIs but not from
  `claude -p` / `codex exec`, so the hooks now run in the foreground. The foreground hook was only re-checked
  here with a simulated harness exit, not in a new live run.
- **dsh**:
  - Not confirmed: whether a root-level `ctx.on('session/event')` receives events with no `inject`, and whether
    `session.id` equals `SubagentRunInfo.id` for in-process children.
  - Remote (`local: false`) subagents emit no local session events anyway.
- **OpenCode**:
  - Not confirmed live: that the assistant `message.updated` arrives before its idle event.
  - In one-shot `opencode run` / `pi -p`, the process may exit before the POST lands.
- **Pi**: child `pi` processes spawned by subagent extensions are not filtered.
- **Hermes**: in one-shot runs (`hermes -q`), the daemon thread can die at exit before posting.

Display layer: everything was checked against a stub moth with a real log (`Display` in `test_integrations.py`):
- `moth tail`: the watch output, wrapping, history, restart, chat echo, and "asleep" said once. Interactive
  redraw was checked by hand in a pty (libedit Python 3.14).
- Hermes registers no slash command.
- Pet `plugin.js` under node with a fake `ctx`: module shape, history skipped, the `moth` id, ttl bounds, 120-char
  pieces, moth down, and no `ctx.pet`.
- Pi widget and `/moth` with a fake `ctx.ui`.
- OpenCode toast with a fake `client.tui`. The event hook doesn't wait on the poll.
- Per-session displays: another harness's reaction landing last is not shown by the Pi widget, two OpenCode
  sessions in one process each get their own toast, the Hermes Desktop pet voices Hermes reactions only, and
  serve.py's `/api/says?source=&session=` filter (stripped answer, `next` unchanged, no ids leaked). The Claude
  Code band's filter passed `claude plugin validate` only; it was not run in a live session.
- Pi's `/moth`: a reply slower than 4 s is picked up by an unfiltered poll (chat replies carry no session), so in
  that rare case another session's reaction could show instead.
- `show_in_harness`: off by default in Pi and OpenCode (the turn is still posted, no `/api/says` poll, no widget or
  toast; Pi's `/moth` answers with a notice); the env beating the file both ways; a live toggle on → off → on of the
  file in one Pi / OpenCode process (the widget is taken down, and turning it back on never shows a line from the off
  turn); Pi's reader and `moth/config.py` agreeing on every spelling. The `python -m moth config` CLI and the browser
  opt-in of `python -m moth` are in `tests/test_config.py`.
- `serve.py` on the real handler, with torch and the model faked out (`Serve`): the Host allow-list (look-alikes,
  a LAN IP, no port, no Host), the token on every guarded route, CORS (local pages everywhere, desktop-shell origins
  on `/api/says` only, look-alikes such as `localhost.evil.example` nowhere), foreign-origin / wrong-type / oversized
  / malformed POSTs, a bad `since`, `X-Frame-Options`, the token in the page but no CORS on it, `source` filtering, and
  "new home" refusing a folder without moth's marker.
- Every adapter sends the token and its `source`; the Claude Code hook posts past an `http_proxy` in its environment.
- Every adapter's `session` and `label` from fake payloads, end to end through the real handler and the real
  `Moth.hear_agent` onto the log's cards (`AdapterToLog`); their cleaning in serve.py (id charset, 40-char one-line
  label, ANSI stripped); the session list only in the token-guarded `/api/log`, never in `/api/says`; and
  `moth tail --session`.

Not verified:
- **Claude Code mod**: only `claude plugin validate` was run, which is static. Not confirmed live:
  - that `$.http.fetch` reaches `127.0.0.1` from the hooks worker;
  - that `$.clock.every` started in `session.start` keeps ticking;
  - how the band looks;
  - the band's text field: focusing it (ctrl+x tab or a click), Enter sending, and the field coming back empty
    and focused after a send (it is redrawn under a new key);
  - whether the poller survives a mid-session reload of the mod. It is started in `session.start`, and a reload
    drops timers.
  `claude plugin test` was not used, because it starts the engine host.
- **Hermes Desktop**:
  - that the packaged renderer really sends `Origin: null`, or `file://`, both of which are allowed;
  - that the renderer's `fetch` to `http://127.0.0.1:4747` isn't blocked by a CSP or by Chromium's
    private-network rules. serve.py deliberately does not send `Access-Control-Allow-Private-Network`;
  - that the copied desktop half shows up in the Plugins page as described in `electron/desktop-plugins-root.ts`.
- **Pi**: how ANSI color looks inside a string-array widget. The interactive renderer wraps each string in a `Text`.
- **OpenCode**: that `client.tui.showToast` works from a plugin in 1.2.10 (the SDK types have it), and that
  `opencode run` exits promptly with the poll pending.
- **tail**: readline's redraw is called from the poll thread. It is cosmetic, but GNU readline was not tried.
- **Token reads inside harnesses**: that the Claude Code mod's `$.fs.read` may read `~/.moth/token` (outside the
  session folder) without an fs hook objecting, and that Pi's jiti and OpenCode's Bun load the `node:fs` imports.
- **Proxies in JS**: Node's `fetch` ignores `HTTP_PROXY`, but Bun's (OpenCode) honours it; with a proxy set and no
  `NO_PROXY=127.0.0.1`, OpenCode's posts may go to the proxy.
