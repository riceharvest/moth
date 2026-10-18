// moth plugin for OpenCode (opencode-ai 1.2-1.19): send the agent's final reply of each turn to the local moth
// companion, and optionally show moth's reaction as a toast. Install: copy to ~/.config/opencode/plugins/moth.js
// (global) or .opencode/plugins/moth.js (project); files there load at startup (docs: opencode.ai/docs/plugins).
// The idle event carries only a sessionID, so the `event` hook keeps a little state from the bus
// (schema: packages/schema/src/v1/session.ts and session-status-event.ts at v1.19.3):
//   message.updated        info.role === "assistant": remember the message id per session, in arrival order;
//                          `summary` (compaction) messages are skipped, an `error` (abort, API error) means no reply.
//   message.part.updated   type "text" parts, minus `synthetic` / `ignored`: part.text is the full text so far
//                          (streaming deltas go to message.part.delta), so the last update wins.
//   session.created/updated  info.parentID set = a subagent (task tool) session: never posted. info.directory is
//                          remembered for the label.
//   session.status {type: "idle"}  (and the deprecated session.idle, same moment) = the turn is over: post the text
//                          of the newest assistant message that has any, then forget the session's state, so a
//                          turn without a reply posts nothing and the second idle event finds nothing.
// Session and label, so moth can tell several sessions apart: the sessionID, and the basename of the session's
// directory (Session.directory, SDK types.gen.d.ts), else of the plugin input's `directory`.
// Fire-and-forget: fetch with a 1 s timeout, never awaited, every rejection swallowed, nothing printed.
// Display: after posting, poll moth's lines (GET /api/says) every 0.5 s for up to WAIT_MS and show moth's newest line
// as a TUI toast (client.tui.showToast, SDK 1.2.10 TuiShowToastData). The poll is started, never awaited, so the
// event bus is not held up; its timers are unref'd. The log cursor is taken when a turn starts (session.status
// busy) and persists, so a reaction slower than WAIT_MS shows after the next turn instead of being lost.
// Per session: the poll asks only for moth's reactions to the session that just posted (`source=opencode&
// session=<sessionID>`), and the cursor, epoch and newest-poll counter are kept per sessionID, so two sessions in
// one OpenCode neither cancel each other's toast nor skip each other's line, and no toast shows another agent's
// reply.
// The toast is OFF by default (moth's own setting `show_in_harness`, see moth/config.py): enabled() reads
// MOTH_SHOW_IN_HARNESS (1/0, on/off, true/false, yes/no), else "show_in_harness" in ~/.moth/config.json
// (MOTH_CONFIG_FILE overrides), at every turn, so `python -m moth config show_in_harness on|off` applies from the
// next turn without restarting OpenCode. Off: the reply is still posted (the desktop pet reacts), no poll, no toast.
// Only ONE export: OpenCode treats every exported function as a plugin. Env: MOTH_URL (default http://127.0.0.1:4747);
// the token is read from ~/.moth/token on every post (MOTH_TOKEN_FILE overrides).
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const BASE = (process.env.MOTH_URL || "http://127.0.0.1:4747").replace(/\/+$/, "");
const MOTH = BASE + "/api/agent";
const TOKEN_FILE = process.env.MOTH_TOKEN_FILE || join(homedir(), ".moth", "token");
const WAIT_MS = 4000;

function token() {
  try { return readFileSync(TOKEN_FILE, "utf8").trim(); } catch (e) { return ""; }
}

function flag(v) {   // the same spellings as moth/config.py parse_bool; null = not a setting
  if (typeof v === "boolean") return v;
  if (typeof v !== "string") return null;
  const s = v.trim().toLowerCase();
  return ["1", "on", "true", "yes"].includes(s) ? true : ["0", "off", "false", "no"].includes(s) ? false : null;
}

function enabled() {   // show moth's replies in OpenCode? env, else ~/.moth/config.json, else off. Read every turn.
  const env = flag(process.env.MOTH_SHOW_IN_HARNESS || "");
  if (env !== null) return env;
  try {
    const file = process.env.MOTH_CONFIG_FILE || join(homedir(), ".moth", "config.json");
    const v = flag((JSON.parse(readFileSync(file, "utf8")) || {}).show_in_harness);
    if (v !== null) return v;
  } catch (e) { /* no file / bad JSON: the default */ }
  return false;
}

function labelOf(dir, session) {   // the project folder's name; without one, a short session id
  const name = String(dir || "").replace(/[\/\\]+$/, "").split(/[\/\\]/).pop();
  return name || String(session || "").slice(0, 8);
}

function body(text, source, session, label) {   // session / label only when known
  const b = { text, source };
  if (session) b.session = String(session);
  if (label) b.label = String(label);
  return JSON.stringify(b);
}

function post(text, session, label) {
  try {
    fetch(MOTH, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Moth-Token": token() },
      body: body(text, "opencode", session, label), signal: AbortSignal.timeout(1000),
    }).catch(() => {});
  } catch (e) { /* drop it */ }
}

async function getSays(since, epoch, session) {   // moth's own lines only: no token needed
  let q = epoch ? `since=${since}&epoch=${encodeURIComponent(epoch)}` : `since=${since}`;
  if (session) q += `&source=opencode&session=${encodeURIComponent(session)}`;   // only moth's reactions to it
  const r = await fetch(`${BASE}/api/says?${q}`, { signal: AbortSignal.timeout(1000) });
  const d = await r.json();
  if (!Array.isArray(d.events) || typeof d.next !== "number") throw new Error("not moth");
  return d;
}

export const Moth = async ({ client, directory } = {}) => {
  const replies = new Map();   // sessionID -> [{ id, skip, error }] assistant messages in arrival order
  const texts = new Map();     // messageID -> { sid, parts: Map(partID -> text) }
  const children = new Set();  // subagent session ids
  const dirs = new Map();      // sessionID -> its directory (for the label)
  const canToast = Boolean(client && client.tui && typeof client.tui.showToast === "function");
  // sessionID -> { cursor: index into moth's log (null = not known yet), epoch: changes when moth restarts or the
  // chat is reset (the server then sends the new log), run: the newest poll of that session wins }
  const polls = new Map();
  const pollOf = (sid) => {
    let p = polls.get(sid);
    if (!p) polls.set(sid, (p = { cursor: null, epoch: null, run: 0 }));
    return p;
  };

  function toast(text) {
    try {
      const flat = String(text).replace(/\s+/g, " ").trim();
      Promise.resolve(client.tui.showToast({ body: { title: "moth", message: flat, variant: "info", duration: 8000 } }))
        .catch(() => {});
    } catch (e) { /* no TUI */ }
  }

  const start = (p, d) => { if (p.cursor === null) { p.cursor = d.next; p.epoch = d.epoch; } };

  function baseline(sid) {
    const p = pollOf(sid);
    if (canToast && p.cursor === null && enabled()) getSays(1e9).then((d) => start(p, d), () => {});
  }

  // Setting off: stop every poll and forget the cursors, so turning it back on starts fresh, never with an old line.
  function quiet() { for (const p of polls.values()) { p.cursor = null; p.epoch = null; p.run++; } }

  async function refresh(sid) {   // one poll: true once a new moth line (a reaction to this session) is shown
    const p = pollOf(sid);
    if (p.cursor === null) { start(p, await getSays(1e9)); return false; }
    const d = await getSays(p.cursor, p.epoch, sid);
    p.cursor = d.next; p.epoch = d.epoch;
    const says = d.events.filter((e) => e && typeof e.say === "string" && e.say.trim());
    if (says.length) toast(says[says.length - 1].say);
    return says.length > 0;
  }

  function watch(sid) {
    if (!canToast || !sid) return;
    const p = pollOf(sid), mine = ++p.run, until = Date.now() + WAIT_MS;
    const step = async () => {
      let done = false;
      try { done = await refresh(sid); } catch (e) { /* moth asleep: keep trying until the window ends */ }
      if (!done && mine === p.run && Date.now() < until) { const t = setTimeout(step, 500); t.unref?.(); }
    };
    const t = setTimeout(step, 300); t.unref?.();
  }

  function finalText(sid) {
    for (const m of [...(replies.get(sid) || [])].reverse()) {
      if (m.error) return "";
      if (m.skip) continue;
      const t = [...((texts.get(m.id) || {}).parts || new Map()).values()].join("\n");
      if (t.trim()) return t;
    }
    return "";
  }

  function forget(sid) {
    replies.delete(sid);
    for (const [id, t] of texts) if (t.sid === sid) texts.delete(id);
  }

  return {
    event: async ({ event }) => {
      try {
        const p = (event && event.properties) || {};
        switch (event && event.type) {
          case "session.created":
          case "session.updated":
            if (p.info && p.info.parentID) children.add(p.info.id);
            if (p.info && p.info.id && typeof p.info.directory === "string") dirs.set(p.info.id, p.info.directory);
            break;
          case "message.updated": {
            const m = p.info;
            if (!m || m.role !== "assistant" || !m.sessionID) break;
            const list = replies.get(m.sessionID) || [];
            let r = list.find((x) => x.id === m.id);
            if (!r) list.push((r = { id: m.id }));
            r.skip = Boolean(m.summary);
            r.error = Boolean(m.error);
            replies.set(m.sessionID, list);
            break;
          }
          case "message.part.updated": {
            const part = p.part;
            if (!part || part.type !== "text" || part.synthetic || part.ignored || typeof part.text !== "string") break;
            const t = texts.get(part.messageID) || { sid: part.sessionID, parts: new Map() };
            t.parts.set(part.id, part.text);
            texts.set(part.messageID, t);
            break;
          }
          case "session.status":
          case "session.idle": {
            if (event.type === "session.status" && !(p.status && p.status.type === "idle")) {
              if (p.status && p.status.type === "busy" && !children.has(p.sessionID)) baseline(p.sessionID);
              break;
            }
            const sid = p.sessionID;
            const text = finalText(sid);
            forget(sid);
            if (text && !children.has(sid)) {
              post(text, sid, labelOf(dirs.get(sid) || directory, sid));   // always: the pet reacts either way
              if (enabled()) watch(sid); else quiet();
            }
            break;
          }
        }
      } catch (e) { /* fail open */ }
    },
  };
};
