// moth for DeepSeek Harness (dsh, @deepseek-ai/dsh 0.2.x): send the agent's final reply of each turn to the local
// moth companion (one-way; moth never talks back). A cordis plugin; load it with a profile patch line in
// ~/.dsh/profiles/<profile>/cordis.patch.yml:   - insert: [{ id: moth, name: '/abs/path/moth.mjs' }]
// Events (docs/subsystems/session.md "session/event" + SessionEventMap + TurnEndReasonMap; docs/subsystems/subagent.md
// "subagent/start"; checked at tags dsh-v0.2.0-rc.2 and dsh-v0.2.1-rc.2):
//   session/event      post-commit, fire-and-forget feed of every appended session-log event (session, event):
//     turn/start       forget the session's remembered text.
//     assistant/message  one per step; remember the text blocks of the newest one with text, tagged with its turn.
//                      `interrupted: true` (cancelled mid-stream) is not a reply.
//     turn/end         reason.kind "completed" or "max-tokens": post the text remembered for that same turn.
//                      aborted / error / blocked turns post nothing.
//   subagent/start     info.id = the child agent's session id: its turns are never posted.
// Session and label, so moth can tell several sessions apart: session.id, and the basename of process.cwd() (dsh is
// started in the project; no cwd field on the session was checked in the dsh sources).
// Fire-and-forget: fetch with a 1 s timeout, never awaited, every rejection swallowed, nothing printed.
// Env: MOTH_URL (default http://127.0.0.1:4747); the token is read from ~/.moth/token on every post
// (MOTH_TOKEN_FILE overrides).
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

export const name = 'moth'

const MOTH = (process.env.MOTH_URL || 'http://127.0.0.1:4747').replace(/\/+$/, '') + '/api/agent'
const TOKEN_FILE = process.env.MOTH_TOKEN_FILE || join(homedir(), '.moth', 'token')

function token() {
  try { return readFileSync(TOKEN_FILE, 'utf8').trim() } catch { return '' }
}

function labelOf(dir, session) {   // the project folder's name; without one, a short session id
  const name = String(dir || '').replace(/[\/\\]+$/, '').split(/[\/\\]/).pop()
  return name || String(session || '').slice(0, 8)
}

function body(text, source, session, label) {   // session / label only when known
  const b = { text, source }
  if (session) b.session = String(session)
  if (label) b.label = String(label)
  return JSON.stringify(b)
}

function post(text, session, label) {
  try {
    fetch(MOTH, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Moth-Token': token() },
      body: body(text, 'dsh', session, label), signal: AbortSignal.timeout(1000),
    }).catch(() => {})
  } catch { /* drop it */ }
}

export function apply(ctx) {
  const last = new Map()       // session id -> { turn, text }
  const children = new Set()   // subagent session ids

  ctx.on('subagent/start', (info) => {
    try { if (info && info.id) children.add(String(info.id)) } catch { /* fail open */ }
  })

  ctx.on('session/event', (session, event) => {
    try {
      const sid = String(session && session.id)
      const d = (event && event.data) || {}
      switch (event && event.type) {
        case 'turn/start':
          last.delete(sid)
          break
        case 'assistant/message': {
          if (d.interrupted) break
          const blocks = d.message && Array.isArray(d.message.content) ? d.message.content : []
          const text = blocks.filter((b) => b && b.type === 'text' && typeof b.text === 'string').map((b) => b.text).join('\n')
          if (text.trim()) last.set(sid, { turn: d.turn, text })
          break
        }
        case 'turn/end': {
          const l = last.get(sid)
          last.delete(sid)
          const kind = d.reason && d.reason.kind
          if (l && l.turn === d.turn && (kind === 'completed' || kind === 'max-tokens') && !children.has(sid)) post(l.text, sid, labelOf(process.cwd(), sid))
          break
        }
      }
    } catch { /* fail open */ }
  })
}
