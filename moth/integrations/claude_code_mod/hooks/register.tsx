// moth for Claude Code, as a mod (function hooks, early-access API; typed against Claude Code 2.1.302's
// claude-code.d.ts from the bundled plugin-authoring skill).
//
// Display only. The Stop command hook (../../claude_code.py) is what tells moth about each turn; this mod never
// posts to /api/agent, so moth never hears a turn twice. It:
//   - polls moth's own lines (GET /api/says, no token) every 2 s with $.clock.every + $.http.fetch, started in
//     session.start, asking only for moth's reactions to THIS session (`source=claude-code&session=<id>`, the id
//     being $.session.id(), the transcript name, which is the Stop hook's session_id). It is read on every tick,
//     since /clear starts a new session without a new session.start. With several agents running, the band so never
//     shows moth's reply to another session; with no id, it shows no reactions at all;
//   - draws, in the AbovePrompt band, moth's newest line ("moth> ...", lamp-yellow, at most two rows) and a text
//     field to talk to it (ctrl+x tab or a click focuses the band; Enter sends; Esc goes back to the prompt). The
//     reply to what you type there comes in /api/chat's own answer.
// Talking to moth stays out of Claude's context: the field's Enter raises `ui.input` to this plugin alone ("No
// model turn unless it asks one", InputProps.onSubmit), the band is drawn UI, not a transcript row. There is no
// /moth slash command on purpose: a command's output row is read by the model (CommandRunResult.text).
// Every network call is caught: moth down never breaks a session (the band hides). History is skipped at start,
// and when moth restarts or the chat is reset (a new epoch) the poll follows the new log. MOTH_URL overrides the
// address. Talking needs moth's per-install token: read from ~/.moth/token (MOTH_TOKEN_FILE overrides) on each send.
// moth.line / moth.asleep / moth.sent are declared in ../types/index.d.ts, the PluginState contract plugin.json names.
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

const POLL_MS = 2000
const line = atom({ plugin: 'moth', key: 'line' } as const, null)
const asleep = atom({ plugin: 'moth', key: 'asleep' } as const, true)
const sent = atom({ plugin: 'moth', key: 'sent' } as const, 0)   // bumps the field's key: a fresh, empty field

type Ev = { say?: unknown }
type Log = { events: Ev[]; next: number; epoch?: string; error?: string }
const ANSI = /\x1b\[[0-?]*[ -\/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f-\x9f]/g

const saysOf = (events: Ev[]) =>
  events.filter(e => e && typeof e.say === 'string' && e.say.trim()).map(e => String(e.say))

let url = 'http://127.0.0.1:4747'   // MOTH_URL overrides it, read at session start

async function token($: EngineInterface): Promise<string> {
  try {
    const home = (await $.env.get('HOME')) || (await $.env.get('USERPROFILE')) || ''
    const file = (await $.env.get('MOTH_TOKEN_FILE')) || `${home}/.moth/token`
    return String(await $.fs.read(file)).trim()
  } catch {
    return ''   // moth never ran here: it answers 401 and the band says so
  }
}

// The field's Enter: straight to moth's /api/chat, never through the prompt.
async function talk($: EngineInterface, value: string) {
  const text = value.trim().slice(0, 500)
  if (!text) return
  try {
    await update($, sent, n => (n ?? 0) + 1)
    await update($, line, () => '...')
    const r = await $.http.fetch(`${url}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Moth-Token': await token($) },
      body: JSON.stringify({ text }),
    })
    const d = JSON.parse(r.text) as Log
    if (!r.ok) {
      await update($, line, () => `(moth didn't take that: ${d.error || r.status})`)
      return
    }
    const says = saysOf(d.events || [])
    await update($, line, () => says.join(' ') || '...')
  } catch {
    await update($, line, () => "(moth is asleep and didn't hear that)").catch(() => {})
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    try {
      url = ((await $.env.get('MOTH_URL')) || url).replace(/\/+$/, '')
    } catch { /* keep the default */ }
    let cursor: number | null = null   // index into moth's log; null = skip what is there now
    let epoch: string | undefined      // a new one = moth restarted or the chat was reset: the server sends the new log
    let busy = false

    let session: string | null = null   // this session's id: only moth's reactions to it are shown

    const getSays = async (since: number, ep?: string): Promise<Log> => {
      let q = ep ? `since=${since}&epoch=${encodeURIComponent(ep)}` : `since=${since}`
      if (session) q += `&source=claude-code&session=${encodeURIComponent(session)}`
      const r = await $.http.fetch(`${url}/api/says?${q}`)
      const d = JSON.parse(r.text) as Log
      if (!Array.isArray(d.events) || typeof d.next !== 'number') throw new Error('not moth')
      return d
    }

    const tick = async () => {
      if (busy) return
      busy = true
      try {
        let id: string | null = null
        try {
          id = String((await $.session.id()) || '') || null
        } catch { /* an older build without session.id: no reactions shown, the field still works */ }
        if (id !== session) {   // first tick, or /clear / resume: a new session, so not the old session's line
          if (session !== null) await update($, line, () => null)
          session = id
        }
        if (cursor === null) {
          const d = await getSays(1e9)   // no events, just the current end of the log
          cursor = d.next
          epoch = d.epoch
        } else {
          const d = await getSays(cursor, epoch)
          cursor = d.next
          epoch = d.epoch
          const says = session ? saysOf(d.events) : []   // no id: no filter possible, so show nothing
          if (says.length) await update($, line, () => says[says.length - 1]!)
        }
        if (await read($, asleep)) await update($, asleep, () => false)
      } catch {
        try {
          if (!(await read($, asleep))) await update($, asleep, () => true)
        } catch { /* state unavailable: nothing to draw anyway */ }
      } finally {
        busy = false
      }
    }

    try {
      $.clock.every(POLL_MS, () => void tick())
    } catch {
      /* an older build without these nouns: the session goes on without moth */
    }
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey || (await read($, asleep))) return next(e)
    const said = await read($, line)
    const n = await read($, sent)
    const { Box, Text } = $.ui.resolve(e)
    const room = Math.max(10, e.props.bodyColumns * 2 - 8)   // two rows at most
    const flat = (said || '').replace(ANSI, '').replace(/\s+/g, ' ').trim()
    let field = null
    if (e.surface !== 'mobile') {   // mobile's table has no Input: the line alone
      const { Input } = $.ui.resolve(e)
      field = (
        <Input
          key={`talk-${n}`}
          placeholder="talk to moth (ctrl+x tab, Enter; Claude never sees it)"
          submitLabel="tell moth"
          onSubmit={value => void talk($, value)}
        />
      )
    }
    return (
      <Box flexDirection="column">
        {flat && (
          <Text wrap="wrap">
            <Text color="warning">moth&gt; </Text>
            {flat.length > room ? flat.slice(0, room - 1) + '…' : flat}
          </Text>
        )}
        {field}
      </Box>
    )
  })
}
