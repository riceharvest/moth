/**
 * moth for Hermes Desktop: moth's reaction to each agent turn, said by the desktop pet.
 *
 * The desktop half of the `moth` agent plugin (../__init__.py posts each turn's reply to moth; this half only
 * shows what moth says back). It polls moth's own lines (GET /api/says, every 2.5 s, through ctx.setInterval so the
 * poller dies with the plugin) and hands each new moth line to the pet bubble with `ctx.pet.say`
 * (apps/desktop/src/contrib/plugin.ts PluginPet; store/pet-plugin-messages.ts):
 *   - plain text, 120 chars max: a longer line is cut into up to 3 pieces shown one after another;
 *   - one stable id ('moth'), so a newer line replaces the old one in place (no pile-up, well under the
 *     10-per-10 s rate limit);
 *   - ttl from the length, inside the host's 1-30 s clamp.
 * It never posts anything to moth, so moth never hears a turn twice. History is skipped on start; when moth
 * restarts or the chat is reset (a new `epoch`) it follows the new log. Moth down = silence, errors swallowed.
 * No imports: the loader only allows @hermes/plugin-sdk and react, and this needs neither (global fetch).
 * It asks only for moth's reactions to Hermes turns (`source=hermes`), so the pet never voices moth's reply to
 * Claude Code, Codex or another harness running next to it. Not narrower: the desktop's gateway events carry the
 * TUI session id, while the agent posts its session key (tui_gateway _make_agent(sid, key, session_id=key)), so
 * the renderer can't tell which Hermes session a line answers. Your own chat with moth (moth tail) isn't voiced.
 * /api/says is moth's lines and nothing else (no agent replies, nothing you typed), so it needs no token: the
 * renderer can't read ~/.moth/token anyway. serve.py lets the packaged desktop page (file://, Origin `null`)
 * and the dev renderer (http://127.0.0.1:5174) read it; the full log stays token-only.
 * The pet must be on (Settings) for lines to show; `ctx.pet` is feature-detected for older desktops.
 */
const MOTH = 'http://127.0.0.1:4747'
const POLL_MS = 2500
const MAX = 118

/** Cut a long line at word boundaries into pieces the bubble shows whole. */
function pieces(text) {
  const out = []
  let rest = String(text || '').replace(/\s+/g, ' ').trim()
  while (rest && out.length < 3) {
    if (rest.length <= MAX) { out.push(rest); break }
    let cut = rest.lastIndexOf(' ', MAX)
    if (cut < MAX / 2) cut = MAX
    out.push(out.length === 2 ? rest.slice(0, MAX) + '…' : rest.slice(0, cut))
    rest = rest.slice(cut).trim()
  }
  return out
}

const ttlFor = (piece) => Math.min(30000, Math.max(4000, 2500 + piece.length * 70))

async function getSays(since, epoch) {
  const q = (epoch ? `since=${since}&epoch=${encodeURIComponent(epoch)}` : `since=${since}`) + '&source=hermes'
  const r = await fetch(`${MOTH}/api/says?${q}`, { signal: AbortSignal.timeout(2000) })
  const d = await r.json()
  if (!Array.isArray(d.events) || typeof d.next !== 'number') throw new Error('not moth')
  return d
}

export default {
  id: 'moth',
  name: 'moth',
  description: "Shows moth's reaction to each agent turn in the pet's speech bubble (needs moth running: python -m moth.serve).",
  register(ctx) {
    if (!ctx.pet || typeof ctx.pet.say !== 'function') return
    let cursor = null   // index into moth's log; null = skip whatever is there now
    let epoch = null    // a new one = moth restarted or the chat was reset; the server then sends the new log
    let busy = false
    let cancelQueue = []

    function say(text) {
      cancelQueue.forEach((cancel) => cancel())
      cancelQueue = []
      let at = 0
      for (const piece of pieces(text)) {
        const show = () => ctx.pet.say(piece, { id: 'moth', ttlMs: ttlFor(piece) })
        if (at === 0) show()
        else cancelQueue.push(ctx.setTimeout(show, at))
        at += ttlFor(piece) - 500
      }
    }

    async function poll() {
      if (busy) return
      busy = true
      try {
        if (cursor === null) {
          const d = await getSays(1e9)   // events: [] and the current end of the log
          cursor = d.next
          epoch = d.epoch
          return
        }
        const d = await getSays(cursor, epoch)
        cursor = d.next
        epoch = d.epoch
        const says = d.events.filter((e) => e && typeof e.say === 'string' && e.say.trim())
        if (says.length) say(says[says.length - 1].say)
      } catch {
        /* moth asleep or slow: try again next tick */
      } finally {
        busy = false
      }
    }

    ctx.setInterval(poll, POLL_MS)
    poll()
  }
}
