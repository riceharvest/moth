// The mod's state, as the engine's PluginState contract (named in plugin.json "types").
export type MothLine = string | null

declare module 'claude-code' {
  interface PluginState {
    moth: { line: string | null; asleep: boolean; sent: number }
  }
}
