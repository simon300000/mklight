import { describe, expect, it } from 'vitest'
import { COMMANDS } from './commands'
import { LAYOUT_SHORTCUTS } from '@/views/layout/shortcuts'

/** The shortcut a key event produces on a platform (as `platformShortcut`). */
function on(mac: boolean, s: string): string {
  const parts = s.split('+')
  const key = parts.pop()!
  const mods = new Set(parts)
  if (!mac && mods.has('ctrl')) {
    mods.delete('ctrl')
    mods.add('alt')
    mods.add('shift')
  }
  return [...['ctrl', 'alt', 'shift', 'mod'].filter((m) => mods.has(m)), key.toLowerCase()].join('+')
}

/** Keys the browsers keep for themselves (tabs, zoom, reload, windows, developer tools). */
const BROWSER = ['mod+t', 'mod+n', 'mod+w', 'shift+mod+t', 'shift+mod+n', 'mod+r', 'shift+mod+r', 'mod+0', 'alt+mod+i', 'alt+mod+j', 'mod+l', 'mod+q']
  .concat([1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => `mod+${d}`))

describe('shortcuts', () => {
  const all = [...COMMANDS.map((c) => c.shortcut), ...LAYOUT_SHORTCUTS.map((c) => c.shortcut)].filter((s): s is string => !!s)

  it.each([true, false])('no two commands share a key (Mac: %s)', (mac) => {
    const keys = all.map((s) => on(mac, s))
    expect(keys.filter((k, i) => keys.indexOf(k) !== i)).toEqual([])
  })

  it('none takes a key the browser keeps', () => {
    expect(all.map((s) => on(true, s)).filter((k) => BROWSER.includes(k))).toEqual([])
  })
})
