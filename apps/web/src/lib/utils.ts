import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)

/** A shortcut as shown in menus: ⌘ on Mac, Ctrl elsewhere. `mod` is ⌘/Ctrl. */
export function shortcutLabel(s: string): string {
  const parts = s.split('+')
  const key = parts.pop()!
  const names: Record<string, string> = { Backspace: '⌫', Enter: '↩', Escape: 'Esc', ArrowUp: '↑', ArrowDown: '↓', Period: '.' }
  const k = names[key] ?? key.toUpperCase()
  if (isMac) {
    const sym: Record<string, string> = { ctrl: '⌃', alt: '⌥', shift: '⇧', mod: '⌘' }
    const order = ['ctrl', 'alt', 'shift', 'mod']
    return order.filter((m) => parts.includes(m)).map((m) => sym[m]).join('') + k
  }
  const word: Record<string, string> = { ctrl: 'Ctrl', alt: 'Alt', shift: 'Shift', mod: 'Ctrl' }
  const mods = [...new Set(parts.map((m) => word[m]))]
  return [...mods, k].join('+')
}
