// Separate windows, the browser way (docs/APP.md "Panels and windows"; the
// macOS app's `DetachedWindows`). A separate window is a same-origin popup
// (`window.open('')`, an about:blank document that shares this page's
// origin): the app copies its stylesheets into it and renders the window's
// panels there through a React portal from the main window's React tree. So
// the popup has no state of its own: the document, the workspace and the
// window state are the main window's stores, and every edit, selection,
// analysis update and preview is live in both, as with the macOS app's shared
// `DesignSession`. Popups close with the main window (and open again with
// the project), and a browser may block a popup that does not follow a click.

import { create } from 'zustand'
import { MAIN, type Frame } from '@/state/arrangement'

export interface Popup {
  id: string
  win: Window
  /** The element the window's panels render into. */
  mount: HTMLElement
}

interface PopupState {
  open: Record<string, Popup>
}

export const usePopups = create<PopupState>()(() => ({ open: {} }))

/** Called when the user closes a popup (not when the app does). */
let onUserClose: (id: string) => void = () => {}
let onFrameChange: (id: string, frame: Frame) => void = () => {}
export function setPopupHandlers(h: { closed: (id: string) => void; frame: (id: string, f: Frame) => void }) {
  onUserClose = h.closed
  onFrameChange = h.frame
}

const programmatic = new Set<string>()
let tearingDown = false

// MARK: Screen geometry

/** Per window: screen position of its client area's origin, measured from
 *  pointer events (screenX − clientX), else estimated from the window frame. */
const origins = new Map<string, { x: number; y: number }>()
/** When each window last had the focus: the front window wins a hit test. */
const focusTimes = new Map<string, number>()

export function windowOf(id: string): Window | null {
  if (id === MAIN) return window
  return usePopups.getState().open[id]?.win ?? null
}

export function windowIds(): string[] {
  return [MAIN, ...Object.keys(usePopups.getState().open)]
}

export function clientOrigin(id: string): { x: number; y: number } | null {
  const w = windowOf(id)
  if (!w || w.closed) return null
  const m = origins.get(id)
  if (m) return m
  return { x: w.screenX + (w.outerWidth - w.innerWidth) / 2, y: w.screenY + Math.max(w.outerHeight - w.innerHeight, 0) }
}

/** Per window: its outer frame on screen (title bar included), read when the
 *  window opens, gains the focus, resizes or moves, and polled during a drag. */
const frames = new Map<string, Frame>()

/** Reads every window's frame now (cheap; called on each pointer move of a drag). */
export function refreshFrames() {
  for (const id of windowIds()) {
    const w = windowOf(id)
    if (w && !w.closed) frames.set(id, frameOf(w))
    else frames.delete(id)
  }
}

/** A window's outer frame on screen, as last read. */
export function outerFrame(id: string): Frame | null {
  const w = windowOf(id)
  if (!w || w.closed) return null
  return frames.get(id) ?? frameOf(w)
}

/** Brings a window to the front of the hit-test order (it was focused or clicked). */
export function markFront(id: string) {
  focusTimes.set(id, performance.now())
}

/** The design's windows whose frame holds a screen point, front (most
 *  recently focused) first: a drop is judged against the front one. */
export function windowsAt(sx: number, sy: number): string[] {
  const ids = windowIds().filter((id) => {
    const f = outerFrame(id)
    return !!f && sx >= f[0] && sx < f[0] + f[2] && sy >= f[1] && sy < f[1] + f[3]
  })
  return ids.sort((a, b) => (focusTimes.get(b) ?? 0) - (focusTimes.get(a) ?? 0))
}

/** Whether a screen point is in a window's content (not its title bar or border). */
export function inClientArea(id: string, sx: number, sy: number): boolean {
  const w = windowOf(id)
  const o = clientOrigin(id)
  if (!w || !o) return false
  return sx >= o.x && sx < o.x + w.innerWidth && sy >= o.y && sy < o.y + w.innerHeight
}

/** Watches a window's pointer (for its screen origin), focus and frame. */
export function trackWindow(id: string, w: Window): () => void {
  const move = (e: PointerEvent) => origins.set(id, { x: e.screenX - e.clientX, y: e.screenY - e.clientY })
  const front = () => {
    markFront(id)
    frames.set(id, frameOf(w))
  }
  const resize = () => {
    frames.set(id, frameOf(w))
    origins.delete(id)
  }
  w.addEventListener('pointermove', move, { capture: true, passive: true })
  w.addEventListener('pointerdown', move, { capture: true, passive: true })
  w.addEventListener('pointerdown', front, { capture: true, passive: true })
  w.addEventListener('focus', front)
  w.addEventListener('resize', resize)
  frames.set(id, frameOf(w))
  if (w.document.hasFocus()) front()
  return () => {
    w.removeEventListener('pointermove', move, { capture: true })
    w.removeEventListener('pointerdown', move, { capture: true })
    w.removeEventListener('pointerdown', front, { capture: true })
    w.removeEventListener('focus', front)
    w.removeEventListener('resize', resize)
    frames.delete(id)
    origins.delete(id)
  }
}

/** A window's frame now (top left, CSS pixels). */
export function frameOf(w: Window): Frame {
  return [w.screenX, w.screenY, w.outerWidth, w.outerHeight]
}

/** A frame moved onto the screen. */
export function onScreen(f: Frame): Frame {
  const s = window.screen as Screen & { availLeft?: number; availTop?: number }
  const left = s.availLeft ?? 0, top = s.availTop ?? 0
  const w = Math.min(f[2], s.availWidth || f[2]), h = Math.min(f[3], s.availHeight || f[3])
  const x = Math.min(Math.max(f[0], left), left + (s.availWidth || w) - w)
  const y = Math.min(Math.max(f[1], top), top + (s.availHeight || h) - h)
  return [x, y, w, h]
}

/** Frames are saved as the macOS app does: AppKit screen coordinates, the
 *  origin at the bottom left of the main screen. */
export const toSavedFrame = (f: Frame): Frame => [f[0], (window.screen?.height || 0) - (f[1] + f[3]), f[2], f[3]]
export const fromSavedFrame = (f: Frame): Frame => [f[0], (window.screen?.height || 0) - (f[1] + f[3]), f[2], f[3]]

// MARK: Opening and closing

/** Opens (or finds) the popup of a separate window. Null when the browser blocked it. */
export function openPopup(id: string, frame: Frame | null, title: string): Popup | null {
  const existing = usePopups.getState().open[id]
  if (existing && !existing.win.closed) return existing
  const f = onScreen(frame ?? [window.screenX + 80, window.screenY + 80, 640, 480])
  // width/height are the content's; the frame is restored exactly once the window exists.
  const features = `popup=yes,left=${Math.round(f[0])},top=${Math.round(f[1])},width=${Math.round(f[2])},height=${Math.round(Math.max(f[3] - 28, 200))}`
  let win: Window | null = null
  try {
    win = window.open('', `mklight-${id}`, features)
  } catch {
    win = null
  }
  if (!win) return null
  const doc = win.document
  doc.open()
  doc.write('<!doctype html><html lang="en"><head><meta charset="utf-8"></head><body></body></html>')
  doc.close()
  doc.title = title
  // The app's styles (Vite's <style> tags in development, <link>s in a build).
  for (const el of document.head.querySelectorAll('style, link[rel="stylesheet"], meta[name="color-scheme"]')) {
    doc.head.appendChild(doc.importNode(el, true))
  }
  doc.documentElement.className = document.documentElement.className
  doc.documentElement.style.cssText = 'height: 100%'
  doc.body.className = document.body.className
  doc.body.style.cssText = 'margin: 0; height: 100%; overflow: hidden'
  const mount = doc.createElement('div')
  mount.style.cssText = 'height: 100%; width: 100%'
  doc.body.appendChild(mount)
  try {
    win.moveTo(f[0], f[1])
    win.resizeTo(f[2], f[3])
  } catch {
    // Some browsers refuse; the window keeps the size it opened with.
  }

  const popup: Popup = { id, win, mount }
  const untrack = trackWindow(id, win)
  let last = frameOf(win)
  const check = () => {
    if (win!.closed) {
      gone()
      return
    }
    const now = frameOf(win!)
    if (now.some((v, k) => v !== last[k])) {
      // Moved or resized: the measured client origin is stale until the pointer comes back.
      if (now[0] !== last[0] || now[1] !== last[1]) origins.delete(id)
      last = now
      frames.set(id, now)
      onFrameChange(id, now)
    }
  }
  const timer = window.setInterval(check, 700)
  let done = false
  const gone = () => {
    if (done) return
    done = true
    window.clearInterval(timer)
    untrack()
    const open = { ...usePopups.getState().open }
    if (open[id]?.win === win) {
      delete open[id]
      usePopups.setState({ open })
    }
    if (!programmatic.delete(id) && !tearingDown) onUserClose(id)
  }
  win.addEventListener('pagehide', () => setTimeout(() => win!.closed && gone(), 0))
  win.addEventListener('resize', check)
  usePopups.setState({ open: { ...usePopups.getState().open, [id]: popup } })
  return popup
}

/** Closes a popup without closing its panels (they moved, or the main window closes). */
export function closePopup(id: string) {
  const p = usePopups.getState().open[id]
  if (!p) return
  programmatic.add(id)
  const open = { ...usePopups.getState().open }
  delete open[id]
  usePopups.setState({ open })
  try {
    p.win.close()
  } catch {
    // Already gone.
  }
}

export function focusWindow(id: string) {
  try {
    windowOf(id)?.focus()
  } catch {
    // Browsers may refuse to raise a window.
  }
}

/** The main window closes: the separate windows close with it; they stay in
 *  the saved arrangement and open again with the project. */
export function installTeardown() {
  trackWindow(MAIN, window)
  window.addEventListener('pagehide', () => {
    tearingDown = true
    for (const id of Object.keys(usePopups.getState().open)) closePopup(id)
  })
}
