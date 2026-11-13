import { useSyncExternalStore } from 'react'

const query = () => (typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null)

function subscribe(cb: () => void) {
  const q = query()
  q?.addEventListener('change', cb)
  return () => q?.removeEventListener('change', cb)
}

/** True while the system appearance is dark. */
export function useDark(): boolean {
  return useSyncExternalStore(subscribe, () => query()?.matches ?? false, () => false)
}

/** Resolved value of a CSS custom property (`--label`), for canvas drawing. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/** Validated categorical palette for plot series (fields), light and dark. */
export const SERIES_LIGHT = ['#0a74ff', '#e8590c', '#2b8a3e', '#c2255c', '#7048e8', '#0c8599', '#a8730f', '#5c5f66']
export const SERIES_DARK = ['#4d9dff', '#ff8c42', '#51cf66', '#f06595', '#9775fa', '#3bc9db', '#fcc419', '#adb5bd']
export const seriesColor = (k: number, dark: boolean) => (dark ? SERIES_DARK : SERIES_LIGHT)[k % 8]

/** The colour of light of a wavelength (µm), for wavelength series. */
export function wavelengthColor(um: number, dark = false): string {
  const nm = um * 1000
  let r = 0, g = 0, b = 0
  if (nm < 380) [r, g, b] = [0.45, 0, 0.8]
  else if (nm < 440) [r, g, b] = [-(nm - 440) / 60, 0, 1]
  else if (nm < 490) [r, g, b] = [0, (nm - 440) / 50, 1]
  else if (nm < 510) [r, g, b] = [0, 1, -(nm - 510) / 20]
  else if (nm < 580) [r, g, b] = [(nm - 510) / 70, 1, 0]
  else if (nm < 645) [r, g, b] = [1, -(nm - 645) / 65, 0]
  else if (nm <= 780) [r, g, b] = [1, 0, 0]
  else [r, g, b] = [0.55, 0.1, 0.1]
  // Darken greens and yellows on light backgrounds so lines stay readable.
  const k = dark ? 1 : 0.78
  const c = (v: number) => Math.round(255 * Math.min(1, Math.max(0, v)) * k)
  return `rgb(${c(r)}, ${c(g)}, ${c(b)})`
}
