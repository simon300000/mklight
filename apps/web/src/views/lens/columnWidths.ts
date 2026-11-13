// The lens data's column widths the user dragged, kept for the browser
// session (sessionStorage, like the macOS table's widths for the window's
// life). Columns without an entry keep their default width, which the
// Playwright tests rely on.
import { useCallback, useState } from 'react'

const KEY = 'mklight.lensData.columnWidths'

export type Widths = Record<string, number>

export function loadWidths(storage: Pick<Storage, 'getItem'> | null = safeSession()): Widths {
  try {
    const raw = storage?.getItem(KEY)
    const v = raw ? (JSON.parse(raw) as unknown) : null
    if (!v || typeof v !== 'object' || Array.isArray(v)) return {}
    const out: Widths = {}
    for (const [k, w] of Object.entries(v)) if (typeof w === 'number' && Number.isFinite(w) && w >= 20 && w <= 1000) out[k] = Math.round(w)
    return out
  } catch {
    return {}
  }
}

function saveWidths(w: Widths) {
  try {
    const s = safeSession()
    if (!s) return
    if (Object.keys(w).length) s.setItem(KEY, JSON.stringify(w))
    else s.removeItem(KEY)
  } catch {
    // Storage may be unavailable (private windows); the widths stay for this view.
  }
}

function safeSession(): Storage | null {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage
  } catch {
    return null
  }
}

/** The widths, a setter for one column and a reset to the defaults. */
export function useColumnWidths(): [Widths, (id: string, width: number) => void, () => void] {
  const [widths, setWidths] = useState<Widths>(() => loadWidths())
  const set = useCallback((id: string, width: number) => {
    setWidths((w) => {
      const next = { ...w, [id]: Math.round(width) }
      saveWidths(next)
      return next
    })
  }, [])
  const reset = useCallback(() => {
    saveWidths({})
    setWidths({})
  }, [])
  return [widths, set, reset]
}
