// Glide's overlay editor follows the window the lens data is in.
//
// Glide Data Grid opens its cell editor through a React portal into
// `document.getElementById('portal')` of the global document, which is the
// main window's. When the lens data sits in a separate window (a same-origin
// popup rendered from the main window's React tree, see panels/windows.ts),
// the editor must open in that window's document instead. While a lens data
// table is mounted in a popup, the popup gets its own `#portal` element and
// the main document's lookup of `portal` answers with it; Glide's outside-click
// listener also listens only on the main document, so a press elsewhere in the
// popup is forwarded to it.
import { useEffect } from 'react'

let override: HTMLElement | null = null
let installed = false

function install() {
  if (installed || typeof document === 'undefined') return
  installed = true
  const original = document.getElementById.bind(document)
  document.getElementById = (id: string) => (id === 'portal' && override?.isConnected ? override : original(id))
}

/** `container`: the portal container of the window the table is in (null in the main window). */
export function useGlidePortal(container: HTMLElement | null) {
  useEffect(() => {
    if (!container) return
    install()
    const doc = container.ownerDocument
    const el = doc.createElement('div')
    el.id = 'portal'
    el.style.cssText = 'position: fixed; left: 0; top: 0; z-index: 9999'
    container.appendChild(el)
    override = el
    const forward = (e: Event) => {
      const t = e.target
      if (t instanceof (doc.defaultView?.Element ?? Element) && (el.contains(t) || (t as Element).closest('[data-lens-completions]'))) return
      if (!el.firstChild) return
      document.body.dispatchEvent(new MouseEvent(e.type, { bubbles: true, cancelable: true }))
    }
    const kinds = ['mousedown', 'touchend', 'contextmenu'] as const
    for (const k of kinds) doc.addEventListener(k, forward, true)
    return () => {
      for (const k of kinds) doc.removeEventListener(k, forward, true)
      if (override === el) override = null
      el.remove()
    }
  }, [container])
}
