// Cross-origin isolation on any static host (docs/WEB.md "Isolation"):
// registers `coi-serviceworker.js`, which adds COOP/COEP/CORP headers to
// every same-origin response, and reloads the page once so it is served
// through it. Hosts that send the headers themselves need none of this
// (`ensureIsolation` then returns at once).
//
// The service worker file must be served from the app's base path (copy
// `apps/web/engine/coi-serviceworker.js` into the app's `public/`), since a
// service worker only controls pages at or below its own path. It is also a
// plain script: `<script src="coi-serviceworker.js"></script>` first in the
// page's <head> does the same as this function, before any app code loads.

const RELOAD_KEY = 'mklight-coi-reload'

/** Make the page cross-origin isolated, reloading once if needed. Resolves
 * `true` when isolated; `false` when it cannot be (no service workers,
 * not a secure context, or the reload did not help). Never resolves while a
 * reload is under way. */
export async function ensureIsolation(swUrl = 'coi-serviceworker.js'): Promise<boolean> {
  if (globalThis.crossOriginIsolated) {
    sessionStorage.removeItem(RELOAD_KEY)
    return true
  }
  if (!globalThis.isSecureContext || !('serviceWorker' in navigator)) return false
  // Reloaded once already and still not isolated: give up (and say why in
  // `ready()`), never loop.
  if (sessionStorage.getItem(RELOAD_KEY)) return false
  const url = new URL(swUrl, document.baseURI)
  const reg = await navigator.serviceWorker.register(url, { scope: new URL('./', url).pathname })
  const reload = () => {
    sessionStorage.setItem(RELOAD_KEY, '1')
    location.reload()
  }
  if (navigator.serviceWorker.controller) {
    // Controlled but not isolated: an older worker; reload through it.
    reload()
  } else {
    const sw = reg.installing ?? reg.waiting ?? reg.active
    if (sw?.state === 'activated') reload()
    else sw?.addEventListener('statechange', () => sw.state === 'activated' && reload())
  }
  return new Promise<boolean>(() => {})
}
