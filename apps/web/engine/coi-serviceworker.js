/* MKLight cross-origin isolation service worker (docs/WEB.md "Isolation").
 *
 * SharedArrayBuffer, and with it the multithreaded engine, needs a
 * cross-origin isolated page: the document must be served with
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 * Static hosts such as GitHub Pages cannot set headers, so this service
 * worker adds them to every same-origin response (and
 * Cross-Origin-Resource-Policy, so the page's own subresources load under
 * COEP). Cross-origin resources must send CORP/CORS themselves.
 *
 * One file, two roles:
 *  - as a page script (`<script src="coi-serviceworker.js"></script>`,
 *    first in <head>): registers itself and reloads the page once, so the
 *    page is served through the worker;
 *  - as the service worker: rewrites the headers.
 * Serve it from the app's base path (a worker controls only pages at or
 * below its own path).
 */
'use strict'

if (typeof window === 'undefined') {
  // ---- Service worker
  self.addEventListener('install', () => self.skipWaiting())
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()))

  self.addEventListener('fetch', (event) => {
    const req = event.request
    // A request the browser may only answer from cache (devtools) would
    // throw when fetched.
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return
    event.respondWith(
      fetch(req).then((res) => {
        // Opaque responses cannot be rewritten (and are not ours).
        if (res.status === 0) return res
        const headers = new Headers(res.headers)
        headers.set('Cross-Origin-Opener-Policy', 'same-origin')
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
        if (!headers.has('Cross-Origin-Resource-Policy')) {
          headers.set('Cross-Origin-Resource-Policy', 'same-origin')
        }
        return new Response(res.body, { status: res.status, statusText: res.statusText, headers })
      }),
    )
  })
} else {
  // ---- Page script
  ;(() => {
    const KEY = 'mklight-coi-reload'
    if (window.crossOriginIsolated) {
      sessionStorage.removeItem(KEY)
      return
    }
    if (!window.isSecureContext || !('serviceWorker' in navigator)) return
    // Reloaded once already and still not isolated: stop (the app's
    // `ready()` explains why), never loop.
    if (sessionStorage.getItem(KEY)) return
    const reload = () => {
      sessionStorage.setItem(KEY, '1')
      location.reload()
    }
    const src = document.currentScript && document.currentScript.src
    if (!src) return
    navigator.serviceWorker.register(src).then(
      (reg) => {
        if (navigator.serviceWorker.controller) return reload()
        const sw = reg.installing || reg.waiting || reg.active
        if (!sw) return
        if (sw.state === 'activated') return reload()
        sw.addEventListener('statechange', () => sw.state === 'activated' && reload())
      },
      (err) => console.warn('MKLight: cross-origin isolation service worker failed to register:', err),
    )
  })()
}
