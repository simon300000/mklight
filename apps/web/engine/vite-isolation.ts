// Cross-origin isolation headers for the Vite dev and preview servers
// (docs/WEB.md "Isolation"). Use this instead of `server.headers` /
// `preview.headers`: those are not sent on `304 Not Modified`
// revalidations, and WebKit checks a worker script's COEP on the 304's own
// headers, so it then refuses to start the worker ("Worker load was blocked
// by Cross-Origin-Embedder-Policy"). This middleware sets the headers before
// any response is written, 304s included.
import type { Connect, Plugin } from 'vite'

export const ISOLATION_HEADERS: Record<string, string> = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
}

const setHeaders: Connect.NextHandleFunction = (_req, res, next) => {
  for (const [k, v] of Object.entries(ISOLATION_HEADERS)) res.setHeader(k, v)
  next()
}

export function crossOriginIsolation(): Plugin {
  return {
    name: 'mklight-cross-origin-isolation',
    configureServer(server) {
      server.middlewares.use(setHeaders)
    },
    configurePreviewServer(server) {
      server.middlewares.use(setHeaders)
    },
  }
}
