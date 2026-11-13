/// <reference types="vitest/config" />
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { crossOriginIsolation } from './engine/vite-isolation'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))

// The real engine module lives in ./engine; its wasm package (engine/pkg) is
// built by scripts/build-wasm.sh. Without the package `@engine` points at a
// stub that reports it as missing, and the app runs on the mock engine
// (src/engine-mock). VITE_ENGINE=mock also leaves the wasm out of the build.
const realEngine = here('./engine/src/index.ts')
const hasWasm = existsSync(here('./engine/pkg/mklight_wasm.js'))
const engineEntry = hasWasm && process.env.VITE_ENGINE !== 'mock' ? realEngine : here('./src/engine/missing.ts')

// The isolation service worker (engine/coi-serviceworker.js): served at the
// app's base path in development and emitted next to index.html in the build,
// so static hosts without COOP/COEP headers still get a cross-origin isolated page.
const coiPath = here('./engine/coi-serviceworker.js')
function isolationWorker(): Plugin {
  return {
    name: 'mklight-isolation-worker',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.split('?')[0].endsWith('/coi-serviceworker.js') || !existsSync(coiPath)) return next()
        res.setHeader('Content-Type', 'text/javascript')
        res.end(readFileSync(coiPath))
      })
    },
    generateBundle() {
      if (existsSync(coiPath)) this.emitFile({ type: 'asset', fileName: 'coi-serviceworker.js', source: readFileSync(coiPath) })
    },
  }
}

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), isolationWorker(), crossOriginIsolation()],
  worker: { format: 'es' },
  resolve: {
    alias: {
      '@engine': engineEntry,
      '@': here('./src'),
      '@samples': here('../macos/Sources/MKLightModel/Resources/samples.json'),
    },
  },
  // Cross-origin isolation (COOP/COEP on every response, 304s included), so
  // the threaded WASM engine can use SharedArrayBuffer: crossOriginIsolation().
  server: {
    fs: { allow: [here('../..')] },
    watch: { ignored: ['**/engine/pkg/**'] },
  },
  build: { target: 'es2023', chunkSizeWarningLimit: 2000 },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    setupFiles: ['src/test-setup.ts'],
  },
})
