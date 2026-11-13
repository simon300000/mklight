// The engine's own test harness (harness/), not the web app. It shows the
// Vite settings the app needs too: ES-module workers, and the isolation
// headers on the dev server (docs/WEB.md).
import { defineConfig } from 'vite'

const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig({
  root: 'harness',
  base: './',
  worker: { format: 'es' },
  server: { headers: isolation },
  preview: { headers: isolation },
  build: { outDir: '../dist', emptyOutDir: true, target: 'es2022' },
})
