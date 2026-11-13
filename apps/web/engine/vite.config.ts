// The engine's own test harness (harness/), not the web app. It shows the
// Vite settings the app needs too: ES-module workers, and the isolation
// headers on the dev and preview servers, 304s included (docs/WEB.md).
import { defineConfig } from 'vite'
import { crossOriginIsolation } from './vite-isolation'

export default defineConfig({
  root: 'harness',
  base: './',
  plugins: [crossOriginIsolation()],
  worker: { format: 'es' },
  build: { outDir: '../dist', emptyOutDir: true, target: 'es2022' },
})
