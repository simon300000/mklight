/// <reference types="vitest/config" />
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url))

// The real engine module (owned by the wasm-engine work) lives in ./engine.
// Until it exists, `@engine` points at a stub that reports it as missing, and
// the app runs on the mock engine (src/engine-mock).
const realEngine = here('./engine/src/index.ts')
const engineEntry = existsSync(realEngine) ? realEngine : here('./src/engine/missing.ts')

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@engine': engineEntry,
      '@': here('./src'),
      '@samples': here('../macos/Sources/MKLightModel/Resources/samples.json'),
    },
  },
  server: {
    // Cross-origin isolation, so a threaded WASM engine can use SharedArrayBuffer.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: { allow: [here('../..')] },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: { target: 'es2023', chunkSizeWarningLimit: 2000 },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    setupFiles: ['src/test-setup.ts'],
  },
})
