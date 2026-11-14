import { defineConfig, devices } from '@playwright/test'

// The UI suite on the real WASM engine (build it first: npm run build:wasm), in
// Chromium and WebKit, on the built app served by `vite preview` with COOP/COEP
// (cross-origin isolated, like a real deployment). The engine runs a thread
// pool per page, so the tests run one at a time.
export default defineConfig({
  testDir: 'e2e',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: { baseURL: 'http://localhost:4318/', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1600, height: 960 } } },
    { name: 'webkit', use: { ...devices['Desktop Safari'], viewport: { width: 1600, height: 960 } } },
  ],
  webServer: {
    command: 'pnpm exec vite build --outDir dist-e2e && pnpm exec vite preview --outDir dist-e2e --port 4318 --strictPort',
    url: 'http://localhost:4318/',
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
  },
})
