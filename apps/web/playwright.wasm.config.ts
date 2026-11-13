import { defineConfig, devices } from '@playwright/test'

// The app on the real WASM engine (build it first: scripts/build-wasm.sh),
// in Chromium and WebKit, served by `vite preview` with COOP/COEP.
export default defineConfig({
  testDir: 'e2e-wasm',
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
    command: 'pnpm exec vite build --outDir dist-wasm && pnpm exec vite preview --outDir dist-wasm --port 4318 --strictPort',
    url: 'http://localhost:4318/',
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
    env: { VITE_ENGINE: 'wasm' },
  },
})
