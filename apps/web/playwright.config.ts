import { defineConfig, devices } from '@playwright/test'

// Smoke tests against the mock engine, in Chromium and WebKit, on the built app
// served by `vite preview` (cross-origin isolated, like a real deployment).
export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  fullyParallel: true,
  reporter: [['list']],
  use: { baseURL: 'http://localhost:4317/', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'webkit', use: { ...devices['Desktop Safari'], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: {
    command: 'pnpm build && pnpm exec vite preview --port 4317 --strictPort',
    url: 'http://localhost:4317/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { VITE_ENGINE: 'mock' },
  },
})
