// Engine tests in headless Chromium and WebKit (docs/WEB.md "Tests").
// The harness is built with Vite and served twice: without isolation
// headers (port 4710, the static-host case: the service worker isolates
// the page) and with them (port 4711).
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: 'tests',
  timeout: 300_000,
  workers: 1,
  globalSetup: './tests/reference.ts',
  reporter: [['list']],
  use: { baseURL: 'http://localhost:4710/' },
  projects: [
    // The full Chromium in new-headless mode (the headless shell has no
    // WebGPU adapter), with WebGPU enabled for the GPU tests.
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chromium',
        launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'] },
      },
    },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: [
    {
      command: 'npx vite build && node tests/static-server.mjs dist 4710',
      url: 'http://localhost:4710/',
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: 'node tests/static-server.mjs dist 4711 --isolate',
      url: 'http://localhost:4711/',
      reuseExistingServer: false,
    },
  ],
})
