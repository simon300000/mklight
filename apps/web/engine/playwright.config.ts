// Engine tests in headless Chromium and WebKit (docs/WEB.md "Tests").
// The harness is built with Vite and served three times: without isolation
// headers (port MKLIGHT_WEB_PORT, default 4710, the static-host case: the
// service worker isolates the page), with them (the next port), and with
// them on 200s only and bare 304 revalidations (the port after), on which
// WebKit refuses some worker starts.
import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.MKLIGHT_WEB_PORT ?? 4710)

export default defineConfig({
  testDir: 'tests',
  timeout: 300_000,
  workers: 1,
  globalSetup: './tests/reference.ts',
  reporter: [['list']],
  use: { baseURL: `http://localhost:${PORT}/` },
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
      command: `npx vite build && node tests/static-server.mjs dist ${PORT}`,
      url: `http://localhost:${PORT}/`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `node tests/static-server.mjs dist ${PORT + 1} --isolate`,
      url: `http://localhost:${PORT + 1}/`,
      reuseExistingServer: false,
    },
    {
      command: `node tests/static-server.mjs dist ${PORT + 2} --isolate-200`,
      url: `http://localhost:${PORT + 2}/`,
      reuseExistingServer: false,
    },
  ],
})
