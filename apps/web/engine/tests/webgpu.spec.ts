// The WebGPU backend in the browser (docs/GPU.md §11): the GPU worker opens
// the adapter, the engine registers `gpu:webgpu`, and its analyses run on the
// device within the GPU accuracy contracts of the in-browser CPU reference.
// A browser without a WebGPU adapter skips with the reason (`ready().webgpu`).
// The timing test prints CPU (all threads) against WebGPU per analysis.

import { expect, type Page, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { REF_DIR } from './reference'

async function openHarness(page: Page) {
  await page.goto('http://localhost:4711/')
  await page.waitForFunction(() => globalThis.crossOriginIsolated && 'mk' in window, null, { timeout: 30_000 })
}

type Buffers = Record<string, { dtype: string; shape: number[]; data: (number | string)[] }>

/** Largest |a − b| over the numbers of two JSON values with the same shape,
 * by top-level key path prefix. */
function diffs(a: unknown, b: unknown, path: string, out: Map<string, number>) {
  if (typeof a === 'number' && typeof b === 'number') {
    const key = path.replace(/\[\d+\]/g, '[]')
    out.set(key, Math.max(out.get(key) ?? 0, Math.abs(a - b)))
    return
  }
  if (Array.isArray(a) && Array.isArray(b)) return a.forEach((v, i) => diffs(v, b[i], `${path}[${i}]`, out))
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const k of Object.keys(a)) diffs((a as never)[k], (b as never)[k], `${path}.${k}`, out)
  }
}

/** Relative L1 difference of two buffers (image contract, GPU.md §9.4). */
function l1(a: (number | string)[], b: (number | string)[]) {
  let d = 0
  let n = 0
  for (let i = 0; i < a.length; i++) {
    d += Math.abs(Number(a[i]) - Number(b[i]))
    n += Math.abs(Number(b[i]))
  }
  return d / Math.max(n, 1e-300)
}

/**
 * Per-analysis gates (GPU.md §9.4 classes, applied to this test's fields):
 * spot: positions/radii 2e-5 mm (+ RMS 0.5 %); MTF: modulation 2e-3; PSF:
 * image L1 2e-3.
 */
const CASES: Array<{ analysis: string; settings: object; check: (gpu: Run, cpu: Run) => void }> = [
  {
    analysis: 'spot_diagram',
    settings: { rings: 32 },
    check(g, c) {
      const d = new Map<string, number>()
      diffs(g.json, c.json, '', d)
      for (const [k, v] of d) {
        if (/centroid|radius|rms|chief/.test(k)) expect(v, k).toBeLessThanOrEqual(2e-5 + 0.005 * maxAbs(c.json, k))
      }
    },
  },
  {
    analysis: 'mtf',
    settings: {},
    check(g, c) {
      const d = new Map<string, number>()
      diffs(g.json, c.json, '', d)
      for (const [k, v] of d) if (/mtf|modulation|tangential|sagittal/.test(k)) expect(v, k).toBeLessThanOrEqual(2e-3)
    },
  },
  {
    analysis: 'psf',
    settings: {},
    check(g, c) {
      for (const [name, b] of Object.entries(c.buffers)) {
        if (b.dtype !== 'f64' && b.dtype !== 'f32') continue
        expect(l1(g.buffers[name].data, b.data), `buffer ${name}`).toBeLessThanOrEqual(2e-3)
      }
    },
  },
]

interface Run {
  json: unknown
  buffers: Buffers
  backend: { used?: string; fallback_reason?: string } | null
}

function maxAbs(v: unknown, key: string): number {
  const m = new Map<string, number>()
  diffs(v, JSON.parse(JSON.stringify(v, (_k, x) => (typeof x === 'number' ? 0 : x))), '', m)
  return m.get(key) ?? 0
}

test('WebGPU: gpu:webgpu runs analyses within the GPU contracts', async ({ page }) => {
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, `this browser has no WebGPU adapter (${JSON.stringify(r)})`)
  const created = await page.evaluate(() => window.mk.create())
  const webgpu = created.info.backends.find((b) => b.id === 'gpu:webgpu')
  console.log('gpu:webgpu', JSON.stringify(webgpu))
  expect(webgpu, 'gpu:webgpu registered').toBeTruthy()
  expect(webgpu!.available, `gpu:webgpu available: ${webgpu!.reason}`).toBe(true)
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t]) => window.mk.openProject(0, t), [text])
  for (const c of CASES) {
    const run = (backend: string) =>
      page.evaluate(
        ([a, s, b]) => window.mk.run(0, { analysis: a as string, settings: s as never, backend: b as never }),
        [c.analysis, c.settings, backend] as const,
      )
    const cpu = (await run('cpu_reference')).result as unknown as Run
    const gpu = (await run('gpu')).result as unknown as Run
    console.log(`${c.analysis}: used ${gpu.backend?.used}${gpu.backend?.fallback_reason ? ` (${gpu.backend.fallback_reason})` : ''}`)
    expect(gpu.backend?.used).toBe('gpu:webgpu')
    c.check(gpu, cpu)
  }
})

test('WebGPU: auto routes by the measured break-even (GPU.md §11.6)', async ({ page }) => {
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  await page.evaluate(() => window.mk.create())
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t]) => window.mk.openProject(0, t), [text])
  const used = async (analysis: string, settings: object) => {
    const { result } = await page.evaluate(
      ([a, s]) => window.mk.run(0, { analysis: a as string, settings: s as never, backend: 'auto' }),
      [analysis, settings] as const,
    )
    return (result.backend as { used: string }).used
  }
  // Cooke triplet: 3 fields × 3 wavelengths × 27 937 rays at 96 rings.
  expect(await used('ray_grid', { rings: 96 })).toBe('gpu:webgpu')
  expect(await used('ray_grid', { rings: 8 })).toBe('cpu_reference')
  expect(await used('mtf', {})).toBe('cpu_reference')
})

/**
 * CPU (all threads) against WebGPU, fresh runs after a warm-up, best of
 * three: the browser columns of GPU.md §11's benchmark table. Set
 * MKLIGHT_WEB_BENCH=1 to run.
 */
test('WebGPU: timing against the wasm CPU reference', async ({ page }) => {
  test.skip(process.env.MKLIGHT_WEB_BENCH !== '1', 'set MKLIGHT_WEB_BENCH=1')
  test.setTimeout(1_800_000)
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  const project = process.env.MKLIGHT_WEB_BENCH_PROJECT ?? join(REF_DIR, 'cooke.mklight')
  const text = readFileSync(project, 'utf8')
  const cases: Array<[string, object, string, object?]> = JSON.parse(
    process.env.MKLIGHT_WEB_BENCH_CASES ??
      JSON.stringify([
        ['spot_diagram', { rings: 6 }, 'r6'],
        ['spot_diagram', { rings: 32 }, 'r32'],
        ['spot_diagram', { rings: 128 }, 'r128'],
        ['geometric_mtf', { rings: 32 }, 'r32'],
        ['ray_grid', { rings: 32 }, 'r32'],
        ['psf', {}, 'default'],
        ['psf', { pupil: { grid: 256 } }, 'g256'],
        ['mtf', {}, 'default'],
        ['mtf_vs_field', {}, 'default'],
        ['through_focus_mtf', {}, 'default'],
      ]),
  )
  const rows = await page.evaluate(
    async ([t, cs]) => {
      const e = (await window.mk.create()).index
      await window.mk.openProject(e, t as string)
      const out: Array<{ tag: string; cpu: number; gpu: number; used: string }> = []
      // A case's optional fourth element: route options of its WebGPU run.
      for (const [a, s, tag, route] of cs as Array<[string, object, string, object?]>) {
        const req = (b: string) => ({
          analysis: a,
          settings: s as never,
          backend: b as never,
          ...(route && b === 'gpu' ? { route: route as never } : {}),
        })
        const used = ((await window.mk.run(e, req('gpu'))).result.backend as { used: string }).used
        await window.mk.time(e, req('cpu_reference'), 1)
        const cpu = Math.min(...(await window.mk.time(e, req('cpu_reference'), 3)))
        const gpu = Math.min(...(await window.mk.time(e, req('gpu'), 3)))
        out.push({ tag: `${a} ${tag}`, cpu, gpu, used })
      }
      return out
    },
    [text, cases] as const,
  )
  for (const x of rows) {
    console.log(
      `WEB_BENCH ${x.tag.padEnd(28)} cpu ${x.cpu.toFixed(1).padStart(8)} ms | webgpu ${x.gpu.toFixed(1).padStart(8)} ms ` +
        `(${x.used}) | ${(x.cpu / x.gpu).toFixed(2)}×`,
    )
  }
})
