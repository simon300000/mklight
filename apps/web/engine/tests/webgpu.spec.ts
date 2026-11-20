// The WebGPU backend in the browser (docs/GPU.md §11): the GPU worker opens
// the adapter, the engine registers `gpu:webgpu`, and its analyses run on the
// device within the GPU accuracy contracts of the in-browser CPU reference.
// A browser without a WebGPU adapter skips with the reason (`ready().webgpu`).
// The timing test prints CPU (all threads) against WebGPU per analysis.

import { expect, type Page, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { REF_DIR, ISOLATED_URL } from './reference'

async function openHarness(page: Page) {
  await page.goto(ISOLATED_URL)
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
  const created = await page.evaluate(() => window.mk.create(undefined, 'off'))
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

test('WebGPU: auto routes by the prior break-even before calibration (GPU.md §11.6)', async ({ page }) => {
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  // The prior (no calibration).
  await page.evaluate(() => window.mk.create(undefined, 'off'))
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

test('WebGPU: calibration is measured, cached, reused, and routes auto', async ({ page }) => {
  test.setTimeout(240_000)
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  // A new engine on an empty IndexedDB (a fresh browser context) measures.
  const first = await page.evaluate(() => window.mk.create())
  const settle = (i: number) =>
    page.evaluate(async (k) => {
      for (let n = 0; n < 1200; n++) {
        const s = (await window.mk.calibration(k)) as { state: string }
        if (s.state !== 'running' && s.state !== 'idle') return s
        await new Promise((res) => setTimeout(res, 100))
      }
      return window.mk.calibration(k)
    }, i)
  const measured = (await settle(first.index)) as { state: string; calibration: Record<string, unknown> }
  console.log('calibration', JSON.stringify(measured))
  expect(measured.state).toBe('ready')
  expect(measured.calibration.source).toBe('measured')
  expect(Object.keys(measured.calibration.analyses as object).sort()).toEqual(['geometric_mtf', 'ray_grid', 'spot_diagram'])
  expect(measured.calibration.seconds as number).toBeLessThan(30)
  // The next engine reuses it from IndexedDB (the worker stores it once ready).
  await page.waitForTimeout(500)
  const second = await page.evaluate(() => window.mk.create())
  const reused = (await settle(second.index)) as { state: string; calibration: Record<string, unknown> }
  expect(reused.calibration.source).toBe('cached')
  expect(reused.calibration.analyses).toEqual(measured.calibration.analyses)

  // Forced estimates change the route; the decision records its inputs.
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t, i]) => window.mk.openProject(i as number, t as string), [text, second.index] as const)
  const auto = async (rings: number) => {
    const { result } = await page.evaluate(
      ([i, n]) => window.mk.run(i as number, { analysis: 'ray_grid', settings: { rings: n } as never, backend: 'auto', fresh: true }),
      [second.index, rings] as const,
    )
    return result.backend as { used: string; cost_decision: Record<string, unknown> }
  }
  const forced = (gpu: { fixed_ms: number; per_mwork_ms: number }) => {
    const c = structuredClone(measured.calibration) as { analyses: Record<string, { cpu: object; webgpu: object }>; source: string }
    for (const a of Object.values(c.analyses)) {
      a.cpu = { fixed_ms: 50, per_mwork_ms: 500 }
      a.webgpu = gpu
    }
    c.source = 'forced'
    return c
  }
  await page.evaluate(([i, c]) => window.mk.setCalibration(i as number, c as never), [second.index, forced({ fixed_ms: 0, per_mwork_ms: 1 })] as const)
  const fast = await auto(8)
  expect(fast.used).toBe('gpu:webgpu')
  expect(fast.cost_decision.rule).toBe('webgpu calibration')
  expect((fast.cost_decision.calibration as { source: string }).source).toBe('forced')
  await page.evaluate(([i, c]) => window.mk.setCalibration(i as number, c as never), [second.index, forced({ fixed_ms: 1e6, per_mwork_ms: 1 })] as const)
  const slow = await auto(250)
  expect(slow.used).toBe('cpu_reference')
  expect(slow.cost_decision.rule).toBe('webgpu calibration')
  // Back to the prior.
  await page.evaluate((i) => window.mk.setCalibration(i, null), second.index)
  expect((await auto(8)).cost_decision.rule).toBe('webgpu break-even')
})

test('WebGPU: a pipeline that never compiles is a reason, not a hang, and is remembered', async ({ page }) => {
  test.setTimeout(240_000)
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  // The trace pipelines never finish compiling (the watchdog's test stall),
  // with a 1.5 s limit. A fresh browser context: nothing remembered yet.
  const watchdog = { timeoutMs: 1500, stalledModules: ['trace', 'trace_overlays'] }
  const first = await page.evaluate((w) => window.mk.create(undefined, 'auto', w), watchdog)
  const webgpu = first.info.backends.find((b) => b.id === 'gpu:webgpu')
  expect(webgpu?.available, `gpu:webgpu opens: ${webgpu?.reason}`).toBe(true)
  const REASON = 'compute pipeline did not finish compiling within 1.5 s, so WebGPU is not used on this device'

  // The calibration meets the stall: it fails with the reason, records no
  // numbers, and does not hang.
  const settled = (await page.evaluate(async (k) => {
    for (let n = 0; n < 1200; n++) {
      const s = (await window.mk.calibration(k)) as { state: string }
      if (s.state !== 'running' && s.state !== 'idle') return s
      await new Promise((res) => setTimeout(res, 100))
    }
    return window.mk.calibration(k)
  }, first.index)) as { state: string; message?: string; calibration?: unknown }
  console.log('calibration', JSON.stringify(settled))
  expect(settled.state).toBe('failed')
  expect(settled.message).toContain(REASON)
  expect(settled.calibration).toBeUndefined()

  // `auto` sends a large ray grid (the prior's WebGPU route) to the CPU
  // with the reason; an explicit request without fallback fails with it;
  // with fallback (the default) it runs on the CPU and says why.
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t, i]) => window.mk.openProject(i as number, t as string), [text, first.index] as const)
  const run = (backend: unknown, fallback = true) =>
    page.evaluate(
      async ([i, b, f]) => {
        const req = {
          analysis: 'ray_grid',
          settings: { rings: 96 } as never,
          backend: b as never,
          allow_backend_fallback: f as boolean,
          fresh: true,
        }
        try {
          const { result } = await window.mk.run(i as number, req)
          return { backend: result.backend as { used: string; fallback_reason?: string }, error: null }
        } catch (e) {
          const x = e as { kind?: string; message?: string }
          return { backend: null, error: { kind: x.kind, message: String(x.message) } }
        }
      },
      [first.index, backend, fallback] as const,
    )
  const auto = await run('auto')
  expect(auto.backend?.used).toBe('cpu_reference')
  expect(auto.backend?.fallback_reason).toContain('gpu:webgpu: the trace::')
  expect(auto.backend?.fallback_reason).toContain(REASON)
  const strict = await run('gpu', false)
  expect(strict.error?.kind).toBe('unsupported')
  expect(strict.error?.message).toContain(REASON)
  const lenient = await run('gpu')
  expect(lenient.backend?.used).toBe('cpu_reference')
  expect(lenient.backend?.fallback_reason).toContain(REASON)

  // A later load on this browser and adapter, without the stall, does not
  // wait again: gpu:webgpu is unavailable at once with the remembered
  // reason, and no calibration starts.
  const second = await page.evaluate(() => window.mk.create())
  const again = second.info.backends.find((b) => b.id === 'gpu:webgpu')
  console.log('remembered', JSON.stringify(again))
  expect(again?.available).toBe(false)
  expect(again?.reason).toContain(REASON)
  expect(again?.reason).toContain('remembered for this browser and GPU')
  expect(((await page.evaluate((i) => window.mk.calibration(i), second.index)) as { state: string }).state).toBe('idle')
  await page.evaluate(([t, i]) => window.mk.openProject(i as number, t as string), [text, second.index] as const)
  const t0 = Date.now()
  const { result } = await page.evaluate(
    (i) => window.mk.run(i, { analysis: 'ray_grid', settings: { rings: 96 } as never, backend: 'auto', fresh: true }),
    second.index,
  )
  expect(Date.now() - t0).toBeLessThan(60_000)
  const used = result.backend as { used: string; fallback_reason: string }
  expect(used.used).toBe('cpu_reference')
  expect(used.fallback_reason).toContain('remembered for this browser and GPU')

  // A reload (the uncancellable compilation may still occupy the browser's
  // GPU process) makes no WebGPU call at all: not the page (`ready()`), not
  // the engine worker, and no GPU worker starts. The engine is ready at once.
  await countAdapterRequests(page)
  const workers = watchWorkers(page)
  await page.reload()
  await page.waitForFunction(() => globalThis.crossOriginIsolated && 'mk' in window, null, { timeout: 30_000 })
  const t1 = Date.now()
  const reloaded = await page.evaluate(() => window.mk.create(undefined, 'auto', { countCalls: true }))
  const startMs = Date.now() - t1
  const readiness = await page.evaluate(() => window.mk.ready())
  console.log('after reload', startMs, 'ms', JSON.stringify(reloaded.info.webgpuCalls), JSON.stringify(readiness))
  expect(startMs).toBeLessThan(10_000)
  expect(reloaded.info.webgpu).toBe(false)
  expect(reloaded.info.webgpuReason).toContain(REASON)
  expect(reloaded.info.webgpuCalls).toEqual({ requestAdapter: 0, requestDevice: 0 })
  const off = reloaded.info.backends.find((b) => b.id === 'gpu:webgpu')
  expect(off?.available).toBe(false)
  expect(off?.reason).toContain('remembered for this browser and GPU')
  expect(readiness.webgpu).toBe(false)
  expect(readiness.webgpuReason).toContain(REASON)
  expect(await page.evaluate(() => (window as unknown as { __adapterRequests: number }).__adapterRequests)).toBe(0)
  expect(workers.filter((u) => /gpu-host/.test(u)), 'no GPU worker').toEqual([])
  expect(workers.some((u) => /worker/.test(u)), `engine worker seen: ${workers}`).toBe(true)
})

/** Count the page's `requestAdapter` calls (`window.__adapterRequests`)
 * from the next load on; `stall` makes them never answer. */
async function countAdapterRequests(page: Page, stall = false) {
  await page.addInitScript((s) => {
    const w = window as unknown as { __adapterRequests: number; GPU?: { prototype: { requestAdapter: () => Promise<unknown> } } }
    w.__adapterRequests = 0
    const proto = w.GPU?.prototype
    if (!proto) return
    const f = proto.requestAdapter
    proto.requestAdapter = function (this: unknown) {
      w.__adapterRequests++
      return s ? new Promise(() => {}) : f.call(this)
    }
  }, stall)
}

/** URLs of the workers the page starts from now on (nested ones too). */
function watchWorkers(page: Page): string[] {
  const urls: string[] = []
  page.on('worker', (w) => urls.push(w.url()))
  return urls
}

test('WebGPU: an adapter request that never answers is a reason, not a hang', async ({ page, browserName }) => {
  test.setTimeout(120_000)
  await openHarness(page)
  const r = await page.evaluate(() => window.mk.ready())
  test.skip(!r.webgpu, 'no WebGPU adapter')
  // The page: `ready()` gives up after a few seconds with the reason.
  await countAdapterRequests(page, true)
  await page.reload()
  await page.waitForFunction(() => globalThis.crossOriginIsolated && 'mk' in window, null, { timeout: 30_000 })
  const t0 = Date.now()
  const stalled = await page.evaluate(() => window.mk.ready())
  expect(Date.now() - t0).toBeLessThan(8_000)
  expect(stalled.webgpu).toBe(false)
  expect(stalled.webgpuReason).toContain('did not answer the WebGPU adapter request')
  expect(await page.evaluate(() => (window as unknown as { __adapterRequests: number }).__adapterRequests)).toBe(1)

  // The engine worker: its adapter request never answers either; the engine
  // starts without WebGPU after the limit, with the reason, and no GPU worker.
  const workers = watchWorkers(page)
  const t1 = Date.now()
  const e = await page.evaluate(() =>
    window.mk.create(undefined, 'auto', { stallAdapter: true, adapterTimeoutMs: 1000, countCalls: true }),
  )
  const startMs = Date.now() - t1
  console.log('stalled adapter', startMs, 'ms', JSON.stringify(e.info.webgpuCalls), e.info.webgpuReason)
  expect(startMs).toBeLessThan(10_000)
  expect(e.info.webgpu).toBe(false)
  expect(e.info.webgpuReason).toContain('did not answer the WebGPU adapter request within 1 s')
  expect(e.info.webgpuCalls).toEqual({ requestAdapter: 1, requestDevice: 0 })
  const off = e.info.backends.find((b) => b.id === 'gpu:webgpu')
  expect(off?.available).toBe(false)
  expect(off?.reason).toContain('did not answer the WebGPU adapter request')
  expect(workers.filter((u) => /gpu-host/.test(u)), 'no GPU worker').toEqual([])
  // Analyses run (on the CPU).
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t, i]) => window.mk.openProject(i as number, t as string), [text, e.index] as const)
  const { result } = await page.evaluate(
    (i) => window.mk.run(i, { analysis: 'ray_grid', settings: { rings: 96 } as never, backend: 'auto', fresh: true }),
    e.index,
  )
  expect((result.backend as { used: string }).used).toBe('cpu_reference')

  // Only this load: nothing is remembered, the next engine uses WebGPU.
  const next = await page.evaluate(() => window.mk.create(undefined, 'off'))
  expect(next.info.webgpu, `${next.info.webgpuReason}`).toBe(true)
  expect(next.info.backends.find((b) => b.id === 'gpu:webgpu')?.available).toBe(true)
  // (Chromium reports nested workers, so there the worker watch sees GPU
  // workers and the checks above are not vacuous; WebKit reports only the
  // page's own workers.)
  if (browserName === 'chromium') expect(workers.filter((u) => /gpu-host/.test(u)).length).toBe(1)
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
      // No background calibration competing with the timed runs.
      const e = (await window.mk.create(undefined, 'off')).index
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
