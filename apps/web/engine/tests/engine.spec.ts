// The web engine in a real browser (Chromium and WebKit): isolation on a
// header-less static host, loading, a Cooke triplet, analyses compared with
// the native engine, errors, projects, import, cancellation and the rayon
// speed-up. docs/WEB.md "Tests".

import { expect, type Page, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { COMPARED, REF_DIR, ROOT } from './reference'

/** Agreement with the native engine: relative 1e-9 with an absolute floor
 * of 1e-12 (wasm and native differ only in libm's last bits; docs/WEB.md). */
const REL_TOL = 1e-9
const ABS_TOL = 1e-12

/** RunInfo fields that legitimately differ between runs/platforms. */
const PLATFORM_FIELDS = new Set(['started_unix_ms', 'duration_ms'])

async function openHarness(page: Page, url = '/') {
  await page.goto(url)
  // On the header-less host the service worker reloads the page once.
  await page.waitForFunction(() => globalThis.crossOriginIsolated && 'mk' in window, null, { timeout: 30_000 })
}

interface Stats {
  numbers: number
  identical: number
  maxAbs: number
  maxRel: number
  worst: string
  mismatches: string[]
}

function compare(a: unknown, b: unknown, path: string, s: Stats) {
  const isNum = (x: unknown) => typeof x === 'number' || x === null
  if (isNum(a) && isNum(b) && (typeof a === 'number' || typeof b === 'number')) {
    const x = a === null ? NaN : (a as number)
    const y = b === null ? NaN : (b as number)
    s.numbers++
    if (Object.is(x, y) || (Number.isNaN(x) && Number.isNaN(y))) {
      s.identical++
      return
    }
    const abs = Math.abs(x - y)
    const rel = abs / Math.max(Math.abs(x), Math.abs(y))
    if (abs > s.maxAbs) s.maxAbs = abs
    if (rel > s.maxRel) {
      s.maxRel = rel
      s.worst = `${path}: ${x} vs ${y}`
    }
    if (!(abs <= ABS_TOL || rel <= REL_TOL)) s.mismatches.push(`${path}: ${x} vs ${y}`)
    return
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return void s.mismatches.push(`${path}: length ${a.length} vs ${b.length}`)
    a.forEach((v, i) => compare(v, b[i], `${path}[${i}]`, s))
    return
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)])
    for (const k of keys) compare((a as never)[k], (b as never)[k], `${path}.${k}`, s)
    return
  }
  if (a !== b) s.mismatches.push(`${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`)
}

const newStats = (): Stats => ({ numbers: 0, identical: 0, maxAbs: 0, maxRel: 0, worst: '', mismatches: [] })

test('isolation by the service worker on a header-less static host', async ({ page }) => {
  await openHarness(page, 'http://localhost:4710/')
  const r = await page.evaluate(() => window.mk.ready())
  expect(r.isolated).toBe(true)
  expect(r.threaded).toBe(true)
  expect(r.threads).toBe(r.hardwareConcurrency)
  expect(await page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true)
  console.log('readiness', JSON.stringify(r))
})

test('isolation by response headers', async ({ page }) => {
  await page.goto('http://localhost:4711/')
  await page.waitForFunction(() => 'mk' in window)
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true)
})

test('Cooke triplet: analyses match the native engine', async ({ page }) => {
  await openHarness(page)
  const created = await page.evaluate(() => window.mk.create())
  const info = created.info
  console.log('engine', JSON.stringify(info))
  expect(info.version).toMatch(/^\d+\.\d+\.\d+/)
  expect(info.isolated).toBe(true)
  expect(info.backends.some((b) => b.id === 'cpu_reference' && b.available)).toBe(true)

  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  const opened = await page.evaluate(([t]) => window.mk.openProject(0, t), [text])
  const names = (await page.evaluate(() => window.mk.descriptors(0))).map((d) => d.name)
  for (const a of Object.keys(COMPARED)) expect(names).toContain(a)

  for (const [analysis, settings] of Object.entries(COMPARED)) {
    const ref = JSON.parse(readFileSync(join(REF_DIR, `${analysis}.json`), 'utf8'))
    expect(opened.snapshot.hash).toBe(ref.run_info.snapshot_hash)
    const { result, stages } = await page.evaluate(
      ([a, s]) => window.mk.run(0, { analysis: a as string, settings: s as never, backend: 'cpu_reference' }),
      [analysis, settings],
    )
    expect(stages.at(-1)).toBe('1.00 Done')

    // RunInfo: equal apart from the platform fields.
    for (const [k, v] of Object.entries(ref.run_info)) {
      if (!PLATFORM_FIELDS.has(k)) expect((result.runInfo as Record<string, unknown>)[k], `run_info.${k}`).toEqual(v)
    }
    expect(result.libraryHash).toBe(ref.library_hash)

    const s = newStats()
    compare(result.json, ref.result, 'result', s)
    for (const b of ref.buffers as Array<{ name: string; dtype: string; shape: number[]; data: unknown[] }>) {
      const mine = result.buffers[b.name]
      expect(mine, `buffer ${b.name}`).toBeTruthy()
      expect(mine.dtype).toBe(b.dtype)
      expect(mine.shape).toEqual(b.shape)
      const data = b.dtype === 'i64' || b.dtype === 'u64' ? b.data.map(String) : b.data
      compare(mine.data, data, `buffer ${b.name}`, s)
    }
    console.log(
      `${analysis}: ${s.numbers} numbers, ${s.identical} bit-identical, max abs ${s.maxAbs.toExponential(2)}, ` +
        `max rel ${s.maxRel.toExponential(2)}${s.worst ? ` (${s.worst})` : ''}`,
    )
    expect(s.mismatches.slice(0, 10)).toEqual([])
  }
})

test('errors, projects, catalogs, import, cancellation', async ({ page }) => {
  await openHarness(page)
  await page.evaluate(() => window.mk.create())
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  await page.evaluate(([t]) => window.mk.openProject(0, t), [text])

  // An invalid model.
  const bad = await page.evaluate(() =>
    window.mk.errorOf(() => window.mk.engine(0).openSnapshot({ surfaces: 'nope' } as never)),
  )
  expect(bad?.kind).toBe('invalidModel')
  expect(bad?.issues?.length).toBeGreaterThan(0)

  // Unknown analysis, invalid settings.
  const errs = await page.evaluate(async () => {
    const e = window.mk.engine(0)
    const snap = (await e.openProject(await e.saveProject())).snapshot
    const a = await window.mk.errorOf(() => e.submit({ analysis: 'no_such_analysis' }, snap).result)
    const b = await window.mk.errorOf(() => e.submit({ analysis: 'spot_diagram', settings: { rings: -3 } }, snap).result)
    return [a, b]
  })
  expect(errs[0]?.kind).toBe('notFound')
  expect(errs[1]?.kind).toBe('invalidSettings')

  // Project round trip.
  const saved = await page.evaluate(() => window.mk.engine(0).saveProject())
  expect(JSON.parse(saved).format).toBe('mklight-project')

  // Catalogs and materials.
  const cats = await page.evaluate(() => window.mk.engine(0).catalogs())
  expect(cats[0].source).toBe('builtin')
  const mats = await page.evaluate(() => window.mk.engine(0).materialNames())
  expect(mats.length).toBeGreaterThan(5)

  // Prescription import.
  const zmx = readFileSync(join(ROOT, 'crates/mklight-core/tests/fixtures/prescriptions/cooke_triplet.zmx'))
  const imported = await page.evaluate(
    ([bytes]) => window.mk.engine(0).importPrescription(new Uint8Array(bytes as number[]).buffer, 'zmx'),
    [Array.from(zmx)],
  )
  expect(imported.format).toBe('zmx')
  expect((imported.system.surfaces as unknown[]).length).toBeGreaterThan(5)

  // A second document engine has its own session.
  const second = await page.evaluate(async () => {
    const d = window.mk.engine(0).makeDocumentEngine()
    const info = await d.sessionInfo()
    d.dispose()
    return info
  })
  expect(second.name).toBe('Untitled')

  // Cancellation of a long job.
  const cancelled = await page.evaluate(async () => {
    const e = window.mk.engine(0)
    const snap = (await e.openProject(await e.saveProject())).snapshot
    const job = e.submit({ analysis: 'psf', settings: { grid_size: 1024 }, fresh: true }, snap)
    setTimeout(() => job.cancel(), 50)
    return window.mk.errorOf(() => job.result)
  })
  expect(cancelled?.kind).toBe('cancelled')
})

test('rayon runs in parallel: 1 thread vs all threads', async ({ page }) => {
  await openHarness(page)
  const text = readFileSync(join(REF_DIR, 'cooke.mklight'), 'utf8')
  // Spot diagram, 250 rings (about 190k rays per field and wavelength), no
  // point buffers: ray tracing, parallel over rays.
  const req = {
    analysis: 'spot_diagram',
    settings: { rings: 250, include_points: false },
    backend: 'cpu_reference' as const,
  }
  const r = await page.evaluate(
    async ([t, q]) => {
      const one = (await window.mk.create(1)).index
      const all = await window.mk.create()
      await window.mk.openProject(one, t as string)
      await window.mk.openProject(all.index, t as string)
      await window.mk.time(all.index, q as never, 1) // warm-up
      await window.mk.time(one, q as never, 1)
      const t1 = await window.mk.time(one, q as never, 3)
      const tn = await window.mk.time(all.index, q as never, 3)
      return { threads: all.info.threads, t1, tn }
    },
    [text, req],
  )
  const best = (a: number[]) => Math.min(...a)
  const speedup = best(r.t1) / best(r.tn)
  console.log(
    `spot_diagram 250 rings: 1 thread ${best(r.t1).toFixed(0)} ms, ${r.threads} threads ${best(r.tn).toFixed(0)} ms, ` +
      `speed-up ${speedup.toFixed(2)}×`,
  )
  if (r.threads >= 4) expect(speedup).toBeGreaterThan(2)
})
