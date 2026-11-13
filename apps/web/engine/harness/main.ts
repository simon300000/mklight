// Test harness of @engine: exposes `window.mk` for the Playwright tests
// (tests/engine.spec.ts) and prints the readiness report.

import {
  createEngine,
  type EngineClient,
  EngineError,
  ready,
  type AnalysisRequest,
  type AnalysisResult,
  type ModelSnapshot,
} from '../src/index'

type Plain = {
  json: unknown
  runInfo: unknown
  libraryHash: string | null
  backend: unknown
  buffers: Record<string, { dtype: string; shape: number[]; data: (number | string)[] }>
}

/** An AnalysisResult as plain data for the test runner (BigInts as strings). */
function plain(r: AnalysisResult): Plain {
  const buffers: Plain['buffers'] = {}
  for (const b of r.bufferInfo) {
    const a = r.buffers[b.name] as ArrayLike<number | bigint>
    buffers[b.name] = { dtype: b.dtype, shape: b.shape, data: Array.from(a, (x) => (typeof x === 'bigint' ? x.toString() : x)) }
  }
  return { json: r.json, runInfo: r.runInfo, libraryHash: r.libraryHash, backend: r.backend, buffers }
}

const engines: EngineClient[] = []
const snaps: ModelSnapshot[] = []

const mk = {
  ready,
  /** Create an engine; returns its index. */
  async create(threads?: number, webgpuCalibration?: 'auto' | 'off') {
    const e = await createEngine({ ...(threads ? { threads } : {}), ...(webgpuCalibration ? { webgpuCalibration } : {}) })
    engines.push(e)
    return { index: engines.length - 1, info: e.info }
  },
  async openProject(i: number, text: string) {
    const r = await engines[i].openProject(text)
    snaps[i] = r.snapshot
    return r
  },
  async descriptors(i: number) {
    return engines[i].analysisDescriptors()
  },
  /** Submit on the current snapshot and wait; returns the result as plain data. */
  async run(i: number, req: AnalysisRequest) {
    const job = engines[i].submit(req, snaps[i])
    const stages: string[] = []
    job.onProgress((p) => stages.push(`${p.fraction.toFixed(2)} ${p.stage}`))
    const r = await job.result
    return { result: plain(r), stages }
  },
  /** Time `n` fresh runs of `req` (ms each). */
  async time(i: number, req: AnalysisRequest, n: number) {
    const ms: number[] = []
    for (let k = 0; k < n; k++) {
      const t0 = performance.now()
      await engines[i].submit({ ...req, fresh: true }, snaps[i]).result
      ms.push(performance.now() - t0)
    }
    return ms
  },
  /** SHA-256 of a fresh run's result JSON and buffer bytes (bitwise identity). */
  async digest(i: number, req: AnalysisRequest) {
    const r = await engines[i].submit({ ...req, fresh: true }, snaps[i]).result
    const parts: Uint8Array[] = [new TextEncoder().encode(JSON.stringify(r.json))]
    for (const b of r.bufferInfo) {
      const a = r.buffers[b.name]
      parts.push(new Uint8Array(a.buffer, a.byteOffset, a.byteLength))
    }
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
    let o = 0
    for (const p of parts) {
      all.set(p, o)
      o += p.length
    }
    const h = new Uint8Array(await crypto.subtle.digest('SHA-256', all))
    return Array.from(h.slice(0, 12), (x) => x.toString(16).padStart(2, '0')).join('')
  },
  async errorOf(fn: () => Promise<unknown>) {
    try {
      await fn()
      return null
    } catch (e) {
      return e instanceof EngineError ? { kind: e.kind, message: e.message, issues: e.issues } : { kind: 'other', message: String(e) }
    }
  },
  engine: (i: number) => engines[i],
  calibration: (i: number) => engines[i].webgpuCalibration(),
  setCalibration: (i: number, c: Record<string, unknown> | null) => engines[i].setWebgpuCalibration(c as never),
}

declare global {
  interface Window {
    mk: typeof mk
  }
}
window.mk = mk

ready().then((r) => {
  document.getElementById('out')!.textContent = JSON.stringify(r, null, 2)
})
