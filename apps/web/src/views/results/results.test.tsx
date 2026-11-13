import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

// uPlot reads matchMedia when imported.
vi.hoisted(() => {
  window.matchMedia ??= ((q: string) => ({
    matches: false,
    media: q,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia
})
import samples from '@samples'
import { createMockEngine } from '@/engine-mock'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { RENDERERS, ResultView, keyNumbers } from './index'
import { FIXTURES as CURVE_FIXTURES, system as curveSystem } from '../analysis/curves/fixtures'
import { FIXTURES as MAP_FIXTURES, system as mapSystem } from '../analysis/maps/fixtures'

/** One fixture per dedicated renderer, with the system it was made for. */
const RENDERER_FIXTURES: [string, AnalysisResult, System][] = [
  ...Object.entries(CURVE_FIXTURES).map(([a, r]): [string, AnalysisResult, System] => [a, r, curveSystem]),
  ...MAP_FIXTURES.map(([a, r]): [string, AnalysisResult, System] => [a, r, mapSystem]),
]

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

type Sample = { id: string; system: JsonObject }
const system = (samples as unknown as Sample[]).find((s) => s.id === 'double_gauss')!.system
const eng = createMockEngine({ latencyMs: 0 })

const ANALYSES = ['spot_diagram', 'ray_fan', 'opd_fan', 'mtf', 'psf', 'field_curvature_distortion', 'wavefront_map', 'first_order']
const EXTRA = ['system_summary', 'validate', 'seidel']

/** A 2-D context that accepts every call (jsdom has none). */
function fakeContext(): CanvasRenderingContext2D {
  const target: Record<string, unknown> = {
    measureText: (t: string) => ({ width: t.length * 6, actualBoundingBoxAscent: 8, actualBoundingBoxDescent: 2 }),
    createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (_x: number, _y: number, w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    createLinearGradient: () => ({ addColorStop() {} }),
    createPattern: () => ({}),
    getLineDash: () => [],
    isPointInPath: () => false,
  }
  return new Proxy(target, {
    get: (t, k: string) => (k in t ? t[k] : typeof k === 'string' && /^[a-z]/.test(k) && !['canvas'].includes(k) ? () => {} : undefined),
    set: (t, k: string, v) => ((t[k] = v), true),
  }) as unknown as CanvasRenderingContext2D
}

let mode: 'null' | 'fake' = 'null'
const saved = { getContext: HTMLCanvasElement.prototype.getContext, rect: HTMLElement.prototype.getBoundingClientRect }

beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = function () {
    return mode === 'fake' ? fakeContext() : null
  } as typeof HTMLCanvasElement.prototype.getContext
  HTMLElement.prototype.getBoundingClientRect = function () {
    return { x: 0, y: 0, top: 0, left: 0, right: 640, bottom: 480, width: 640, height: 480, toJSON() {} } as DOMRect
  }
  globalThis.Path2D ??= class {
    moveTo() {}
    lineTo() {}
    rect() {}
    arc() {}
    closePath() {}
    bezierCurveTo() {}
    addPath() {}
  } as unknown as typeof Path2D
})
afterAll(() => {
  HTMLCanvasElement.prototype.getContext = saved.getContext
  HTMLElement.prototype.getBoundingClientRect = saved.rect
})

async function run(analysis: string, settings?: JsonObject): Promise<AnalysisResult> {
  const snap = await eng.openSnapshot(system)
  return eng.submit({ analysis, settings }, snap).result
}

async function render(analysis: string, result: AnalysisResult, sys: System = system as System): Promise<HTMLElement> {
  const el = document.createElement('div')
  el.style.position = 'relative'
  document.body.appendChild(el)
  const root = createRoot(el)
  await act(async () => root.render(<ResultView analysis={analysis} result={result} system={sys} onSettingsChange={() => {}} />))
  const html = el.innerHTML
  await act(async () => root.unmount())
  el.remove()
  const out = document.createElement('div')
  out.innerHTML = html
  return out
}

describe.each(['null', 'fake'] as const)('result views (canvas %s)', (m) => {
  beforeAll(() => {
    mode = m
  })

  it.each([...ANALYSES, ...EXTRA])('%s renders', async (analysis) => {
    const r = await run(analysis)
    const el = await render(analysis, r)
    expect(el.textContent ?? '').not.toMatch(/^\s*$/)
    expect(el.textContent).not.toMatch(/No (fields|curves|MTF|PSF|wavefront)/)
    if (m === 'fake' && ['ray_fan', 'mtf', 'field_curvature_distortion'].includes(analysis))
      expect(el.querySelector('.uplot canvas[role="img"]')).not.toBeNull()
  })

  it('spot diagram through focus has one column per plane', async () => {
    const r = await run('spot_diagram', { through_focus: { steps: 3 } })
    const el = await render('spot_diagram', r)
    expect(el.textContent).toMatch(/Δ/)
    expect(el.textContent).toMatch(/Box/)
    const fields = (r.json.fields as unknown[]).length
    expect(el.querySelectorAll('canvas[role="img"]').length).toBe(fields * 3)
  })

  it('every dedicated renderer has a fixture', () => {
    const covered = new Set(RENDERER_FIXTURES.map(([a]) => a))
    expect(Object.keys(RENDERERS).filter((a) => !covered.has(a))).toEqual([])
  })

  it.each(RENDERER_FIXTURES)('%s renders its dedicated view', async (analysis, r, sys) => {
    const accepts = RENDERERS[analysis].accepts
    expect(accepts ? accepts(r.json) : true).toBe(true)
    const el = await render(analysis, r, sys)
    expect(el.textContent ?? '').not.toMatch(/^\s*$/)
    expect(el.textContent).not.toMatch(/could not be drawn/)
  })

  it('unknown results render as data', async () => {
    const r: AnalysisResult = {
      json: { note: 'mock', analysis: 'whatever', nested: { a: [1, 2, 'inf'], rows: [{ x: 1 }, { x: 2, y: null }] } },
      buffers: { b: new Float64Array(4) },
      shapes: { b: [2, 2] },
      runInfo: { snapshot_hash: '', analysis: 'whatever', settings: {}, backend: 'cpu_reference', warnings: [] },
    }
    const el = await render('whatever', r)
    expect(el.textContent).toMatch(/mock/)
    expect(el.textContent).toMatch(/2 × 2/)
  })
})

describe('keyNumbers', () => {
  it.each(RENDERER_FIXTURES)('%s (dedicated) gives readable values', (analysis, r, sys) => {
    for (const x of keyNumbers(analysis, r, sys)) {
      expect(x.value).not.toMatch(/NaN|undefined/)
      expect(x.label).not.toMatch(/undefined/)
    }
  })

  it('spot diagram gives finite RMS and Airy radii', async () => {
    const k = keyNumbers('spot_diagram', await run('spot_diagram'))
    expect(k.map((x) => x.label)).toEqual(['Max RMS radius', 'Airy radius'])
    for (const x of k) expect(x.value).toMatch(/^\d+\.\d+ µm$/)
  })

  it.each(ANALYSES)('%s gives readable values', async (analysis) => {
    for (const x of keyNumbers(analysis, await run(analysis))) {
      expect(x.value).not.toMatch(/NaN|undefined|–/)
    }
  })
})
