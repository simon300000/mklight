import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { AnalysisResult } from '@/engine/types'
import type { System } from '@/model/system'
import { CURVE_RENDERERS } from './index'
import { isAnamorphic } from './tables'
import { length, short } from './kit'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

import { FIXTURES, res, system } from './fixtures'

/** A 2-D context that accepts every call (jsdom has none). */
function fakeContext(): CanvasRenderingContext2D {
  const target: Record<string, unknown> = {
    measureText: (t: string) => ({ width: t.length * 6, actualBoundingBoxAscent: 8, actualBoundingBoxDescent: 2 }),
    createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (_x: number, _y: number, w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    createLinearGradient: () => ({ addColorStop() {} }),
    getLineDash: () => [],
  }
  return new Proxy(target, {
    get: (t, k: string) => (k in t ? t[k] : typeof k === 'string' && /^[a-z]/.test(k) && k !== 'canvas' ? () => {} : undefined),
    set: (t, k: string, v) => ((t[k] = v), true),
  }) as unknown as CanvasRenderingContext2D
}

let mode: 'null' | 'fake' = 'null'
let calls = 0
const saved = { getContext: HTMLCanvasElement.prototype.getContext, rect: HTMLElement.prototype.getBoundingClientRect, mm: window.matchMedia }

beforeAll(() => {
  window.matchMedia ??= ((q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia
  HTMLCanvasElement.prototype.getContext = function () {
    if (mode === 'fake') calls++
    return mode === 'fake' ? fakeContext() : null
  } as typeof HTMLCanvasElement.prototype.getContext
  HTMLElement.prototype.getBoundingClientRect = function () {
    return { x: 0, y: 0, top: 0, left: 0, right: 640, bottom: 480, width: 640, height: 480, toJSON() {} } as DOMRect
  }
})
afterAll(() => {
  HTMLCanvasElement.prototype.getContext = saved.getContext
  HTMLElement.prototype.getBoundingClientRect = saved.rect
})

async function render(analysis: string, result: AnalysisResult): Promise<HTMLElement> {
  const r = CURVE_RENDERERS[analysis]
  const el = document.createElement('div')
  document.body.appendChild(el)
  const root = createRoot(el)
  await act(async () => root.render(<r.View analysis={analysis} result={result} system={system} settings={{}} />))
  const out = document.createElement('div')
  out.innerHTML = el.innerHTML
  await act(async () => root.unmount())
  el.remove()
  return out
}

const NAMES = Object.keys(FIXTURES)

it('every curve renderer has a fixture', () => {
  expect(Object.keys(CURVE_RENDERERS).sort()).toEqual([...NAMES].sort())
})

describe.each(['null', 'fake'] as const)('curve views (canvas %s)', (m) => {
  beforeAll(() => {
    mode = m
  })

  it.each(NAMES)('%s renders', async (analysis) => {
    const r = CURVE_RENDERERS[analysis]
    expect(r.accepts ? r.accepts(FIXTURES[analysis].json) : true).toBe(true)
    calls = 0
    const el = await render(analysis, FIXTURES[analysis])
    expect(el.textContent ?? '').not.toMatch(/^\s*$/)
    expect(el.textContent).not.toMatch(/NaN|undefined|No .* in this result/)
    if (m === 'fake' && !['first_order', 'system_summary', 'validate'].includes(analysis)) expect(calls).toBeGreaterThan(0)
  })

  it('tolerates empty results', async () => {
    for (const a of NAMES) {
      const el = await render(a, res(a, {}))
      expect(el).toBeTruthy()
    }
  })
})

describe('specifics', () => {
  it('first order shows the anamorphic card', async () => {
    const el = await render('first_order', FIXTURES.first_order)
    expect(el.textContent).toMatch(/Anamorphic/)
    expect(el.textContent).toMatch(/Squeeze/)
    expect(el.textContent).toMatch(/Other wavelengths \(1\)/)
    expect(isAnamorphic({}, { surfaces: [{ shape: { type: 'toroid' } }] } as unknown as System)).toBe(true)
    expect(isAnamorphic({ symmetry: 'rotational' })).toBe(false)
  })

  it('seidel legend uses real subscripts, and not-applicable explains', async () => {
    const el = await render('seidel', FIXTURES.seidel)
    expect([...el.querySelectorAll('sub')].map((s) => s.textContent)).toEqual(['I', 'II', 'III', 'IV', 'V'])
    expect(el.textContent).toMatch(/Petzval radius −120\.5 mm/)
    const na = await render('seidel', res('seidel', { status: 'not_applicable', reason: 'Not rotational', surfaces: [], sums: {} }))
    expect(na.textContent).toMatch(/Do Not Apply/)
    expect(CURVE_RENDERERS.seidel.keyNumbers!(res('seidel', { status: 'not_applicable' }))).toEqual([])
  })

  it('validate lists errors first', async () => {
    const el = await render('validate', FIXTURES.validate)
    expect(el.querySelector('li')?.textContent).toMatch(/Material not found/)
    const ok = await render('validate', res('validate', { ok: true, errors: 0, warnings: 0, issues: [] }))
    expect(ok.textContent).toMatch(/No Issues/)
  })

  it('formats like the macOS panes', () => {
    expect(length(0.0123)).toBe('12.30 µm')
    expect(length(-0.00000001)).toBe('0.00 µm')
    expect(length(1.5)).toBe('1.5000 mm')
    expect(short(0.0001234)).toBe('1.234e-4')
    expect(short(Infinity)).toBe('∞')
  })
})

describe('keyNumbers', () => {
  it.each(NAMES)('%s gives readable values', (analysis) => {
    const k = CURVE_RENDERERS[analysis].keyNumbers?.(FIXTURES[analysis], system) ?? []
    for (const x of k) {
      expect(x.label).not.toBe('')
      expect(x.value).not.toMatch(/NaN|undefined|—/)
    }
  })

  it('mirrors the macOS footer for the main ones', () => {
    const kn = (a: string) => CURVE_RENDERERS[a].keyNumbers!(FIXTURES[a], system)
    expect(kn('first_order').map((x) => x.label)).toEqual(['EFL', 'F/#', 'BFL'])
    expect(kn('seidel').map((x) => x.label)).toEqual(['Σ S_I', 'Σ S_II', 'Σ S_III'])
    expect(kn('grid_distortion')[0]).toMatchObject({ label: 'Max distortion', value: '1.235 %' })
    expect(kn('chromatic_focal_shift')[0]).toMatchObject({ label: 'Range', value: '20.00 µm' })
    expect(kn('best_focus')[0]).toMatchObject({ label: 'Best defocus', value: '−12.00 µm' })
    expect(kn('validate').map((x) => x.value)).toEqual(['1', '1'])
  })
})
