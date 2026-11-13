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
import type { AnalysisResult, JsonObject } from '@/engine/types'
import { MAP_RENDERERS } from './index'
import { craBands, limitAt, readFieldMap, isRadial, isSigned } from './fieldMap'
import { contourSegments, heatRange, isoLevels, mapContours, perceptualColor, upsample } from './heatScale'
import { centerTitle, clippedFraction, fieldPoints, patternImage, resampleLinear, resampleSrgb, snap, srgbDecode, srgbEncode, targetSize, toRgba } from './imageSim'

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

import { FIXTURES, cra, ghostPaths, imageSim, pupilAberration, riMap, system } from './fixtures'

// MARK: Rendering harness (as results.test.tsx)

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

async function render(analysis: string, r: AnalysisResult, onSettingsChange?: (p: JsonObject) => void): Promise<HTMLElement> {
  const el = document.createElement('div')
  el.style.position = 'relative'
  document.body.appendChild(el)
  const root = createRoot(el)
  const R = MAP_RENDERERS[analysis]
  await act(async () => root.render(<R.View analysis={analysis} result={r} system={system} settings={r.runInfo.settings} onSettingsChange={onSettingsChange} />))
  const html = el.innerHTML
  await act(async () => root.unmount())
  el.remove()
  const out = document.createElement('div')
  out.innerHTML = html
  return out
}

describe.each(['null', 'fake'] as const)('map views (canvas %s)', (m) => {
  beforeAll(() => {
    mode = m
  })

  it.each(FIXTURES)('%s renders', async (analysis, r) => {
    expect(MAP_RENDERERS[analysis].accepts?.(r.json) ?? true).toBe(true)
    const el = await render(analysis, r, () => {})
    expect(el.textContent ?? '').not.toMatch(/^\s*$/)
    expect(el.textContent).not.toMatch(/^No (field map|chief|pupil|simulated|ghost irradiance|stray)/)
  })

  it('field maps show the column definition', async () => {
    const el = await render('pupil_aberration', pupilAberration)
    expect(el.textContent).toMatch(/stop_error_x definition/)
  })

  it('image simulation names the patch and the PSF window', async () => {
    const el = await render('image_simulation', imageSim, () => {})
    expect(el.textContent).toMatch(/At Field 2 \(0\.00, \+0\.71\)/)
    expect(el.textContent).toMatch(/PSF 505 × 0\.39 µm/)
    expect(el.textContent).toMatch(/Choose Image…/)
    expect(el.textContent).toMatch(/View 10 %/)
  })

  it('ghost paths list the ranking', async () => {
    const el = await render('ghost_paths', ghostPaths)
    expect(el.textContent).toMatch(/R4·R2/)
    expect(el.textContent).toMatch(/2\.1e−4/)
    expect(el.textContent).toMatch(/path budget/)
  })
})

describe('map key numbers', () => {
  it.each(FIXTURES)('%s gives readable values', (analysis, r) => {
    const k = MAP_RENDERERS[analysis].keyNumbers?.(r, system) ?? []
    expect(k.length).toBeGreaterThan(0)
    for (const x of k) expect(x.value).not.toMatch(/NaN|undefined|Infinity/)
  })

  it('chief ray angle reports its maximum and margin', () => {
    const k = MAP_RENDERERS.chief_ray_angle.keyNumbers!(cra)
    expect(k.map((x) => x.label)).toEqual(['Max CRA', 'at', 'Min margin'])
    expect(k[0].value).toBe('33.94°')
  })

  it('relative illumination map reports the lowest RI inside the field', () => {
    const k = MAP_RENDERERS.relative_illumination_map.keyNumbers!(riMap)
    expect(k[0]).toMatchObject({ label: 'Min RI', value: '0.870' })
  })

  it('image simulation reports pixel, distortion and RI', () => {
    const k = MAP_RENDERERS.image_simulation.keyNumbers!(imageSim)
    expect(k.map((x) => x.label)).toEqual(['Pixel', 'Max distortion', 'Min RI'])
    expect(k[2].value).toBe('0.880')
  })
})

// MARK: Pure helpers

describe('heat scale', () => {
  it('signed data is symmetric about 0', () => {
    expect(heatRange({ min: -0.2, max: 0.5 }, true)).toEqual({ min: -0.5, max: 0.5 })
  })
  it('other data uses its own range, fractions may use 0 to 1', () => {
    expect(heatRange({ min: 0.85, max: 1 }, false)).toEqual({ min: 0.85, max: 1 })
    expect(heatRange({ min: 0.85, max: 1 }, false, 'unit', true)).toEqual({ min: 0, max: 1 })
    expect(heatRange({ min: 0.85, max: 1 }, false, 'unit', false)).toEqual({ min: 0.85, max: 1 })
  })
  it('a flat map gets a narrow band around its value', () => {
    const r = heatRange({ min: 2, max: 2 }, false)
    expect(r.min).toBeLessThan(2)
    expect(r.max).toBeGreaterThan(2)
    expect(r.max - r.min).toBeLessThan(0.01)
  })
  it('the perceptual ramp starts at viridis 0.1 and ends yellow', () => {
    expect(perceptualColor(0)).toEqual([0x48, 0x24, 0x75])
    expect(perceptualColor(1)).toEqual([0xfd, 0xe7, 0x25])
    // Lightness rises monotonically.
    const lum = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    for (let t = 0.1; t <= 1; t += 0.1) expect(lum(perceptualColor(t))).toBeGreaterThan(lum(perceptualColor(t - 0.1)))
  })
  it('field-map signedness', () => {
    const d = readFieldMap(pupilAberration)!
    expect(isSigned(d, 0)).toBe(true)
    expect(isSigned(d, 2)).toBe(true) // a variation, zero on axis
    expect(isRadial(readFieldMap(riMap)!, 0)).toBe(true)
    expect(isRadial(d, 0)).toBe(false)
  })
})

describe('iso-lines', () => {
  const g = { nx: 3, ny: 3, values: [0, 0, 0, 0, 1, 0, 0, 0, 0], x: [-1, 1] as [number, number], y: [-1, 1] as [number, number] }
  it('trace a closed diamond around a peak', () => {
    const s = contourSegments(g, 0.5)
    expect(s).toHaveLength(4)
    for (const [a, b] of s) for (const p of [a, b]) expect(Math.abs(p[0]) + Math.abs(p[1])).toBeCloseTo(0.5)
  })
  it('skip cells with a missing corner', () => {
    expect(contourSegments({ ...g, values: [NaN, 0, 0, 0, 1, 0, 0, 0, 0] }, 0.5)).toHaveLength(3)
  })
  it('levels are round and strictly inside the range', () => {
    expect(isoLevels({ min: 0.87, max: 1 })).toEqual([0.9, 0.95])
    expect(isoLevels({ min: 0.5, max: 0.5 })).toEqual([])
    for (const v of isoLevels({ min: 0.62, max: 0.99 })) expect(v > 0.62 && v < 0.99).toBe(true)
  })
  it('a masked map is traced to the field edge and never outside it', () => {
    const d = readFieldMap(riMap)!
    const grid = { nx: d.nx, ny: d.ny, values: Array.from({ length: d.nx * d.ny }, (_, k) => d.values[k * 2]), x: [-1, 1] as [number, number], y: [-1, 1] as [number, number], unitDisk: true }
    const up = upsample(grid, 64)
    expect(Number.isNaN(up.values[0])).toBe(true) // the corner is outside the field
    const segs = mapContours(grid, 0.95)
    expect(segs.length).toBeGreaterThan(10)
    for (const [a, b] of segs) for (const p of [a, b]) expect(Math.hypot(p[0], p[1])).toBeLessThanOrEqual(1 + 1e-6)
    const rmax = Math.max(...segs.flatMap(([a, b]) => [Math.hypot(...a), Math.hypot(...b)]))
    expect(rmax).toBeGreaterThan(0.55)
  })
})

describe('chief ray angle limit', () => {
  const lim: [number, number][] = [[0, 25], [10, 15]]
  it('is linear and held at the ends', () => {
    expect(limitAt(lim, -1)).toBe(25)
    expect(limitAt(lim, 5)).toBe(20)
    expect(limitAt(lim, 20)).toBe(15)
  })
  it('bands split where the curve crosses the limit', () => {
    const b = craBands([[0, 0], [10, 20]], lim)
    expect(b.map((x) => x.violated)).toEqual([false, true])
    expect(b[0].points[b[0].points.length - 1].x).toBeCloseTo(25 / 3)
  })
})

describe('image simulation helpers', () => {
  it('sRGB round-trips', () => {
    for (const v of [0, 0.001, 0.2, 0.5, 1]) expect(srgbDecode(srgbEncode(v))).toBeCloseTo(v, 9)
  })
  it('linear-light resampling averages light, not code values', () => {
    // A 2 × 1 black/white image averaged to one pixel is 50 % light (sRGB code 188), not code 128.
    const lin = resampleSrgb({ width: 2, height: 1, data: [0, 0, 0, 255, 255, 255, 255, 255] }, 1, 1)
    expect(lin.channels).toBe(1)
    expect(lin.data[0]).toBeCloseTo(0.5, 6)
    expect(toRgba(lin)[0]).toBe(188)
  })
  it('crops to the field aspect about the centre and flips rows to bottom-up', () => {
    // 4 × 2 RGB image: left half red, right half blue; top row bright, bottom dark.
    const px: number[] = []
    for (let y = 0; y < 2; y++) for (let x = 0; x < 4; x++) px.push(x < 2 ? (y === 0 ? 255 : 64) : 0, 0, x >= 2 ? (y === 0 ? 255 : 64) : 0, 255)
    const lin = resampleSrgb({ width: 4, height: 2, data: px }, 2, 2, 1)
    expect(lin.channels).toBe(3)
    // Cropped to the central 2 × 2: column 0 red, column 1 blue; row 0 is the bottom (dark).
    expect(lin.data[0]).toBeCloseTo(srgbDecode(64 / 255))
    expect(lin.data[2]).toBe(0)
    expect(lin.data[(1 * 2 + 1) * 3 + 2]).toBeCloseTo(1)
  })
  it('resamples a linear buffer by area', () => {
    const r = resampleLinear({ width: 4, height: 1, channels: 1, data: [0, 1, 1, 1] }, 2, 1)
    expect(Array.from(r.data)).toEqual([0.5, 1])
  })
  it('built-in patterns follow the engine definition', () => {
    const star = patternImage('siemens_star', 32, 32, 4)!
    const mean = Array.from(star.data).reduce((a, b) => a + b) / star.data.length
    expect(mean).toBeGreaterThan(0.4)
    expect(mean).toBeLessThan(0.6)
    expect(patternImage('grid', 80, 80)!.data[0]).toBe(0)
    expect(patternImage('grid', 80, 80)!.data[5 * 80 + 5]).toBe(1)
    expect(patternImage('nope', 8, 8)).toBeNull()
    expect(patternImage('text_chart', 30, 30)).not.toBeNull()
  })
  it('target size is at most 512 on a side', () => {
    expect(targetSize([1024, 1024])).toEqual([512, 512])
    expect(targetSize([256, 128])).toEqual([256, 128])
  })
  it('clipped fraction follows exposure', () => {
    const img = { width: 2, height: 1, channels: 1, data: [0.4, 0.9] }
    expect(clippedFraction(img, 1)).toBe(0)
    expect(clippedFraction(img, 2)).toBe(0.5)
  })
  it('field points, centre titles and snapping', () => {
    const pts = fieldPoints(system)
    expect(pts.map((p) => p.title)).toEqual(['On axis', 'Field 2 (0.00, +0.71)', 'Field 3 (0.00, +1.00)'])
    expect(centerTitle([0, 0], pts)).toBe('on axis')
    expect(centerTitle([0.3, -0.45], pts)).toBe('(+0.30, −0.45)')
    expect(snap([0.02, 0.7], pts.map((p) => p.field), 0.06)[1]).toBeCloseTo(10 / 14)
    expect(snap([0.5, 0.5], [], 0.06)).toEqual([0.5, 0.5])
  })
})
