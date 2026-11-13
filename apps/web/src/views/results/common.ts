// Shared helpers of the analysis result views: JSON access that tolerates any
// shape, buffer lookup, field labels, canvas set-up and colour ramps.
import { useLayoutEffect, useRef, useState, type RefObject } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf, sig } from '@/model/format'
import { fieldUnit, type System } from '@/model/system'
import { wavelengthColor } from '@/lib/theme'

export const FONT = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'

export const arr = (v: Json | undefined): Json[] => (Array.isArray(v) ? v : [])
export const obj = (v: Json | undefined): JsonObject | null =>
  v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null
export const isObj = (v: Json | undefined): v is JsonObject => obj(v) !== null

/** A numeric array from JSON (`"inf"`/`"nan"`/null become numbers). */
export const nums = (v: Json | undefined): number[] => arr(v).map(nf)

/** A buffer referenced by name (JSON string), as plain numbers. */
export function buffer(r: AnalysisResult, name: Json | undefined): ArrayLike<number> | null {
  if (typeof name !== 'string') return null
  const b = r.buffers[name]
  if (!b) return null
  if (b instanceof BigInt64Array || b instanceof BigUint64Array) return Array.from(b, Number)
  return b as ArrayLike<number>
}

/** The unit of the system's field values: `°` for angles, `mm` for heights. */
export function systemFieldUnit(system: System): string {
  const kind = system.fields?.kind ?? ''
  return fieldUnit(kind) || (kind === 'object_angle' ? '°' : 'mm')
}

function fieldValueText(fv: Json | undefined, unit: string): string {
  const [x, y] = nums(fv)
  const u = unit === '°' ? '°' : ' ' + unit
  if (x === undefined) return ''
  if (y === undefined || (x !== 0 && Number.isFinite(x))) {
    if (y === undefined) return sig(x, 4) + u
    return `(${sig(x, 4)}, ${sig(y, 4)})${u}`
  }
  return sig(y, 4) + u
}

/** "Field 2 · 14°" from a result's field index and value. */
export function fieldLabel(system: System, field: Json | undefined, value: Json | undefined): string {
  const k = typeof field === 'number' ? field : NaN
  const head = Number.isFinite(k) ? `Field ${k + 1}` : 'Field'
  const v = fieldValueText(value, systemFieldUnit(system))
  return v ? `${head} · ${v}` : head
}

/** Wavelengths (µm) by system index: from the result, else from the system. */
export function wavelengthsUm(json: JsonObject, system: System): number[] {
  const w = nums(json.wavelengths_um)
  if (w.length) return w
  return (system.wavelengths?.wavelengths ?? []).map((x) => nf(x.um))
}

export function wlColor(wls: number[], k: number, dark: boolean): string {
  const um = wls[k]
  return Number.isFinite(um) ? wavelengthColor(um, dark) : '#888'
}

/** 1, 2, 5 × 10^k at or above `v`. */
export function niceCeil(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 1
  const e = Math.pow(10, Math.floor(Math.log10(v)))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * e >= v * (1 - 1e-9)) return m * e
  return 10 * e
}

/** About `n` round tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, n = 4): number[] {
  if (!(hi > lo)) return [lo]
  const step = niceCeil((hi - lo) / n)
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

/** Sizes the canvas for the device pixel ratio; null when 2-D canvas is missing. */
export function setupCanvas(c: HTMLCanvasElement | null, w: number, h: number): CanvasRenderingContext2D | null {
  if (!c || w <= 0 || h <= 0) return null
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1
  c.width = Math.round(w * dpr)
  c.height = Math.round(h * dpr)
  c.style.width = `${w}px`
  c.style.height = `${h}px`
  let ctx: CanvasRenderingContext2D | null = null
  try {
    ctx = c.getContext('2d')
  } catch {
    ctx = null
  }
  if (!ctx) return null
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  return ctx
}

/** Content size of an element, tracked by ResizeObserver. */
export function useSize<T extends HTMLElement>(): [RefObject<T | null>, { w: number; h: number }] {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const read = () => {
      const r = el.getBoundingClientRect()
      setSize((s) => (s.w === Math.floor(r.width) && s.h === Math.floor(r.height) ? s : { w: Math.floor(r.width), h: Math.floor(r.height) }))
    }
    read()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(read)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size]
}

// MARK: Colour ramps

type RGB = [number, number, number]
const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
const lerp = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]

function ramp(stops: RGB[], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  return lerp(stops[i], stops[i + 1], x - i)
}

const MAG_LIGHT = ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'].map(hex)
const MAG_DARK = ['#101114', '#0b2c5a', '#2171b5', '#6baed6', '#e6f1ff'].map(hex)
/** One-hue magnitude ramp whose zero recedes into the background (t in 0…1). */
export const magnitudeColor = (t: number, dark: boolean): RGB => ramp(dark ? MAG_DARK : MAG_LIGHT, t)

const DIV_LIGHT = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'].map(hex)
const DIV_DARK = ['#4393c3', '#3b6f99', '#2c3e55', '#3a3a3c', '#5c3a35', '#c0584a', '#e8705f'].map(hex)
/** Diverging blue–red scale; t = −1 … 1 with 0 neutral. */
export const divergingColor = (t: number, dark: boolean): RGB => ramp(dark ? DIV_DARK : DIV_LIGHT, (t + 1) / 2)

export const rgbCss = (c: RGB) => `rgb(${Math.round(c[0])}, ${Math.round(c[1])}, ${Math.round(c[2])})`

/** Largest finite |v| of a list. */
export function maxAbs(vs: Iterable<number>): number {
  let m = 0
  for (const v of vs) if (Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v)
  return m
}
