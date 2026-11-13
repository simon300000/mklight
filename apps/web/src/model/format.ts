// One number format everywhere (docs/APP.md "One number format"): a decimal
// point, no digit grouping, whatever the browser's locale. Typed numbers also
// accept a decimal comma and ∞ / inf.

/** Engine JSON numbers may be "inf", "-inf", "nan" or null (RESULTS §3). */
export function nf(v: unknown): number {
  if (typeof v === 'number') return v
  if (v === 'inf' || v === 'Infinity') return Infinity
  if (v === '-inf' || v === '-Infinity') return -Infinity
  return NaN
}

const MINUS = '−'

function sign(s: string): string {
  return s.startsWith('-') ? MINUS + s.slice(1) : s
}

/** Fixed decimals for ordinary magnitudes, scientific outside; never −0.0000. */
export function fmt(v: number | null | undefined, decimals = 4): string {
  if (v == null || Number.isNaN(v)) return '–'
  if (!Number.isFinite(v)) return v > 0 ? '∞' : MINUS + '∞'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e7)) return sign(v.toExponential(Math.max(decimals - 1, 1)))
  let s = v.toFixed(decimals)
  if (/^-0\.?0*$/.test(s)) s = s.slice(1)
  return sign(s)
}

/** Up to `digits` significant digits, trailing zeros dropped. */
export function sig(v: number | null | undefined, digits = 6): string {
  if (v == null || Number.isNaN(v)) return '–'
  if (!Number.isFinite(v)) return v > 0 ? '∞' : MINUS + '∞'
  if (v === 0) return '0'
  const a = Math.abs(v)
  let s: string
  if (a < 1e-4 || a >= 1e9) s = v.toExponential(Math.max(digits - 1, 0)).replace(/\.?0+e/, 'e')
  else {
    s = v.toPrecision(digits)
    if (s.includes('e')) s = String(Number(s))
    if (s.includes('.')) s = s.replace(/\.?0+$/, '')
  }
  return sign(s)
}

/** Compact text for a cell at rest: `sig` with 6 digits for lens data. */
export function compact(v: number | null | undefined): string {
  return sig(v, 6)
}

/** Shortest text that parses back to the same number (shown while editing). */
export function exact(v: number): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf'
  return String(v)
}

/** Parses typed text: decimal point or comma, ∞ / inf / infinity, a leading − sign. */
export function parseNumber(text: string): number | null {
  let t = text.trim().replace(/−/g, '-').replace(/\s/g, '')
  if (t === '') return null
  const low = t.toLowerCase()
  if (['∞', '+∞', 'inf', '+inf', 'infinity', '+infinity'].includes(low)) return Infinity
  if (['-∞', '-inf', '-infinity'].includes(low)) return -Infinity
  if (/^[+-]?\d*,\d+(e[+-]?\d+)?$/i.test(t)) t = t.replace(',', '.')
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return null
  const v = Number(t)
  return Number.isNaN(v) ? null : v
}

export function plural(n: number, word: string, pluralWord = word + 's'): string {
  return `${n} ${n === 1 ? word : pluralWord}`
}
