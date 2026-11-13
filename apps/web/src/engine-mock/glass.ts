// MOCK ENGINE — glass helpers on the mock catalogue: partial dispersion,
// indices at a wavelength, substitution candidates.
import { BUILTIN_CATALOG, CATALOG, type GlassEntry } from './model'
import { cauchy } from './optics'

export const LINE_G = 0.4358343
export const LINE_E = 0.546074

function hash01(name: string): number {
  let h = 2166136261
  for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 16777619) >>> 0
  return (h % 10000) / 10000
}

/** Anomalous partial dispersion ΔP_g,F of a mock glass (small, deterministic). */
export function dpgfOf(name: string): number {
  return (hash01(name.toUpperCase()) - 0.5) * 0.012
}

export function pgfOf(name: string, vd: number): number {
  return 0.6438 - 0.001682 * vd + dpgfOf(name)
}

export function glassEntry(name: string): GlassEntry | null {
  return CATALOG[name.toUpperCase()] ?? null
}

export const catalogOf = (g: GlassEntry) => g.catalog ?? BUILTIN_CATALOG

/** Index of a mock glass (with offsets) at a wavelength in µm. */
export function glassN(g: GlassEntry, um: number, dn = 0, dv = 0): number {
  return cauchy(g.nd + dn, Math.max(g.vd + dv, 1), um)
}

export interface Candidate {
  name: string
  catalog: string
  nd: number
  vd: number
  dpgf: number
  distance: number
}

export function candidateFor(name: string, g: GlassEntry, from: { nd: number; vd: number; dpgf: number }): Candidate {
  const dpgf = dpgfOf(name)
  const d2 = (g.nd - from.nd) ** 2 + ((g.vd - from.vd) / 100) ** 2 + 100 * (dpgf - from.dpgf) ** 2
  return { name, catalog: catalogOf(g), nd: g.nd, vd: g.vd, dpgf, distance: Math.sqrt(d2) }
}

/** The nearest `k` catalogue glasses to `from` (n_d, V_d/100, ΔP_g,F metric). */
export function nearestGlasses(
  from: { nd: number; vd: number; dpgf: number },
  opts: { nearest?: number | null; include?: string[]; exclude?: string[]; catalogs?: string[] } = {},
): Candidate[] {
  const ex = new Set((opts.exclude ?? []).map((s) => s.toUpperCase()))
  const inc = new Set((opts.include ?? []).map((s) => s.toUpperCase()))
  const cats = new Set((opts.catalogs ?? []).map((s) => s.toUpperCase()))
  const user = Object.values(CATALOG).some((g) => g.catalog && g.catalog !== BUILTIN_CATALOG)
  const all = Object.entries(CATALOG)
    .filter(([name, g]) => {
      if (ex.has(name)) return false
      if (cats.size) return cats.has(catalogOf(g).toUpperCase())
      return !user || catalogOf(g) !== BUILTIN_CATALOG || Object.values(CATALOG).every((x) => catalogOf(x) === BUILTIN_CATALOG)
    })
    .map(([name, g]) => ({ ...candidateFor(name, g, from), include: inc.has(name) }))
    .sort((a, b) => a.distance - b.distance || a.name.localeCompare(b.name))
  const k = opts.nearest === null ? all.length : Math.max(opts.nearest ?? 10, 1)
  const kept = all.filter((c, i) => i < k || c.include)
  return kept.map(({ include: _i, ...c }) => c)
}
