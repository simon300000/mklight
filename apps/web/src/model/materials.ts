// Materials of the lens data (apps/macos/Sources/MKLightModel/Materials.swift,
// MKLightUI/ViewModels/GlassOffsets.swift, Support/LensValueParser.swift):
// what the material column accepts, completions from the session's
// catalogs, and the melt offsets (Δn_d, ΔV_d) of catalogue glasses
// (docs/MATERIALS.md).

import type { Json, JsonObject } from '@/engine/types'
import { parseNumber } from './format'
import type { Material, Surface } from './system'

export type MaterialEdit = { interaction: string; material: Material | null } | { error: string }

/** Keywords the material column understands besides glass names. */
export const MATERIAL_KEYWORDS = ['MIRROR', 'AIR', 'VACUUM', 'DUMMY'] as const

/** Parses the material column: a glass name (or `CATALOG:NAME`), `MIRROR`,
 *  `DUMMY`, `VACUUM`, empty or `AIR`, a constant index (`n=1.5`, `=1.5`,
 *  `1.5`), a model glass `n_d/V_d` (`1.5168/64.17`). With `known` glass
 *  names, an unknown name is refused (the engine would reject it). */
export function parseMaterial(text: string, known?: Set<string> | readonly string[]): MaterialEdit {
  const t = text.trim()
  const up = t.toUpperCase()
  if (t === '' || up === 'AIR') return { interaction: 'refract', material: { type: 'air' } }
  if (up === 'MIRROR') return { interaction: 'reflect', material: null }
  if (up === 'DUMMY') return { interaction: 'dummy', material: null }
  if (up === 'VACUUM') return { interaction: 'refract', material: { type: 'vacuum' } }
  const parts = t.split(/\s*[/]\s*/)
  if (parts.length === 2) {
    const nd = parseNumber(parts[0])
    const vd = parseNumber(parts[1])
    if (nd == null || vd == null || !(nd >= 1) || !Number.isFinite(nd) || !(vd > 0) || !Number.isFinite(vd)) {
      return { error: `"${t}" is not a model glass n_d/V_d (n_d ≥ 1, V_d > 0).` }
    }
    return { interaction: 'refract', material: { type: 'model', nd, vd, dpgf: 0 } }
  }
  let constant = up.replace(/\s/g, '')
  if (constant.startsWith('N=')) constant = constant.slice(2)
  else if (constant.startsWith('=')) constant = constant.slice(1)
  if (/^[0-9.,]/.test(constant)) {
    const n = parseNumber(constant)
    if (n != null) {
      if (!(n >= 1) || !Number.isFinite(n)) return { error: `Index ${constant} is not valid (n ≥ 1).` }
      return { interaction: 'refract', material: { type: 'constant', n } }
    }
  }
  const colon = t.indexOf(':')
  if (colon >= 0) {
    const cat = t.slice(0, colon).trim()
    const name = t.slice(colon + 1).trim()
    if (!cat || !name) return { error: `"${t}" is not a material.` }
    return { interaction: 'refract', material: { type: 'catalog', catalog: cat.toUpperCase(), name: name.toUpperCase() } }
  }
  if (!/^[\p{L}\p{N}\-_.+]+$/u.test(t)) return { error: `"${t}" is not a material.` }
  const names = known ? [...known] : []
  if (names.length) {
    const hit = names.find((g) => g.toUpperCase() === up)
    if (!hit) return { error: `Unknown glass "${t}". Load its catalog from the System inspector.` }
    return { interaction: 'refract', material: { type: 'catalog', catalog: null, name: hit } }
  }
  return { interaction: 'refract', material: { type: 'catalog', catalog: null, name: up } }
}

/** Completions for typed material text: keywords and catalogue glasses that
 *  start with it (then those containing it), at most `limit`. */
export function materialCompletions(text: string, glasses: readonly string[], limit = 12): string[] {
  const q = text.trim().toUpperCase()
  if (!q || /^[0-9.=]/.test(q) || q.includes('/')) return []
  const all = [...MATERIAL_KEYWORDS, ...glasses]
  const starts = all.filter((g) => g.toUpperCase().startsWith(q))
  const contains = all.filter((g) => !g.toUpperCase().startsWith(q) && g.toUpperCase().includes(q))
  const out = [...new Set([...starts, ...contains])]
  return out.length === 1 && out[0].toUpperCase() === q ? [] : out.slice(0, limit)
}

// MARK: Melt offsets

export const MAX_DN_D = 0.05
export const MAX_DV_D = 10

export interface MaterialOffset {
  dn_d: number
  dv_d: number
}

export function offsetOf(m: Material | null | undefined): MaterialOffset | null {
  if (!m || m.type !== 'catalog') return null
  const o = (m.offset as JsonObject | undefined) ?? {}
  return { dn_d: Number(o.dn_d ?? 0) || 0, dv_d: Number(o.dv_d ?? 0) || 0 }
}

export const isZeroOffset = (o: MaterialOffset | null) => !o || (o.dn_d === 0 && o.dv_d === 0)

/** The same catalogue glass with offset `o` (omitted when zero). */
export function withOffset(m: Material, o: MaterialOffset): Material {
  const { offset: _old, ...rest } = m
  void _old
  return isZeroOffset(o) ? (rest as Material) : ({ ...rest, offset: { dn_d: o.dn_d, dv_d: o.dv_d } } as Material)
}

/** A signed number with a true minus and no trailing zeros; `0` for zero. */
export function signedText(v: number, decimals: number): string {
  if (v === 0) return '0'
  let s = Math.abs(v).toFixed(decimals)
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '')
  return (v < 0 ? '−' : '+') + s
}

/** The table's subtle suffix `+0.0012/−0.8` ('' when zero). */
export function offsetCompactText(o: MaterialOffset | null): string {
  if (!o || isZeroOffset(o)) return ''
  return `${signedText(o.dn_d, 4)}/${signedText(o.dv_d, 2)}`
}

/** Why `v` cannot be a Δn_d / ΔV_d (the engine's ±0.05 and ±10), if it cannot. */
export function offsetRejection(which: 'dn_d' | 'dv_d', v: number): string | null {
  const max = which === 'dn_d' ? MAX_DN_D : MAX_DV_D
  if (Number.isFinite(v) && Math.abs(v) <= max) return null
  return `${which === 'dn_d' ? 'Δn_d' : 'ΔV_d'} must lie within ±${max}`
}

/** `new` typed over `old`: the same catalogue glass keeps its melt offset;
 *  another glass starts without, since the offset belonged to that melt. */
export function replacedMaterial(old: Material, next: Material): Material {
  const o = offsetOf(old)
  if (!o || isZeroOffset(o) || next.type !== 'catalog' || !isZeroOffset(offsetOf(next))) return next
  if (String(next.name).toUpperCase() !== String(old.name).toUpperCase()) return next
  const nc = next.catalog as string | null | undefined
  const oc = old.catalog as string | null | undefined
  if (nc && oc && nc.toUpperCase() !== oc.toUpperCase()) return next
  return { ...next, catalog: (nc ?? oc ?? null) as Json, offset: { dn_d: o.dn_d, dv_d: o.dv_d } }
}

/** Applies a material edit to a surface (keeps a retyped glass's offset). */
export function applyMaterial(surface: Surface, edit: { interaction: string; material: Material | null }) {
  surface.interaction = edit.interaction
  if (edit.material) surface.material = replacedMaterial(surface.material, edit.material)
}

/** A catalogue glass's d-line numbers from the engine's `glass_info`: with
 *  its offset and as catalogued. */
export interface GlassNumbers {
  catalog: string | null
  nd: number
  vd: number
  catalogNd: number
  catalogVd: number
}

export function glassNumbers(json: JsonObject | null | undefined): GlassNumbers | null {
  if (!json) return null
  const nd = Number(json.nd)
  const vd = Number(json.vd)
  if (!Number.isFinite(nd) || !Number.isFinite(vd)) return null
  const cn = Number(json.catalog_nd)
  const cv = Number(json.catalog_vd)
  return {
    catalog: typeof json.catalog === 'string' ? json.catalog : null,
    nd,
    vd,
    catalogNd: Number.isFinite(cn) ? cn : nd,
    catalogVd: Number.isFinite(cv) ? cv : vd,
  }
}

// MARK: Glass catalogs of the project

/** A catalog reference of the project's `catalogs` section (docs/FFI.md). */
export interface CatalogReference extends JsonObject {
  name: string
  path: string | null
  relative_path: string | null
  format: string
}

export function catalogReferences(catalogs: Json[] | null | undefined): CatalogReference[] {
  return (catalogs ?? []).filter((c): c is CatalogReference => !!c && typeof c === 'object' && !Array.isArray(c) && typeof (c as JsonObject).name === 'string')
}

/** The reference a loaded file records: named like the engine names it (the
 *  file stem for AGF), replacing a catalog of the same name. */
export function catalogReference(fileName: string, name?: string): CatalogReference {
  const stem = fileName.replace(/\.[^.]+$/, '')
  const ext = /\.([^.]+)$/.exec(fileName)?.[1]?.toLowerCase() ?? 'agf'
  return { name: name ?? stem.toUpperCase(), path: fileName, relative_path: null, format: ext === 'json' ? 'json' : 'agf' }
}

export function addCatalogReference(catalogs: Json[], ref: CatalogReference): Json[] {
  const rest = catalogReferences(catalogs).filter((c) => c.name.toUpperCase() !== ref.name.toUpperCase())
  return [...rest, ref]
}
