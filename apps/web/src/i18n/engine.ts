// Text the engine writes is translated through the shared catalog
// `locales/<lang>/engine.json` (docs/I18N.md): `strings` by exact English,
// `enums` by raw settings value, `diagnostics` by code. Looked up only where
// shown: anything the app parses or matches stays the engine's English. A
// miss shows the English. The file loads with the language's UI catalog (a
// separate chunk); the build tolerates its absence.

import { fill } from './fill'
import type { Params } from './types'

export interface EngineCatalog {
  strings?: Record<string, string>
  enums?: Record<string, string>
  diagnostics?: Record<string, string>
}

// A glob, not an import: no file, no chunk, no build error.
const files = import.meta.glob<EngineCatalog>('../../../../locales/*/engine.json', { import: 'default' })

/** The engine catalog for a language, or null when there is none. */
export async function loadEngineCatalog(lang: string): Promise<EngineCatalog | null> {
  const load = files[`../../../../locales/${lang}/engine.json`]
  if (!load) return null
  try {
    return await load()
  } catch {
    return null
  }
}

let active: EngineCatalog | null = null

/** Installs the engine catalog of the current language (null: English). */
export function setEngineCatalog(c: EngineCatalog | null) {
  active = c
}

/** An engine string in the current language (English when not translated). */
export function engineText(s: string): string
export function engineText(s: string | null | undefined): string | undefined
export function engineText(s: string | null | undefined): string | undefined {
  if (s == null) return undefined
  return active?.strings?.[s] ?? s
}

/** The label of a settings enum value, or undefined (the app formats the
 *  raw value itself). The value written to settings stays the raw one. */
export function engineEnum(value: string): string | undefined {
  return active?.enums?.[value]
}

/** An issue by its stable code: the catalog's sentence with `{surface}` and
 *  `{path}` filled, and the English message as detail (it holds the values the
 *  sentence leaves out); with no template, or a placeholder's value missing,
 *  the English message alone. */
export function engineIssue(code: string | null | undefined, fields: { surface?: number | null; path?: string | null }, message: string): { text: string; detail?: string } {
  const template = code ? active?.diagnostics?.[code] : undefined
  if (template) {
    const params: Params = {}
    if (fields.surface != null) params.surface = fields.surface
    if (fields.path != null) params.path = fields.path
    const out = fill(template, params, true)
    if (out != null) return { text: out, detail: message }
  }
  return { text: message }
}

/** A structural issue of an invalid model, `invalid model: [code] path: message`
 *  (or without the prefix): its parts, or null for other text. */
export function parseStructuralIssue(text: string): { code: string; path: string; message: string } | null {
  const m = /\[([a-z_]+)\] ([^:]+): (.*)/s.exec(text)
  return m ? { code: m[1], path: m[2], message: m[3] } : null
}
