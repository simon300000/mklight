// Text the engine writes (analysis titles and descriptions, settings help and
// choices, run warnings, issues, job stages, sample names, …) is translated
// through the shared catalog `locales/<lang>/engine.json` at the repo root:
// `{ "strings": { "<exact English>": "…" }, "diagnostics": { "<code>": "<template>" } }`.
// Exact-match lookup; a miss shows the English. The file is loaded with the
// language's UI catalog (a separate chunk), and the build tolerates its absence.

import { fill } from './fill'
import type { Params } from './types'

export interface EngineCatalog {
  strings?: Record<string, string>
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

/** An engine issue by its stable code: the catalog's template filled from
 *  `params` (the issue's own fields); the engine's English `message` when the
 *  code has no template or the template names a value the issue lacks. */
export function engineDiagnostic(code: string | null | undefined, params: Params, message: string): string {
  const template = code ? active?.diagnostics?.[code] : undefined
  if (template) {
    const out = fill(template, params, true)
    if (out != null) return out
  }
  return engineText(message)
}
