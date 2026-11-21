// The `.mklight` project file (docs/FFI.md "Project file"): pretty-printed
// JSON with the system and the project's other sections. Unknown top-level
// keys are preserved on a round trip.

import type { Json, JsonObject } from '@/engine/types'
import samplesData from '@samples'
import type { System } from './system'
import { t } from '@/i18n'

export const APP_VERSION = '0.1.0-web'

export interface Project {
  system: System
  catalogs: Json[]
  views: Json[]
  merit: Json
  variables: Json
  tolerances: Json
  metadata: JsonObject
  /** Every other top-level key, kept as read. */
  extra: JsonObject
}

const KNOWN = new Set(['format', 'schema_version', 'metadata', 'system', 'catalogs', 'views', 'merit', 'variables', 'tolerances'])

export class ProjectError extends Error {}

export function readProject(text: string): Project {
  let j: unknown
  try {
    j = JSON.parse(text)
  } catch (e) {
    throw new ProjectError(t('app.project.invalidJson', { error: (e as Error).message }))
  }
  if (!j || typeof j !== 'object' || Array.isArray(j)) throw new ProjectError(t('app.project.notProject'))
  const o = j as JsonObject
  if (o.format !== 'mklight-project') throw new ProjectError(t('app.project.notProjectFormat'))
  if (o.schema_version !== 1) throw new ProjectError(t('app.project.unsupportedVersion', { version: String(o.schema_version) }))
  if (!o.system || typeof o.system !== 'object') throw new ProjectError(t('app.project.noSystem'))
  const extra: JsonObject = {}
  for (const [k, v] of Object.entries(o)) if (!KNOWN.has(k)) extra[k] = v
  return {
    system: o.system as System,
    catalogs: Array.isArray(o.catalogs) ? o.catalogs : [],
    views: Array.isArray(o.views) ? o.views : [],
    merit: o.merit ?? null,
    variables: o.variables ?? null,
    tolerances: o.tolerances ?? null,
    metadata: (o.metadata as JsonObject) ?? {},
    extra,
  }
}

export function writeProject(p: Project, engineVersion: string): string {
  const now = new Date().toISOString().replace(/\.\d+Z$/, 'Z')
  const metadata: JsonObject = {
    created: now,
    author: '',
    ...p.metadata,
    modified: now,
    app_name: 'MKLight',
    app_version: APP_VERSION,
    engine_version: engineVersion,
  }
  const out: JsonObject = {
    format: 'mklight-project',
    schema_version: 1,
    metadata,
    system: p.system,
    catalogs: p.catalogs,
    views: p.views,
    merit: p.merit,
    variables: p.variables,
    tolerances: p.tolerances,
    ...p.extra,
  }
  return JSON.stringify(out, null, 2) + '\n'
}

export function projectFromSystem(system: System): Project {
  return { system, catalogs: [], views: [], merit: null, variables: null, tolerances: null, metadata: {}, extra: {} }
}

// MARK: Samples

export interface Sample {
  id: string
  system: System
}

export const SAMPLES: Sample[] = samplesData as unknown as Sample[]

/** Samples with extended surface types are listed under "Surface Types". */
export function isSurfaceTypeSample(s: Sample): boolean {
  const basic = new Set(['plane', 'conic', 'sphere', 'even_asphere', 'coordinate_break'])
  return s.system.surfaces.some((x) => !basic.has(x.shape.type))
}

export function sampleById(id: string): Sample | undefined {
  return SAMPLES.find((s) => s.id === id)
}
