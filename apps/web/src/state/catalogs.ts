// Glass catalogs of a document, the browser's way. The macOS app records a
// catalog by its path and reads the file again when the project opens; a page
// cannot reopen a path, so the text of every catalog loaded here is kept in
// IndexedDB by name and loaded again from there when a project that lists it
// opens (docs/WEB.md "Differences"). A catalog never seen in this browser shows
// as not loaded, with Locate… to choose the file.
//
// Undoing "Add Glass Catalog" (or removing one) rebuilds the engine session
// without it, as the macOS app does: a fresh document engine with the remaining
// catalogs loaded again.

import { get as idbGet, set as idbSet } from 'idb-keyval'
import type { JsonObject } from '@/engine/types'
import type { Project } from '@/model/project'
import { useWorkspace } from './workspace'

const key = (name: string) => `mklight-catalog:${name.toUpperCase()}`

interface Stored {
  name: string
  fileName: string
  text: string
}

export async function rememberCatalog(name: string, fileName: string, text: string) {
  try {
    await idbSet(key(name), { name, fileName, text } satisfies Stored)
  } catch {
    // Storage may be refused (private windows); the catalog stays loaded for this session.
  }
}

async function stored(name: string): Promise<Stored | null> {
  try {
    return ((await idbGet(key(name))) as Stored | undefined) ?? null
  } catch {
    return null
  }
}

const namesOf = (p: Project): string[] =>
  (Array.isArray(p.catalogs) ? (p.catalogs as JsonObject[]) : []).map((c) => String(c.name ?? '')).filter(Boolean)

/** Loads the project's catalogs that the engine does not have yet, from this
 *  browser's store. Returns the names that could not be found. */
export async function restoreCatalogs(project: Project): Promise<string[]> {
  const ws = useWorkspace.getState()
  const engine = ws.engine
  if (!engine) return namesOf(project)
  let loaded: string[] = []
  try {
    loaded = (await engine.catalogs()).map((c) => c.name.toUpperCase())
  } catch {
    loaded = []
  }
  const missing: string[] = []
  let changed = false
  for (const name of namesOf(project)) {
    if (loaded.includes(name.toUpperCase())) continue
    const s = await stored(name)
    if (!s) {
      missing.push(name)
      continue
    }
    try {
      await engine.loadCatalog(s.fileName, s.text)
      changed = true
    } catch {
      missing.push(name)
    }
  }
  if (changed) {
    useWorkspace.setState({ materials: await engine.materialNames() })
    const sys = project.system
    if (sys?.surfaces) useWorkspace.getState().designChanged(sys)
  }
  return missing
}

/** After the project's catalog list changed: load new ones, and rebuild the
 *  session when one was dropped (undo of Add Glass Catalog, Remove). */
export async function catalogsChanged(prev: Project, next: Project) {
  const before = namesOf(prev).map((n) => n.toUpperCase())
  const after = namesOf(next).map((n) => n.toUpperCase())
  const dropped = before.some((n) => !after.includes(n))
  const ws = useWorkspace.getState()
  if (dropped && ws.engine) {
    const fresh = await ws.engine.makeDocumentEngine()
    if (fresh !== ws.engine) {
      await useWorkspace.getState().setEngine(fresh)
    }
  }
  await restoreCatalogs(next)
  const sys = next.system
  if (dropped && sys?.surfaces) useWorkspace.getState().designChanged(sys)
}
