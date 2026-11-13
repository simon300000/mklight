// Beam footprints for the 3-D view: the engine's `footprint` analysis on the
// snapshot the layout shows, its `global_outlines` per surface and field.

import { useEffect, useState } from 'react'
import type { AnalysisJob } from '@/engine/types'
import { activeSnapshot, useWorkspace } from '@/state/workspace'
import { readFootprints, type Footprint } from './scene'

const cache = new Map<string, Footprint[]>()

export function useFootprints(enabled: boolean): { footprints: Footprint[]; running: boolean; error: string | null } {
  const engine = useWorkspace((s) => s.engine)
  const snap = useWorkspace((s) => activeSnapshot(s))
  const [state, setState] = useState<{ key: string; footprints: Footprint[]; error: string | null } | null>(null)
  const key = snap?.hash ?? ''
  useEffect(() => {
    if (!enabled || !engine || !snap) return
    const hit = cache.get(key)
    if (hit) {
      setState({ key, footprints: hit, error: null })
      return
    }
    let job: AnalysisJob | null = null
    let live = true
    try {
      job = engine.submit({ analysis: 'footprint', settings: {} }, snap)
      job.result.then(
        (r) => {
          const f = readFootprints(r)
          cache.set(key, f)
          if (cache.size > 8) cache.delete(cache.keys().next().value!)
          if (live) setState({ key, footprints: f, error: f.length ? null : 'The engine drew no footprints.' })
        },
        (e) => live && setState({ key, footprints: [], error: (e as Error).message }),
      )
    } catch (e) {
      setState({ key, footprints: [], error: (e as Error).message })
    }
    return () => {
      live = false
      job?.cancel()
    }
  }, [enabled, engine, snap, key])
  if (!enabled) return { footprints: [], running: false, error: null }
  const current = state?.key === key ? state : null
  return { footprints: current?.footprints ?? [], running: !current, error: current?.error ?? null }
}
